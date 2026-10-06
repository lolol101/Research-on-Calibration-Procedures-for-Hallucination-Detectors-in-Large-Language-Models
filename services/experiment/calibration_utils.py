import os
from pathlib import Path
from typing import Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import torch
from torch import nn
from tqdm import tqdm

from ..common.calculation_utils import (
    calculate_calibration_metrics,
    calculate_ece_adaptive_bins,
    calculate_feature_shap_values,
    calculate_metrics_bootstrap_ci,
    calculate_roc_auc,
)
from ..common.calibration_heads import CalibrationHead
from ..common.training_utils import fit_stacked_with_early_stopping, fit_with_early_stopping
from ..common.logging_utils import log_data
from ..index import IndexDataset

def find_best_layer_head_hdp(
    data: dict, 
    layers_count: int, 
    heads_count: int, 
    best_heads_group_size: int,
    verbose: bool = False
    ):
    """
    Selects top attention heads by hallucination–truth score gap (TOHA-style).

    For each (layer, head), computes mean attention score on incorrect minus
    correct answers, divided by the pooled standard deviation of both groups
    (Cohen's d), then returns the indices with the largest gaps. Without the
    division, heads whose entropy is large and widely spread rank high even
    when the two groups overlap.

    Args:
        data: Dict with ``labels`` and ``attn_score{l}_{h}`` tensors per head.
        layers_count: Number of transformer layers.
        heads_count: Number of heads per layer.
        best_heads_group_size: How many (layer, head) pairs to return.
        verbose: Plot a layer×head heatmap and print the best gap.

    Returns:
        Tuple ``(best_layers, best_heads)`` index tensors of length
        ``best_heads_group_size``.
    """
    hallu_elem_ids = torch.argwhere(data["labels"] == False)
    truth_elem_ids = torch.argwhere(data["labels"] == True)

    hallu_count, truth_count = len(hallu_elem_ids), len(truth_elem_ids)

    def cohens_d(scores):
        hallu_scores, truth_scores = scores[hallu_elem_ids], scores[truth_elem_ids]
        pooled_var = (
            (hallu_count - 1) * hallu_scores.var() + (truth_count - 1) * truth_scores.var()
        ) / (hallu_count + truth_count - 2)
        return (hallu_scores.mean() - truth_scores.mean()) / (pooled_var.sqrt() + 1e-12)

    hdp_results = torch.stack(
        [
            cohens_d(data[f"attn_score{l}_{h}"])
            for l in range(layers_count)
                for h in range(heads_count)
        ]
    )
    
    hdp_matrix = hdp_results \
        .reshape(layers_count, heads_count) \
        .cpu() \
        .to(dtype=torch.float32)

    best_score_idx = torch.argsort(hdp_results, descending=True)[:best_heads_group_size]
    
    if verbose:
        plt.figure(figsize=(5, 4))
        sns.heatmap(hdp_matrix, annot=False, cmap="Reds")
        plt.title("Hallucination difference power values by Layer and Head")
        plt.xlabel("Head ID")
        plt.ylabel("Layer ID")
        plt.gca().invert_yaxis()
        plt.show()
        print(f"Best metric value: {torch.max(hdp_results).item()}")
    
    return best_score_idx // heads_count, best_score_idx % heads_count

def find_best_layer_head_roc_auc(
    data: dict,
    layers_count: int,
    heads_count: int,
    best_heads_group_size: int,
    verbose: bool = False
    ):
    """
    Selects top attention heads by ROC AUC separating incorrect answers.

    Args:
        data: Dict with ``labels`` and ``attn_score{l}_{h}`` tensors per head.
        layers_count: Number of transformer layers.
        heads_count: Number of heads per layer.
        best_heads_group_size: How many (layer, head) pairs to return.
        verbose: Plot a layer×head heatmap and print the best AUC.

    Returns:
        Tuple ``(best_layers, best_heads)`` index tensors of length
        ``best_heads_group_size``.
    """
    roc_auc_results = torch.stack(
        [
            torch.tensor(calculate_roc_auc(
                data[f"attn_score{l}_{h}"], 
                1 - data[f"labels"]
            ))
            for l in range(layers_count)
                for h in range(heads_count)
        ]
    )

    roc_auc_matrix = roc_auc_results.reshape(layers_count, heads_count).cpu()
    best_score_idx = torch.argsort(roc_auc_results, descending=True)[:best_heads_group_size]

    if verbose:
        plt.figure(figsize=(5, 4))
        sns.heatmap(roc_auc_matrix, annot=False, cmap="Reds")
        plt.title("ROC AUC values by Layer and Head")
        plt.xlabel("Head ID")
        plt.ylabel("Layer ID")
        plt.gca().invert_yaxis()
        plt.show()
        print(f"Best metric value: {torch.max(roc_auc_results).item()}")
    
    return best_score_idx // heads_count, best_score_idx % heads_count

def find_random_layer_head(
    data: dict,
    layers_count: int,
    heads_count: int,
    best_heads_group_size: int,
    seed: int = 0,
    verbose: bool = False
    ):
    """
    Draws (layer, head) pairs uniformly at random, ignoring ``data``.

    Ablation for the HDP and ROC AUC criteria: if randomly drawn heads
    calibrate as well as selected ones, the selection step adds nothing.
    Leading pairs are kept for smaller groups, so groups are nested as for
    the ranked criteria.

    Args:
        data: Head-selection data; unused, kept for the selector interface.
        layers_count: Number of transformer layers.
        heads_count: Number of heads per layer.
        best_heads_group_size: How many (layer, head) pairs to return.
        seed: Seed of the draw.
        verbose: Unused, kept for the selector interface.

    Returns:
        Tuple ``(best_layers, best_heads)`` index tensors of length
        ``best_heads_group_size``.
    """
    generator = torch.Generator().manual_seed(seed)
    drawn_idx = torch.randperm(layers_count * heads_count, generator=generator)[:best_heads_group_size]
    return drawn_idx // heads_count, drawn_idx % heads_count


def test_calibration_model(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    eps: float = 1e-6,
    bootstrap: bool = False,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
    save_predictions: bool = False,
):
    """
    Evaluates calibrated probabilities on a held-out set.

    Same metrics as the baseline helper, from ``calculate_calibration_metrics``.
    Optionally adds percentile bootstrap confidence intervals.

    Args:
        X_test: Predicted probabilities.
        y_test: Binary labels (0/1).
        device: Device for metric computation.
        verbose: Print metrics to stdout.
        logging: Write metrics to ``log_dir``.
        log_dir: Log directory; required when ``logging=True``.
        eps: Clipping bound for probabilities in NLL / BSS.
        bootstrap: If True, also report confidence intervals.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage of the reported intervals.
        random_seed: Seed for the resampling RNG.
        save_predictions: If True and ``logging=True``, also save ``X_test``
            and ``y_test`` to ``test_predictions.pt`` in ``log_dir``.

    Returns:
        Dict of the ``calculate_calibration_metrics`` metrics, plus
        ``{metric}_ci_low`` and ``{metric}_ci_high`` entries when
        ``bootstrap=True``.
    """
    if logging and not log_dir:
        raise ValueError("logging=True requires log_dir")

    metrics = calculate_calibration_metrics(
        X_test,
        y_test,
        device=device,
        eps=eps,
        verbose=verbose,
        logging=logging,
        log_dir=log_dir,
    )

    if bootstrap:
        intervals = calculate_metrics_bootstrap_ci(
            X_test,
            y_test,
            device=device,
            n_resamples=n_resamples,
            confidence=confidence,
            random_seed=random_seed,
            eps=eps,
            verbose=verbose,
        )
        for name, (low, high) in intervals.items():
            metrics[f"{name}_ci_low"] = low
            metrics[f"{name}_ci_high"] = high

    if verbose:
        for name in ("ece", "ce_debiased", "rel", "res", "inv_bss", "nlll", "auroc", "accuracy"):
            line = f"{name} on test data: {metrics[name]}"
            if bootstrap:
                line += (
                    f"  CI [{metrics[f'{name}_ci_low']:.4f}, "
                    f"{metrics[f'{name}_ci_high']:.4f}]"
                )
            print(line)

    if logging:
        log_data(
            data=metrics,
            log_dir=log_dir,
            prefix="calibration_metrics",
            extension=".txt",
            separator="=",
        )
        if save_predictions:
            torch.save(
                {"probs": X_test.detach().cpu(), "labels": y_test.detach().cpu()},
                os.path.join(log_dir, "test_predictions.pt"),
            )

    return metrics


def _calibration_training_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    *,
    l1_lambda: float,
    l2_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Binary cross-entropy plus optional L1/L2 weight penalties.

    Args:
        pred: Model predictions.
        labels: Float targets in [0, 1].
        model: Module whose parameters are penalized.
        l1_lambda: L1 coefficient (0 disables).
        l2_lambda: L2 coefficient (0 disables).

    Returns:
        Tuple ``(total, bce, l1_loss, l2_loss)`` tensors.
    """
    bce = torch.nn.functional.binary_cross_entropy(pred, labels)
    l1_loss = bce * 0
    if l1_lambda > 0:
        l1_penalty = sum(p.abs().sum() for p in model.parameters())
        l1_loss = l1_lambda * l1_penalty
    l2_loss = bce * 0
    if l2_lambda > 0:
        l2_penalty = sum((p**2).sum() for p in model.parameters())
        l2_loss = l2_lambda * l2_penalty
    total = bce + l1_loss + l2_loss
    return total, bce, l1_loss, l2_loss


def fit_calibration_model(
    model: nn.Module,
    train: IndexDataset,
    feature_ids: torch.Tensor,
    device: torch.device,
    test: IndexDataset,
    lr_max=1e-2,
    lr_min=1e-4,
    batch_size=64,
    max_epochs=50,
    patience=3,
    shuffle_seed: Optional[int] = None,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
):
    """
    Trains an experiment calibration head on a feature subset.

    Uses BCE on ``features[:, feature_ids]`` with optional L1/L2
    regularization, stopping early on the unregularized validation BCE (see
    ``fit_with_early_stopping``).

    Args:
        model: Calibration module.
        train: Training ``IndexDataset`` with ``features`` and ``labels``.
        feature_ids: Column indices into the feature matrix.
        device: Device for tensors and the model.
        test: Validation ``IndexDataset`` used for early stopping.
        lr_max: Initial learning rate for AdamW.
        lr_min: Minimum learning rate for the cosine scheduler.
        batch_size: Mini-batch size over ``train``.
        max_epochs: Upper bound on the number of passes over ``train``.
        patience: Epochs without validation improvement before stopping.
        shuffle_seed: Seed for the per-epoch batch order.
        verbose: Show tqdm progress and the loss plot.
        logging: Save the loss plot to ``log_dir``.
        log_dir: Output directory for logged results.
        log_filename: Override auto-generated files names when logging.
        l1_lambda: L1 penalty strength on weights.
        l2_lambda: L2 penalty strength on weights.

    Returns:
        Tuple ``(model, best_epoch)``: the trained ``model`` (mutated in
        place) and the epoch whose weights it holds.
    """
    train_data, val_data = train.get(), test.get()

    def loss_fn(model, features, labels):
        total, bce, _, _ = _calibration_training_loss(
            model(features),
            labels,
            model,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )
        return total, bce

    best_epoch = fit_with_early_stopping(
        model,
        train_data["features"][:, feature_ids].to(device=device, dtype=torch.float32),
        train_data["labels"].to(device=device, dtype=torch.float32),
        val_data["features"][:, feature_ids].to(device=device, dtype=torch.float32),
        val_data["labels"].to(device=device, dtype=torch.float32),
        loss_fn,
        lr_max=lr_max,
        lr_min=lr_min,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        shuffle_seed=shuffle_seed,
        verbose=verbose,
        logging=logging,
        log_dir=log_dir,
        log_filename=log_filename,
    )
    return model, best_epoch

def fit_hparameters(
    model_class: CalibrationHead,
    train: IndexDataset,
    test: IndexDataset,
    features_count: int,
    device: torch.device,
    feature_ids: Optional[torch.Tensor] = None,
    attn_only: bool = False,
    heads_count=15,
    search_trials=20,
    l1_reg=False,
    l2_reg=False,
    random_seed: Optional[int] = None,
    max_epochs=50,
    patience=3,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    ):
    """
    Random search over calibration training hyperparameters.

    Samples trials from a grid (optionally including L1/L2), fits them,
    scores via ECE and ``test_calibration_model``, and records SHAP
    attributions per trial. Every head standardizes its inputs with the mean
    and standard deviation of the training features, kept in its state dict.
    On CUDA the trials of each batch size are trained
    jointly with ``fit_stacked_with_early_stopping`` (no per-trial loss
    plots); otherwise each trial is fitted with ``fit_calibration_model``.

    Args:
        model_class: ``CalibrationHead`` subclass to instantiate.
        train: Training ``IndexDataset``.
        test: Validation ``IndexDataset``.
        features_count: Per-head feature count before broadcasting.
        device: Training and metric device.
        feature_ids: Column subset of ``features``; required at call time.
        attn_only: If True, input dim uses attention heads only (no +1 final).
        heads_count: Number of selected attention heads in the input dim.
        search_trials: Number of grid samples to evaluate.
        l1_reg: Include L1 penalty values in the search grid.
        l2_reg: Include L2 penalty values in the search grid.
        random_seed: RNG seed for shuffling/sampling the grid and the batch order.
        max_epochs: Upper bound on training epochs per trial.
        patience: Epochs without validation improvement before stopping.
        verbose: Per-trial training and metric printing.
        logging: Log trials and best hyperparameters under ``log_dir``.

    Returns:
        Dict of the trial with the lowest validation ``inv_bss``:
        ``parameters``, ``hparameters``, ``inv_bss``, ``val_metrics``,
        ``shap_values``.
    """
    param_grid = {
        "lr_max": [1e-2, 5e-3, 2e-3, 1e-3],
        "lr_min": [1e-3, 5e-4, 2e-4, 1e-4],
        "batch_size": [16, 32],
        "l1_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2] if l1_reg else [0.0],
        "l2_lambda": [0.0, 1e-5, 1e-4, 1e-3, 1e-2] if l2_reg else [0.0],
    }
    all_candidates = list(ParameterGrid(param_grid))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(all_candidates)
    if search_trials <= len(all_candidates):
        sampled_candidates = all_candidates[:search_trials]
    else:
        sampled_candidates = list(all_candidates)
        extra_ids = rng.integers(0, len(all_candidates), size=search_trials - len(all_candidates))
        sampled_candidates.extend([all_candidates[i] for i in extra_ids.tolist()])

    trial_log_dirs = []
    for trial_idx in range(len(sampled_candidates)):
        trial_log_dir = log_dir
        if logging and log_dir:
            trial_log_dir = os.path.join(log_dir, f"search_iter_{trial_idx + 1}")
            Path(trial_log_dir).mkdir(parents=True, exist_ok=True)
        trial_log_dirs.append(trial_log_dir)

    # Created in trial order, so both training paths below start every trial
    # from the same initial weights.
    models = [
        model_class(
            in_features=(features_count + 1) * (heads_count + (not attn_only)),
            device=device
        )
        for _ in sampled_candidates
    ]
    # Inputs are standardized with train statistics; constant columns are only centred.
    train_features = train.get()["features"][:, feature_ids].to(device=device, dtype=torch.float32)
    input_std = train_features.std(0)
    input_std = torch.where(input_std > 1e-8, input_std, torch.ones_like(input_std))
    for model in models:
        model.set_input_scaling(train_features.mean(0), input_std)
    best_epochs = [0] * len(sampled_candidates)

    if device.type == "cuda":
        # Trials that share a batch size are trained jointly: on a GPU one
        # stacked step costs about as much as a single tiny model's step.
        train_data, val_data = train.get(), test.get()
        for batch_size in sorted({int(sampled["batch_size"]) for sampled in sampled_candidates}):
            trial_ids = [
                trial_idx
                for trial_idx, sampled in enumerate(sampled_candidates)
                if int(sampled["batch_size"]) == batch_size
            ]
            group_epochs = fit_stacked_with_early_stopping(
                [models[trial_idx] for trial_idx in trial_ids],
                train_data["features"][:, feature_ids].to(device=device, dtype=torch.float32),
                train_data["labels"].to(device=device, dtype=torch.float32),
                val_data["features"][:, feature_ids].to(device=device, dtype=torch.float32),
                val_data["labels"].to(device=device, dtype=torch.float32),
                lr_max=[float(sampled_candidates[i]["lr_max"]) for i in trial_ids],
                lr_min=[float(sampled_candidates[i]["lr_min"]) for i in trial_ids],
                l1_lambda=[float(sampled_candidates[i]["l1_lambda"]) for i in trial_ids],
                l2_lambda=[float(sampled_candidates[i]["l2_lambda"]) for i in trial_ids],
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                shuffle_seed=random_seed,
            )
            for trial_idx, best_epoch in zip(trial_ids, group_epochs):
                best_epochs[trial_idx] = best_epoch
    else:
        for trial_idx, sampled in enumerate(tqdm(sampled_candidates, disable=not verbose)):
            _, best_epochs[trial_idx] = fit_calibration_model(
                models[trial_idx],
                train=train,
                test=test,
                feature_ids=feature_ids,
                lr_max=float(sampled["lr_max"]),
                lr_min=float(sampled["lr_min"]),
                batch_size=int(sampled["batch_size"]),
                max_epochs=max_epochs,
                patience=patience,
                shuffle_seed=random_seed,
                device=device,
                verbose=verbose,
                logging=logging,
                log_dir=trial_log_dirs[trial_idx],
                l1_lambda=float(sampled["l1_lambda"]),
                l2_lambda=float(sampled["l2_lambda"]),
            )

    results = []
    for trial_idx, (model, sampled) in enumerate(zip(models, sampled_candidates)):
        trial_log_dir = trial_log_dirs[trial_idx]
        lr_max = float(sampled["lr_max"])
        lr_min = float(sampled["lr_min"])
        batch_size = int(sampled["batch_size"])
        l1_lambda = float(sampled["l1_lambda"])
        l2_lambda = float(sampled["l2_lambda"])
        best_epoch = best_epochs[trial_idx]

        test_data = test.get()
        test_data_features = test_data.get("features")[:, feature_ids].to(device=device, dtype=torch.float32)
            
        test_calibrated_probs = model.calibrate(test_data_features, device)    

        ece = calculate_ece_adaptive_bins(
            token_probs=test_calibrated_probs,
            labels=test_data["labels"],
            device=device,
            verbose=verbose
        )

        trial_shap_values = calculate_feature_shap_values(
            model=model,
            features=test_data_features,
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=trial_log_dir
        )
        
        metrics = test_calibration_model(
            X_test=test_calibrated_probs,
            y_test=test_data["labels"],
            device=device,
            logging=logging,
            log_dir=trial_log_dir
        )
        
        if verbose:
            print(f"Current ECE: {ece}")
            
        results.append(
            {
                "parameters": model.state_dict(),
                "hparameters": {
                    "lr_max": lr_max,
                    "lr_min": lr_min,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                    "l1_lambda": l1_lambda,
                    "l2_lambda": l2_lambda,
                },
                "inv_bss": metrics["inv_bss"],
                "val_metrics": metrics,
                "shap_values": trial_shap_values,
            }
        )

    best_result = min(results, key=lambda x: x["inv_bss"])
    if logging and log_dir:
        best_shap_values = best_result.get("shap_values")
        if best_shap_values is not None:
            feature_indices = torch.arange(len(best_shap_values), dtype=torch.long)
            sort_idx = torch.argsort(best_shap_values, descending=True)
            log_data(
                data={
                    str(feature_indices[idx].item()): f"{best_shap_values[idx].item():.6f}"
                    for idx in sort_idx.tolist()
                },
                log_dir=log_dir,
                prefix="calibration_feature_shap",
                extension=".txt",
                separator="\t",
            )

        log_data(
            data={
                "inv_bss": best_result["inv_bss"],
                **best_result["hparameters"],
                **{f"val_{name}": value for name, value in best_result["val_metrics"].items()},
            },
            log_dir=log_dir,
            prefix="best_model_hparameters",
            extension=".txt",
            separator="=",
        )
    return best_result