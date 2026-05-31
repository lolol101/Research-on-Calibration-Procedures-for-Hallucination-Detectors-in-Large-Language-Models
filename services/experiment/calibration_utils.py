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

from ..common.calculation_utils import calculate_feature_shap_values, calculate_ece_adaptive_bins, calculate_roc_auc
from ..common.calibration_heads import CalibrationHead
from ..common.logging_utils import log_data, make_next_indexed_log_filename
from ..index import IndexDataset

def find_best_layer_head_hal_dif_power(
    data: dict, 
    layers_count: int, 
    heads_count: int, 
    best_heads_group_size: int,
    verbose: bool = False
    ):
    """
    Selects top attention heads by hallucination–truth score gap (TOHA-style).

    For each (layer, head), computes mean attention score on incorrect minus
    correct answers, then returns the indices with the largest gaps.

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

    hal_dif_power_results = torch.stack(
        [
            data[f"attn_score{l}_{h}"][hallu_elem_ids].mean() - data[f"attn_score{l}_{h}"][truth_elem_ids].mean()
            for l in range(layers_count)
                for h in range(heads_count)
        ]
    )
    
    hal_dif_power_matrix = hal_dif_power_results \
        .reshape(layers_count, heads_count) \
        .cpu() \
        .to(dtype=torch.float32)

    best_score_idx = torch.argsort(hal_dif_power_results, descending=True)[:best_heads_group_size]
    
    if verbose:
        plt.figure(figsize=(5, 4))
        sns.heatmap(hal_dif_power_matrix, annot=False, cmap="Reds")
        plt.title("Hallucination difference power values by Layer and Head")
        plt.xlabel("Head ID")
        plt.ylabel("Layer ID")
        plt.gca().invert_yaxis()
        plt.show()
        print(f"Best metric value: {torch.max(hal_dif_power_results).item()}")
    
    return best_score_idx // layers_count, best_score_idx % heads_count

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
    
    return best_score_idx // layers_count, best_score_idx % heads_count


def test_calibration_model(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    eps: float = 1e-8,
):
    """
    Evaluates calibrated probabilities on a held-out set.

    Same metrics as the baseline helper: ECE, NLL, MSE, accuracy, inverse
    Brier skill score, and weighted ``ece+inv_bss`` (0.2 BSS + 0.8 ECE).

    Args:
        X_test: Predicted probabilities.
        y_test: Binary labels (0/1).
        device: Device for metric computation.
        verbose: Print metrics to stdout.
        logging: Write metrics to ``log_dir``.
        log_dir: Log directory; required when ``logging=True``.
        eps: Clipping bound for probabilities in NLL / BSS.

    Returns:
        Dict with keys ``ece+inv_bss``, ``ece``, ``inv_bss``, ``nlll``,
        ``mse``, and ``accuracy``.
    """
    if logging and not log_dir:
        raise ValueError("logging=True requires log_dir")
    
    ece_value = calculate_ece_adaptive_bins(
        X_test,
        y_test,
        n_bins=10,
        device=device,
        verbose=verbose,
        logging=logging,
        log_dir=log_dir
    )
    
    y_test_f = y_test.to(torch.float32)
    probs_clipped = torch.clamp(X_test, eps, 1 - eps).to(device)
    nlll = torch.nn.functional.binary_cross_entropy(probs_clipped, y_test_f)
    mse = torch.nn.functional.mse_loss(X_test, y_test_f)
    accuracy = torch.mean(y_test_f)
    
    p_ref = y_test_f.mean()
    brier_score_ref = torch.mean((p_ref - y_test_f) ** 2)
    brier_score = mse
    inv_brier_skill_score = (brier_score / brier_score_ref).item() if brier_score_ref > eps else float("nan")
    
    w_bss, w_ece = 0.2, 0.8
    ece_bss_weighted = w_bss * inv_brier_skill_score + w_ece * ece_value
    
    if verbose:
        print(f"ECE+BSS on calibrated answer (test data): {ece_bss_weighted}")
        print(f"ECE on calibrated answer (test data): {ece_value}")
        print(f"Inv Brier skill score on test data: {inv_brier_skill_score}")
        print(f"NLLL (binary cross-entropy) on test data: {nlll.item()}")
        print(f"MSE on test data: {mse.item()}")
        print(f"Accuracy  on test: {accuracy.item()}")
        
    if logging:
        log_data(
            data={
                "ece+inv_bss": ece_bss_weighted,
                "ece": ece_value,
                "inv_bss": inv_brier_skill_score,
                "nlll": nlll.item(),
                "mse": mse.item(),
                "accuracy": accuracy.item(),
            },
            log_dir=log_dir,
            prefix="calibration_metrics",
            extension=".txt",
            separator="=",
        )
            
    return {
        "ece+inv_bss": ece_bss_weighted,
        "ece": ece_value,
        "inv_bss": inv_brier_skill_score,
        "nlll": nlll.item(),
        "mse": mse.item(),
        "accuracy": accuracy.item(),
    }


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
    test: Optional[IndexDataset] = None,
    lr_max=1e-2,
    lr_min=1e-4,
    batch_size=64,
    epochs=3,
    plot_interval=3,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
):
    """
    Trains an experiment calibration head on a feature subset.

    Uses BCE on ``train_batch["features"][:, feature_ids]`` with optional
    L1/L2 regularization. AdamW + cosine schedule; optional loss curves.

    Args:
        model: Calibration module.
        train: Training ``IndexDataset`` with ``features`` and ``labels``.
        feature_ids: Column indices into the feature matrix.
        device: Device for tensors and the model.
        test: Optional validation set for loss logging.
        lr_max: Initial AdamW learning rate.
        lr_min: Cosine scheduler floor learning rate.
        batch_size: Mini-batch size.
        epochs: Training epochs.
        plot_interval: Log train/test loss every N steps.
        verbose: tqdm and matplotlib display.
        logging: Save loss plot to ``log_dir``.
        log_dir: Plot output directory.
        log_filename: Override default filename when logging.
        l1_lambda: L1 penalty strength on weights.
        l2_lambda: L2 penalty strength on weights.

    Returns:
        Trained ``model`` (mutated in place).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * (len(train) // batch_size), lr_min
    )

    train_losses = []
    test_losses = []
    iterations = []
    
    iteration_counter = 0
    for _ in tqdm(range(epochs), desc="Training (epochs)...", disable=not verbose):
        for start in range(0, len(train), batch_size):
            train_batch = train.get(start, min(start + batch_size, len(train)))
            if not train_batch:
                continue

            optimizer.zero_grad()

            train_batch_features = train_batch["features"][:, feature_ids].to(device=device, dtype=torch.float32)
            train_cal_confidence = model(train_batch_features)
            train_batch_labels = train_batch["labels"].to(device=device, dtype=torch.float32)
            
            train_loss, _, _, _ = _calibration_training_loss(
                train_cal_confidence,
                train_batch_labels,
                model,
                l1_lambda=l1_lambda,
                l2_lambda=l2_lambda,
            )

            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (
                test is not None
                and iteration_counter % plot_interval == 0
            ):
                train_total = train_loss.detach().item()
                train_losses.append(train_total)

                test_data = test.get()
                if test_data:
                    test_batch_features = test_data["features"][:, feature_ids].to(device=device, dtype=torch.float32)
                    test_batch_labels = test_data["labels"].to(device=device, dtype=torch.float32)
                    
                    with torch.no_grad():
                        test_cal_confidence = model(test_batch_features)
                        test_loss, _, _, _ = _calibration_training_loss(
                            test_cal_confidence,
                            test_batch_labels,
                            model,
                            l1_lambda=l1_lambda,
                            l2_lambda=l2_lambda,
                        )
                        test_losses.append(test_loss.item())
                    
                    iterations.append(iteration_counter)
            iteration_counter += 1

    if len(iterations) > 0 and (verbose or logging):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(iterations, train_losses, label="Train Loss", marker="o")
        if len(test_losses) > 0:
            ax.plot(iterations, test_losses, label="Test Loss", marker="s")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(
            f"Training and Test Loss (l1_lambda={l1_lambda}, l2_lambda={l2_lambda})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fname = log_filename or make_next_indexed_log_filename(
                log_dir=log_dir,
                prefix="calibration_fit_loss",
                extension=".png",
            )
            out_path = os.path.join(log_dir, fname)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
        if verbose:
            plt.show()
        plt.close(fig)

    return model

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
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    ):
    """
    Random search over calibration training hyperparameters.

    Samples trials from a grid (optionally including L1/L2), fits with
    ``fit_calibration_model``, scores via ECE and ``test_calibration_model``,
    and records SHAP attributions per trial.

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
        random_seed: RNG seed for shuffling/sampling the grid.
        verbose: Per-trial training and metric printing.
        logging: Log trials and best hyperparameters under ``log_dir``.

    Returns:
        Best trial dict: ``parameters``, ``hparameters``, ``ece+inv_bss``,
        ``shap_values``.
    """
    param_grid = {
        "lr_max": [1e-2, 5e-3, 2e-3, 1e-3],
        "lr_min": [1e-3, 5e-4, 2e-4, 1e-4],
        "batch_size": [16, 32],
        "epochs": [1, 3, 5, 10],
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

    results = []
    for trial_idx, sampled in enumerate(tqdm(sampled_candidates, disable=not verbose)):
        trial_log_dir = log_dir
        if logging and log_dir:
            trial_log_dir = os.path.join(log_dir, f"search_iter_{trial_idx + 1}")
            Path(trial_log_dir).mkdir(parents=True, exist_ok=True)

        lr_max = float(sampled["lr_max"])
        lr_min = float(sampled["lr_min"])
        batch_size = int(sampled["batch_size"])
        epochs = int(sampled["epochs"])
        l1_lambda = float(sampled["l1_lambda"])
        l2_lambda = float(sampled["l2_lambda"])

        model = fit_calibration_model(
            model_class(
                in_features=(features_count + 1) * (heads_count + (not attn_only)),
                device=device
            ),
            train=train,
            test=test,
            feature_ids=feature_ids,
            lr_max=lr_max,
            lr_min=lr_min,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=trial_log_dir,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )

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
                    "epochs": epochs,
                    "l1_lambda": l1_lambda,
                    "l2_lambda": l2_lambda,
                },
                "ece+inv_bss": metrics["ece+inv_bss"],
                "shap_values": trial_shap_values,
            }
        )

    best_result = min(results, key=lambda x: x["ece+inv_bss"])
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
                "ece+inv_bss": best_result["ece+inv_bss"],
                **best_result["hparameters"],
            },
            log_dir=log_dir,
            prefix="best_model_hparameters",
            extension=".txt",
            separator="=",
        )
    return best_result