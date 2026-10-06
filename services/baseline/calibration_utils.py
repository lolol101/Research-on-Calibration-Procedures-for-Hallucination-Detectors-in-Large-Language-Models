import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
from ..common.training_utils import fit_with_early_stopping
from ..common.logging_utils import log_data
from ..index import IndexDataset

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
    Calculates set of metrics on the calibrated probabilities.

    Computes the metrics of ``calculate_calibration_metrics`` (ECE, the
    debiased Brier decomposition, inverse Brier skill score, NLL, MSE, AUROC
    and accuracy). Optionally adds percentile bootstrap confidence intervals
    for every metric.

    Args:
        X_test: Predicted probabilities.
        y_test: Binary labels (0/1).
        device: torch.device to perform computations on.
        verbose: If True, print metrics to stdout.
        logging: If True, write metrics to ``log_dir``.
        log_dir: Directory for log files; required when ``logging=True``.
        eps: Clipping bound for probabilities in NLL / BSS reference.
        bootstrap: If True, also report confidence intervals.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage of the reported intervals.
        random_seed: Seed for the resampling RNG.
        save_predictions: If True and ``logging=True``, also save ``X_test``
            and ``y_test`` to ``test_predictions.pt`` in ``log_dir``.

    Returns:
        Dict of the ``calculate_calibration_metrics`` metrics (scalar
        floats), plus ``{metric}_ci_low`` and ``{metric}_ci_high`` entries
        when ``bootstrap=True``.
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


def fit_calibration_model_beta(
    model: nn.Module,
    train: IndexDataset,
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
):
    """
    Trains a beta-calibration head.

    Optimizes binary cross-entropy between model outputs on ``features`` and
    ``labels``, stopping early on the validation loss (see
    ``fit_with_early_stopping``).

    Args:
        model: Calibration module mapping features to probabilities.
        train: Training ``IndexDataset`` with ``features`` and ``labels``.
        device: torch.device to perform computations on.
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

    Returns:
        Tuple ``(model, best_epoch)``: the trained ``model`` (mutated in
        place) and the epoch whose weights it holds.
    """
    train_data, val_data = train.get(), test.get()

    def loss_fn(model, features, labels):
        loss = torch.nn.functional.binary_cross_entropy(model(features), labels)
        return loss, loss

    best_epoch = fit_with_early_stopping(
        model,
        train_data["features"].to(device=device, dtype=torch.float32),
        train_data["labels"].to(device=device, dtype=torch.float32),
        val_data["features"].to(device=device, dtype=torch.float32),
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

def fit_calibration_model_temp(
    model: nn.Module,
    train: IndexDataset,
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
):
    """
    Train a temperature-scaling calibration head on logits.

    Optimizes cross-entropy on temperature-scaled ``logits`` against
    ``answer_tok_ids``, stopping early on the validation loss (see
    ``fit_with_early_stopping``).

    Args:
        model: Temperature-scaled calibration head.
        train: Training ``IndexDataset`` with ``logits`` and ``answer_tok_ids``.
        device: torch.device to perform computations on.
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

    Returns:
        Tuple ``(model, best_epoch)``: the trained ``model`` (mutated in
        place) and the epoch whose weights it holds.
    """
    train_data, val_data = train.get(), test.get()

    def loss_fn(model, logits, answer_tok_ids):
        loss = torch.nn.functional.cross_entropy(model.scale_logits(logits), answer_tok_ids)
        return loss, loss

    best_epoch = fit_with_early_stopping(
        model,
        train_data["logits"].to(device=device, dtype=torch.float32),
        train_data["answer_tok_ids"].to(device=device, dtype=torch.long),
        val_data["logits"].to(device=device, dtype=torch.float32),
        val_data["answer_tok_ids"].to(device=device, dtype=torch.long),
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

def fit_hparameters_beta(
    model_class: CalibrationHead,
    train: IndexDataset,
    test: IndexDataset,
    features_count: int,
    device: torch.device,
    search_trials=20,
    random_seed: Optional[int] = None,
    max_epochs=50,
    patience=3,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    ):
    """
    Random search over training hyperparameters for beta calibration.

    Samples up to ``search_trials`` configs from a grid over learning rates,
    and batch size; fits each with ``fit_calibration_model_beta`` and
    scores on ``test`` and ``test_calibration_model``. Also computes
    SHAP-like feature attributions per trial.

    Args:
        model_class: ``CalibrationHead`` subclass to instantiate per trial.
        train: Training ``IndexDataset``.
        test: Validation ``IndexDataset``.
        features_count: Input dimension.
        device: torch.device to perform computations on.
        search_trials: Number of hyperparameter combinations to try.
        random_seed: Seed for shuffling/sampling the grid and the batch order;
            None for nondeterministic.
        max_epochs: Upper bound on training epochs per trial.
        patience: Epochs without validation improvement before stopping.
        verbose: Show trial ECE-diagram and enable nested training verbosity.
        logging: Log per-trial outputs.
        log_dir: Root directory; each trial uses ``search_iter_XXX`` subfolders by deafualt.

    Returns:
        Dict with ``hparameters``, best model ``parameters`` (state dict),
        validation ``inv_bss`` and ``shap_values`` from the trial with the
        lowest validation ``inv_bss``.
    """
    param_grid = {
        "lr_max": [1e-2, 5e-3, 2e-3, 1e-3],
        "lr_min": [1e-3, 5e-4, 2e-4, 1e-4],
        "batch_size": [16, 32],
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
            trial_log_dir = os.path.join(log_dir, f"search_iter_{trial_idx + 1:03d}")
            Path(trial_log_dir).mkdir(parents=True, exist_ok=True)

        lr_max = float(sampled["lr_max"])
        lr_min = float(sampled["lr_min"])
        batch_size = int(sampled["batch_size"])

        model, best_epoch = fit_calibration_model_beta(
            model_class(
                in_features=features_count + 1,
                device=device
            ),
            train=train,
            test=test,
            lr_max=lr_max,
            lr_min=lr_min,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            shuffle_seed=random_seed,
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=trial_log_dir,
        )

        test_data = test.get()
        test_data_features = test_data.get("features")
            
        val_calibrated_probs = model.calibrate(test_data_features, device)   
        ece = calculate_ece_adaptive_bins(
            token_probs=val_calibrated_probs,
            labels=test_data["labels"],
            device=device,
            verbose=verbose
        )
        
        metrics = test_calibration_model(
            X_test=val_calibrated_probs,
            y_test=test_data["labels"],
            device=device,
            logging=logging,
            log_dir=trial_log_dir
        )

        trial_shap_values = calculate_feature_shap_values(
            model=model,
            features=test_data_features.to(device=device, dtype=torch.float32),
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=trial_log_dir,
        )
        
        if verbose:
            print(f"Current ECE: {ece}")
            
        results.append(
            {
                "hparameters": {
                    "lr_max": lr_max,
                    "lr_min": lr_min,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                },
                "parameters": model.state_dict(),
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
                    str(feature_indices[idx].item()): f"{best_shap_values[idx].item():.10f}"
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

def fit_hparameters_temp(
    model_class: CalibrationHead,
    train: IndexDataset,
    test: IndexDataset,
    features_count: int,
    device: torch.device,
    search_trials=20,
    random_seed: Optional[int] = None,
    max_epochs=50,
    patience=3,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    ):
    """Random search over training hyperparameters for temperature calibration.

    Same search procedure as ``fit_hparameters_beta``, but uses
    ``fit_calibration_model_temp`` and evaluates calibrated token probabilities
    at ``gen_tok_ids`` positions before computing the metrics.

    Args:
        model_class: ``CalibrationHead`` subclass to instantiate per trial.
        train: Training ``IndexDataset``.
        test: Validation ``IndexDataset``.
        features_count: Input dimension.
        device: torch.device to perform computations on.
        search_trials: Number of hyperparameter combinations to try.
        random_seed: Seed for shuffling/sampling the grid and the batch order;
            None for nondeterministic.
        max_epochs: Upper bound on training epochs per trial.
        patience: Epochs without validation improvement before stopping.
        verbose: Show trial ECE-diagram and enable nested training verbosity.
        logging: Log per-trial outputs.
        log_dir: Root directory; each trial uses ``search_iter_XXX`` subfolders by deafualt.

    Returns:
        Dict with ``hparameters``, best model ``parameters`` (state dict),
        validation ``inv_bss`` and ``shap_values`` from the trial with the
        lowest validation ``inv_bss``.
    """
    param_grid = {
        "lr_max": [1e-2, 5e-3, 2e-3, 1e-3],
        "lr_min": [1e-3, 5e-4, 2e-4, 1e-4],
        "batch_size": [16, 32],
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
            trial_log_dir = os.path.join(log_dir, f"search_iter_{trial_idx + 1:03d}")
            Path(trial_log_dir).mkdir(parents=True, exist_ok=True)

        lr_max = float(sampled["lr_max"])
        lr_min = float(sampled["lr_min"])
        batch_size = int(sampled["batch_size"])

        model, best_epoch = fit_calibration_model_temp(
            model_class(
                in_features=features_count + 1,
                device=device
            ),
            train=train,
            test=test,
            lr_max=lr_max,
            lr_min=lr_min,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            shuffle_seed=random_seed,
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=trial_log_dir,
        )

        test_data = test.get()
        test_data_features = test_data.get("logits")
            
        val_calibrated_probs = model.calibrate(test_data_features, device)
        val_calibrated_probs = val_calibrated_probs.gather(
            1, test_data["gen_tok_ids"].unsqueeze(1)
        ).squeeze(1)

        ece = calculate_ece_adaptive_bins(
            token_probs=val_calibrated_probs,
            labels=test_data["labels"],
            device=device,
            verbose=verbose,
        )
        
        metrics = test_calibration_model(
            X_test=val_calibrated_probs,
            y_test=test_data["labels"],
            device=device,
            logging=logging,
            log_dir=trial_log_dir
        )
        
        if verbose:
            print(f"Current ECE: {ece}")
            
        results.append(
            {
                "hparameters": {
                    "lr_max": lr_max,
                    "lr_min": lr_min,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                },
                "parameters": model.state_dict(),
                "inv_bss": metrics["inv_bss"],
                "val_metrics": metrics,
            }
        )

    best_result = min(results, key=lambda x: x["inv_bss"])
    if logging and log_dir:
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


def fit_logistic_regression(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    c_grid=(1e-3, 1e-2, 1e-1, 1.0),
    penalty: Literal["l1", "l2"] = "l1",
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Logistic regression on standardized features, with ``C`` chosen on validation.

    Used for Platt scaling (one feature, weak L2 penalty) and for the
    no-selection baseline (all attention heads, L1 penalty). Every ``C`` in
    ``c_grid`` is fitted on the training set; the one with the lowest
    validation ``inv_bss`` is kept, as for the gradient-trained heads.

    Args:
        X_train: Training features, shape ``[B, F]``.
        y_train: Training labels (0/1), shape ``[B]``.
        X_val: Validation features, shape ``[B_val, F]``.
        y_val: Validation labels (0/1), shape ``[B_val]``.
        device: torch.device for metric computation.
        c_grid: Inverse regularization strengths to try.
        penalty: ``"l1"`` or ``"l2"``.
        logging: If True, write the chosen ``C`` and validation metrics to ``log_dir``.
        log_dir: Directory for the log; required when ``logging=True``.

    Returns:
        Dict with the fitted ``model`` (scikit-learn pipeline), its ``C`` and
        ``val_metrics``.

    Raises:
        ValueError: If ``logging=True`` and ``log_dir`` is not set.
    """
    if logging and not log_dir:
        raise ValueError("log_dir must be set when logging=True")

    results = []
    for c in c_grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty=penalty, C=c, solver="liblinear", max_iter=1000, random_state=0),
        )
        model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
        val_probs = torch.from_numpy(model.predict_proba(X_val.cpu().numpy())[:, 1]).to(torch.float32)
        metrics = calculate_calibration_metrics(val_probs, y_val.cpu(), device=device)
        results.append({"model": model, "C": c, "val_metrics": metrics})

    best_result = min(results, key=lambda x: x["val_metrics"]["inv_bss"])
    if logging:
        log_data(
            data={
                "inv_bss": best_result["val_metrics"]["inv_bss"],
                "C": best_result["C"],
                **{f"val_{name}": value for name, value in best_result["val_metrics"].items()},
            },
            log_dir=log_dir,
            prefix="best_model_hparameters",
            extension=".txt",
            separator="=",
        )
    return best_result
