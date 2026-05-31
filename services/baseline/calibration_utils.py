import os
from pathlib import Path
from typing import Literal, Optional

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import torch
from torch import nn
from tqdm import tqdm

from ..common.calculation_utils import calculate_ece_adaptive_bins, calculate_feature_shap_values, calculate_roc_auc
from ..common.calibration_heads import CalibrationHead
from ..common.logging_utils import log_data, make_next_indexed_log_filename
from ..index import IndexDataset

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
    Calculates set of metrics on the calibrated probabilities.

    Computes ECE (adaptive bins), negative log-likelihood, MSE, accuracy,
    inverse Brier skill score, and a weighted composite ``ece+inv_bss``
    (0.2 * inv_bss + 0.8 * ece).

    Args:
        X_test: Predicted probabilities.
        y_test: Binary labels (0/1).
        device: torch.device to perform computations on.
        verbose: If True, print metrics to stdout.
        logging: If True, write metrics to ``log_dir``.
        log_dir: Directory for log files; required when ``logging=True``.
        eps: Clipping bound for probabilities in NLL / BSS reference.

    Returns:
        Dict with keys ``ece+inv_bss``, ``ece``, ``inv_bss``, ``nlll``,
        ``mse``, and ``accuracy`` (scalar floats).
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
    
def fit_calibration_model_beta(
    model: nn.Module,
    train: IndexDataset,
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
):
    """
    Trains a beta-calibration head.

    Optimizes binary cross-entropy between model outputs and ``labels`` from
    ``train`` batches (``features`` / ``labels`` keys). Uses AdamW with
    cosine annealing scheduler. Optionally plots train/test loss every ``plot_interval``
    iterations when ``test`` is provided.

    Args:
        model: Calibration module mapping features to probabilities.
        train: Training ``IndexDataset`` with ``features`` and ``labels``.
        device: torch.device to perform computations on.
        test: Optional validation set to obtain intermidiate results.
        lr_max: Initial learning rate for AdamW.
        lr_min: Minimum learning rate for the cosine scheduler.
        batch_size: Mini-batch size over ``train``.
        epochs: Number of passes over the training set.
        plot_interval: Evaluate and record losses every N optimizer steps.
        verbose: Show tqdm progress and loss plots.
        logging: Save loss curve figure to ``log_dir``.
        log_dir: Output directory for logged results.
        log_filename: Override auto-generated files names when logging.

    Returns:
        The trained ``model`` (same object, mutated in place).
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

            train_batch_features = train_batch["features"].to(device=device, dtype=torch.float32)
            train_cal_confidence = model(train_batch_features)

            train_batch_labels = train_batch["labels"].to(device=device, dtype=torch.float32)
            
            train_loss = torch.nn.functional.binary_cross_entropy(
                train_cal_confidence,
                train_batch_labels
            )

            train_loss.backward()
            optimizer.step()
            scheduler.step()

            if (
                test is not None
                and iteration_counter % plot_interval == 0
            ):
                train_loss = train_loss.item()
                train_losses.append(train_loss)

                test_data = test.get()
                if test_data:
                    test_batch_features = test_data["features"].to(device=device, dtype=torch.float32)
                    test_batch_labels = test_data["labels"].to(device=device, dtype=torch.float32)
                    
                    with torch.no_grad():
                        test_cal_confidence = model(test_batch_features)
                        test_loss = torch.nn.functional.binary_cross_entropy(
                            test_cal_confidence,
                            test_batch_labels
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
        ax.set_title("Training and Test Loss over Iterations")
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

def fit_calibration_model_temp(
    model: nn.Module,
    train: IndexDataset,
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
):
    """
    Train a temperature-scaling calibration head on logits.

    Optimizes cross-entropy on temperature-scaled logits against
    ``answer_tok_ids`` from ``train`` (``logits`` / ``answer_tok_ids`` keys).
    Training loop and logging mirror ``fit_calibration_model_beta`` function.

    Args:
        model: Temperature-scaled calibration head.
        train: Training ``IndexDataset`` with ``logits`` and ``answer_tok_ids``.
        device: torch.device to perform computations on.
        test: Optional validation set to obtain intermidiate results.
        lr_max: Initial learning rate for AdamW.
        lr_min: Minimum learning rate for the cosine scheduler.
        batch_size: Mini-batch size over ``train``.
        epochs: Number of passes over the training set.
        plot_interval: Evaluate and record losses every N optimizer steps.
        verbose: Show tqdm progress and loss plots.
        logging: Save loss curve figure to ``log_dir``.
        log_dir: Output directory for logged results.
        log_filename: Override auto-generated files names when logging.

    Returns:
        The trained ``model`` (same object, mutated in place).
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

            train_batch_features = train_batch["logits"].to(device=device, dtype=torch.float32)
            train_cal_confidence = model.scale_logits(train_batch_features)
            train_batch_labels = train_batch["answer_tok_ids"].to(device=device, dtype=torch.long)
            
            train_loss = torch.nn.functional.cross_entropy(
                train_cal_confidence,
                train_batch_labels
            )

            train_loss.backward()
            optimizer.step()
            scheduler.step()

            if (
                test is not None
                and iteration_counter % plot_interval == 0
            ):
                train_loss = train_loss.item()
                train_losses.append(train_loss)

                test_data = test.get()
                if test_data:
                    test_batch_features = test_data["logits"].to(device=device, dtype=torch.float32)
                    test_batch_labels = test_data["answer_tok_ids"].to(device=device, dtype=torch.long)
                    
                    with torch.no_grad():
                        test_cal_confidence = model.scale_logits(test_batch_features)
                        test_loss = torch.nn.functional.cross_entropy(
                            test_cal_confidence,
                            test_batch_labels
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
        ax.set_title("Training and Test Loss over Iterations")
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

def fit_hparameters_beta(
    model_class: CalibrationHead,
    train: IndexDataset,
    test: IndexDataset,
    features_count: int,
    device: torch.device,
    search_trials=20,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    ):
    """
    Random search over training hyperparameters for beta calibration.

    Samples up to ``search_trials`` configs from a grid over learning rates,
    batch size, and epochs; fits each with ``fit_calibration_model_beta`` and
    scores on ``test`` and ``test_calibration_model``. Also computes
    SHAP-like feature attributions per trial.

    Args:
        model_class: ``CalibrationHead`` subclass to instantiate per trial.
        train: Training ``IndexDataset``.
        test: Validation ``IndexDataset``.
        features_count: Input dimension.
        device: torch.device to perform computations on.
        search_trials: Number of hyperparameter combinations to try.
        random_seed: Seed for shuffling/sampling the grid; None for nondeterministic.
        verbose: Show trial ECE-diagram and enable nested training verbosity.
        logging: Log per-trial outputs.
        log_dir: Root directory; each trial uses ``search_iter_XXX`` subfolders by deafualt.

    Returns:
        Dict with ``hparameters``, best model ``parameters`` (state dict),
        ``ece+inv_bss`` and ``shap_values`` from the best trial.
    """
    param_grid = {
        "lr_max": [1e-2, 5e-3, 2e-3, 1e-3],
        "lr_min": [1e-3, 5e-4, 2e-4, 1e-4],
        "batch_size": [16, 32],
        "epochs": [1, 3, 5, 10],
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
        epochs = int(sampled["epochs"])

        model = fit_calibration_model_beta(
            model_class(
                in_features=features_count + 1,
                device=device
            ),
            train=train,
            test=test,
            lr_max=lr_max,
            lr_min=lr_min,
            batch_size=batch_size,
            epochs=epochs,
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
                    "epochs": epochs,
                },
                "parameters": model.state_dict(),
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
                "ece+inv_bss": best_result["ece+inv_bss"],
                **best_result["hparameters"],
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
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    ):
    """Random search over training hyperparameters for temperature calibration.

    Same search procedure as ``fit_hparameters_beta``, but uses
    ``fit_calibration_model_temp`` and evaluates calibrated token probabilities
    at ``gen_tok_ids`` positions before computing ECE and composite metrics.

    Args:
        model_class: ``CalibrationHead`` subclass to instantiate per trial.
        train: Training ``IndexDataset``.
        test: Validation ``IndexDataset``.
        features_count: Input dimension.
        device: torch.device to perform computations on.
        search_trials: Number of hyperparameter combinations to try.
        random_seed: Seed for shuffling/sampling the grid; None for nondeterministic.
        verbose: Show trial ECE-diagram and enable nested training verbosity.
        logging: Log per-trial outputs.
        log_dir: Root directory; each trial uses ``search_iter_XXX`` subfolders by deafualt.

    Returns:
        Dict with ``hparameters``, best model ``parameters`` (state dict),
        ``ece+inv_bss`` and ``shap_values`` from the best trial.
    """
    param_grid = {
        "lr_max": [1e-2, 5e-3, 2e-3, 1e-3],
        "lr_min": [1e-3, 5e-4, 2e-4, 1e-4],
        "batch_size": [16, 32],
        "epochs": [1, 3],
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
        epochs = int(sampled["epochs"])

        model = fit_calibration_model_temp(
            model_class(
                in_features=features_count + 1,
                device=device
            ),
            train=train,
            test=test,
            lr_max=lr_max,
            lr_min=lr_min,
            batch_size=batch_size,
            epochs=epochs,
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
                    "epochs": epochs,
                },
                "parameters": model.state_dict(),
                "ece+inv_bss": metrics["ece+inv_bss"],
            }
        )

    best_result = min(results, key=lambda x: x["ece+inv_bss"])
    if logging and log_dir:
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