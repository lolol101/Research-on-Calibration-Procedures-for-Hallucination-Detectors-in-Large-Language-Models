import os
import torch
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from typing import Optional
from torch import nn
from .logging_utils import log_data, make_next_indexed_log_filename

def calculate_ece_adaptive_bins(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_bins: int = 10,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
):
    """
    Expected calibration error with equal-count (adaptive) bins.

    Sorts samples by confidence, splits into ``n_bins`` groups of equal size,
    and sums ``|avg_confidence - accuracy| * proportion``. Optionally plots
    a reliability diagram.

    Args:
        token_probs: Predicted probabilities.
        labels: Binary labels (0/1).
        device: torch.device to perform computations on.
        n_bins: Number of adaptive bins.
        verbose: Show the reliability plot.
        logging: Save the plot PNG to ``log_dir``.
        log_dir: Directory for logged figures.
        log_filename: Override auto-generated PNG name.

    Returns:
        ECE as a Python float.
    """
    if logging and not log_dir:
        raise ValueError("logging=True requires log_dir")

    token_probs = token_probs.to(device)
    labels = labels.to(device)

    sorted_indices = torch.argsort(token_probs)
    sorted_probs = token_probs[sorted_indices]
    sorted_labels = labels[sorted_indices]

    n_samples = len(sorted_probs)
    bin_size = n_samples // n_bins

    ece = torch.zeros(1, device=device)

    bin_avg_confidences = []
    bin_accuracies_list = []
    bin_conf_min = []
    bin_conf_max = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = n_samples if i == n_bins - 1 else (i + 1) * bin_size

        if end_idx > start_idx:
            bin_probs = sorted_probs[start_idx:end_idx]
            bin_accuracies = sorted_labels[start_idx:end_idx]

            prop_in_bin = (end_idx - start_idx) / n_samples

            accuracy_in_bin = bin_accuracies.float().mean()
            avg_confidence_in_bin = bin_probs.mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if verbose or logging:
                bin_avg_confidences.append(avg_confidence_in_bin.detach().cpu())
                bin_accuracies_list.append(accuracy_in_bin.detach().cpu())
                bin_conf_min.append(bin_probs.min().detach().cpu())
                bin_conf_max.append(bin_probs.max().detach().cpu())

    if verbose or logging:
        bin_avg_confidences = torch.stack(bin_avg_confidences).numpy()
        bin_accuracies_list = torch.stack(bin_accuracies_list).numpy()
        bin_conf_min = torch.stack(bin_conf_min).numpy()
        bin_conf_max = torch.stack(bin_conf_max).numpy()

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(
            bin_avg_confidences,
            bin_accuracies_list,
            marker="o",
            linewidth=2,
            label="Model (adaptive bins)",
        )

        for i in range(len(bin_accuracies_list)):
            ax.fill_between(
                [
                    bin_conf_min[i] if i > 0 else 0,
                    bin_conf_max[i] if i < len(bin_accuracies_list) - 1 else 1,
                ],
                0,
                bin_accuracies_list[i] + 0.005,
                alpha=0.4,
            )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram with Adaptive Bin Coverage")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fname = log_filename or make_next_indexed_log_filename(
                log_dir=log_dir,
                prefix="ece_adaptive_bins",
                extension=".png",
            )
            out_path = os.path.join(log_dir, fname)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")

        if verbose:
            plt.show()

        plt.close(fig)

    return ece.item()

def calculate_ece_fixed_bins(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_bins: int = 10,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
):
    """
    Expected calibration error with uniform-width bins on [0, 1].

    Args:
        token_probs: Predicted probabilities.
        labels: Binary labels (0/1).
        device: Device for masks and ECE accumulation.
        n_bins: Number of fixed-width bins.
        verbose: Show the reliability plot.
        logging: Save the plot PNG to ``log_dir``.
        log_dir: Directory for logged figures.
        log_filename: Override auto-generated PNG name.

    Returns:
        ECE as a Python float.
    """
    if logging and not log_dir:
        raise ValueError("logging=True requires log_dir")

    token_probs = token_probs.to(device)
    labels = labels.to(device)

    n_samples = token_probs.numel()
    ece = torch.zeros(1, device=device)

    bin_avg_confidences = []
    bin_accuracies_list = []
    bin_conf_min = []
    bin_conf_max = []

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)

    for i in range(n_bins):
        if i < n_bins - 1:
            bin_mask = (token_probs >= bin_boundaries[i]) & (token_probs < bin_boundaries[i + 1])
        else:
            bin_mask = (token_probs >= bin_boundaries[i]) & (token_probs <= bin_boundaries[i + 1])

        bin_count = int(bin_mask.sum().item())
        if bin_count == 0:
            continue

        bin_probs = token_probs[bin_mask]
        bin_accuracies = labels[bin_mask]

        prop_in_bin = bin_count / float(n_samples)

        accuracy_in_bin = bin_accuracies.float().mean()
        avg_confidence_in_bin = bin_probs.mean()

        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        if verbose or logging:
            bin_avg_confidences.append(avg_confidence_in_bin.detach().cpu())
            bin_accuracies_list.append(accuracy_in_bin.detach().cpu())
            bin_conf_min.append(bin_probs.min().detach().cpu())
            bin_conf_max.append(bin_probs.max().detach().cpu())

    if verbose or logging:
        if len(bin_accuracies_list) == 0:
            if verbose:
                plt.show()
            return ece.item()

        bin_avg_confidences = torch.stack(bin_avg_confidences).numpy()
        bin_accuracies_list = torch.stack(bin_accuracies_list).numpy()
        bin_conf_min = torch.stack(bin_conf_min).numpy()
        bin_conf_max = torch.stack(bin_conf_max).numpy()

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(
            bin_avg_confidences,
            bin_accuracies_list,
            marker="o",
            linewidth=2,
            label="Model (fixed bins)",
        )

        for i in range(len(bin_accuracies_list)):
            ax.fill_between(
                [
                    bin_conf_min[i] if i > 0 else 0,
                    bin_conf_max[i] if i < len(bin_accuracies_list) - 1 else 1,
                ],
                0,
                bin_accuracies_list[i] + 0.005,
                alpha=0.4,
            )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram with Fixed Bin Coverage")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fname = log_filename or make_next_indexed_log_filename(
                log_dir=log_dir,
                prefix="ece_fixed_bins",
                extension=".png",
            )
            out_path = os.path.join(log_dir, fname)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")

        if verbose:
            plt.show()

        plt.close(fig)

    return ece.item()

def calculate_roc_auc(
    probs: torch.Tensor,
    labels: torch.Tensor,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
):
    """
    Area under the ROC curve for binary scores.

    Args:
        probs: Predicted scores or probabilities.
        labels: Binary ground-truth labels.
        verbose: Display the ROC curve.
        logging: Save the ROC figure to ``log_dir``.
        log_dir: Directory for logged figures.
        log_filename: Override auto-generated PNG name.

    Returns:
        ROC AUC as a Python float.
    """
    if logging and not log_dir:
        raise ValueError("logging=True requires log_dir")

    y = labels.to(dtype=torch.float32).cpu().numpy()
    p = probs.to(dtype=torch.float32).cpu().numpy()

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = float(auc(fpr, tpr))

    if verbose or logging:
        fig, ax = plt.subplots()
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        fig.tight_layout()

        if logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fname = log_filename or make_next_indexed_log_filename(
                log_dir=log_dir,
                prefix="roc_auc",
                extension=".png",
            )
            out_path = os.path.join(log_dir, fname)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")

        if verbose:
            plt.show()

        plt.close(fig)

    return roc_auc


def calculate_feature_shap_values(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    *,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Computes SHAP-like feature attributions for model inputs.
    Uses gradient * (x - baseline) with baseline = mean(features).
    Returns mean absolute attribution per feature.
    """
    if features.ndim == 1:
        features = features.unsqueeze(1)
    elif features.ndim != 2:
        raise ValueError("features must be 2D tensor [n_samples, n_features]")
    if len(features) == 0:
        return torch.empty((0,), dtype=torch.float32)

    model.eval()
    feats = features.to(device=device, dtype=torch.float32)
    baseline = feats.mean(dim=0, keepdim=True)

    attr_sum = torch.zeros(feats.shape[1], device=device, dtype=torch.float32)
    count = 0

    for start in range(0, len(feats), batch_size):
        batch = feats[start : start + batch_size].clone().detach().requires_grad_(True)
        pred = model(batch)
        grads = torch.autograd.grad(pred.sum(), batch, retain_graph=False, create_graph=False)[0]
        attributions = (batch - baseline) * grads
        attr_sum += attributions.abs().sum(dim=0)
        count += batch.shape[0]

    if count == 0:
        return torch.zeros(feats.shape[1], dtype=torch.float32, device=device).cpu()
    shap_values = (attr_sum / count).detach().cpu()
    shap_sum = shap_values.sum()
    if shap_sum > 0:
        shap_values = shap_values / shap_sum


    feature_indices = torch.arange(len(shap_values), dtype=torch.long)

    if verbose and len(shap_values) > 0:
        top_idx = torch.argsort(shap_values, descending=True)[: len(shap_values)]
        top_pairs = [
            (
                int(feature_indices[i].item()),
                float(shap_values[i].item()),
            )
            for i in top_idx.tolist()
        ]
        print(f"len(shap_values) SHAP feature values: {top_pairs}")

    if logging and log_dir:
        sort_idx = torch.argsort(shap_values, descending=True)
        payload = {
            str(feature_indices[idx].item()): f"{shap_values[idx].item():.10f}"
            for idx in sort_idx.tolist()
        }
        log_data(
            data=payload,
            log_dir=log_dir,
            prefix="calibration_feature_shap",
            extension=".txt",
            separator="\t",
        )

    return shap_values

def calculate_entropy(
    probs: torch.Tensor,
):
    """
    Shannon entropy along the last dimension of ``probs``.

    Args:
        probs: Non-negative values in (0, 1], summing to 1 along the last axis.

    Returns:
        Entropy tensor with shape ``probs.shape[:-1]``.
    """
    assert torch.all((probs > 0) & (probs <= 1)), (
        f"prob_scores must be in (0, 1] range, but: min={probs.min():.4f}, "
        f"max={probs.max():.4f}, shape={probs.shape}, "
        f"negative values: {(probs < 0).sum().item()}, "
        f">1 values: {(probs > 1).sum().item()}"
    )

    logprobs = torch.log(probs)
    entropy = -(logprobs * probs).sum(dim=-1)

    return entropy

def calculate_norm_entropy(token_scores: torch.Tensor):
    """
    Entropy normalized by ``log(last_dim_size)`` for the last dimension.

    Args:
        token_scores: Probability distribution(s).

    Returns:
        Normalized entropy (same leading shape as input).
    """
    entropy = calculate_entropy(token_scores)
    norm_attn_entropy = entropy / torch.log(torch.tensor(token_scores.shape[-1]))
    return norm_attn_entropy

def calculate_agg_features(
    t_features: torch.Tensor, # [T, EMB_HEADS_COUNT + (not ATTN_ONLY)]
    early_ratio=0.25, 
    late_ratio=0.75,
    ):
    """
    Aggregate a token-time feature matrix into a fixed-length vector.

    Computes mean, std, quantiles, coefficients of variation, and early/late
    statistics along the time axis (dim 0).

    Args:
        t_features: Tensor ``[T, F]`` of per-token features.
        early_ratio: Quantile and slice boundary for the early segment.
        late_ratio: Quantile and slice start for the late segment.

    Returns:
        1D tensor of concatenated statistics, length ``7 * F``.
    """
    t_features = t_features.to(torch.float32)
    
    mean = t_features.mean(0)
    std = t_features.std(0)
    q_early = torch.quantile(t_features, early_ratio, dim=0)
    q_late = torch.quantile(t_features, late_ratio, dim=0)
    cv = std / mean
    
    mean_early = t_features[:int(t_features.shape[0] * early_ratio)].mean(0)
    std_early = t_features[:int(t_features.shape[0] * early_ratio)].std(0)
    cv_early = std_early / mean_early
    
    mean_late = t_features[int(t_features.shape[0] * late_ratio):].mean(0)
    early_and_late_diff = mean_late - mean_early
    
    return torch.cat(
        [mean, std, q_early, q_late, cv, cv_early, early_and_late_diff]
    ) # [FEATURES_COUNT * (EMB_HEADS_COUNT + (not ATTN_ONLY))]
