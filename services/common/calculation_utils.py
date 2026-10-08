import os
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from typing import Optional
from torch import nn
from tqdm import tqdm
from .logging_utils import log_data, make_next_indexed_log_filename

def adaptive_bin_ids(sorted_probs: torch.Tensor, n_bins: int, device: torch.device) -> torch.Tensor:
    """
    Equal-count bin of every sample, assigned by value.

    Bin edges are the empirical quantiles ``k / n_bins`` of the confidences.
    Samples with equal confidence always share a bin: a tie group crossing a
    quantile goes whole to the upper bin, so the binning does not depend on
    how the sort orders ties. Without ties the bins are of equal size.

    Args:
        sorted_probs: Predicted probabilities sorted in ascending order.
        n_bins: Number of bins.
        device: torch.device to perform computations on.

    Returns:
        Bin index in ``[0, n_bins)`` of every sample, aligned with ``sorted_probs``.
    """
    bin_size = len(sorted_probs) // n_bins
    # Lower edge of every bin after the first: the confidence at rank k * bin_size.
    edges = sorted_probs[torch.arange(1, n_bins, device=device) * bin_size] # [n_bins - 1]
    return torch.bucketize(sorted_probs, edges, right=True) # [N]

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

    Bins samples with ``adaptive_bin_ids`` (empirical quantiles, ties never
    split) and sums ``|avg_confidence - accuracy| * proportion``. Optionally
    plots a reliability diagram.

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
    bin_ids = adaptive_bin_ids(sorted_probs, n_bins, device) # [N]

    ece = torch.zeros(1, device=device)

    bin_avg_confidences = []
    bin_accuracies_list = []
    bin_conf_min = []
    bin_conf_max = []

    for i in range(n_bins):
        in_bin = bin_ids == i
        bin_count = int(in_bin.sum())

        if bin_count > 0:
            bin_probs = sorted_probs[in_bin]
            bin_accuracies = sorted_labels[in_bin]

            prop_in_bin = bin_count / n_samples

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

def calculate_brier_decomposition(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_bins: int = 10,
):
    """
    Murphy decomposition of the Brier score with bias-corrected terms.

    Over the ``adaptive_bin_ids`` bins, ``Brier ~ REL - RES + UNC``:

    - ``rel`` (reliability): ``sum_b w_b (acc_b - conf_b)^2``, the squared
      calibration error; lower is better.
    - ``res`` (resolution): ``sum_b w_b (acc_b - acc)^2``, how far the
      accuracy of the bins spreads around the base rate; higher is better.
    - ``unc`` (uncertainty): ``acc (1 - acc)``, set by the data alone.

    A bin's observed accuracy is noisy, which inflates both plug-in sums by
    its sampling variance; following Kumar, Liang and Ma (2019, "Verified
    Uncertainty Calibration") the estimate ``acc_b (1 - acc_b) / (n_b - 1)``
    is subtracted per bin. The same term enters ``rel`` and ``res``, so
    ``rel - res`` is unchanged, while small and large bins are compared on
    equal terms. The corrected sums may dip slightly below zero.
    ``ce_debiased = sqrt(max(rel, 0))`` is the debiased L2 calibration error
    on the probability scale.

    Args:
        token_probs: Predicted probabilities.
        labels: Binary labels (0/1).
        device: torch.device to perform computations on.
        n_bins: Number of adaptive bins.

    Returns:
        Dict with ``ce_debiased``, ``rel``, ``res`` and ``unc`` as Python floats.
    """
    probs = token_probs.to(device=device, dtype=torch.float64)
    labels_f = labels.to(device=device, dtype=torch.float64)
    order = torch.argsort(probs)
    sorted_probs, sorted_labels = probs[order], labels_f[order]
    bin_ids = adaptive_bin_ids(sorted_probs, n_bins, device) # [N]

    counts = torch.bincount(bin_ids, minlength=n_bins).to(torch.float64) # [n_bins]
    conf_sums = torch.bincount(bin_ids, weights=sorted_probs, minlength=n_bins) # [n_bins]
    acc_sums = torch.bincount(bin_ids, weights=sorted_labels, minlength=n_bins) # [n_bins]
    filled = counts > 0
    counts = counts[filled]
    conf = conf_sums[filled] / counts
    acc = acc_sums[filled] / counts
    weights = counts / len(sorted_probs)
    base_rate = sorted_labels.mean()

    noise = acc * (1 - acc) / (counts - 1).clamp_min(1) # [bins]; 0 for single-sample bins
    rel = (weights * ((acc - conf) ** 2 - noise)).sum()
    res = (weights * ((acc - base_rate) ** 2 - noise)).sum()
    unc = base_rate * (1 - base_rate)

    return {
        "ce_debiased": rel.clamp_min(0).sqrt().item(),
        "rel": rel.item(),
        "res": res.item(),
        "unc": unc.item(),
    }

def calculate_calibration_metrics(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_bins: int = 10,
    eps: float = 1e-6,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Full set of calibration metrics for one set of predictions.

    Shared by the baseline and experiment ``test_calibration_model`` helpers,
    and reused by ``calculate_metrics_bootstrap_ci`` on each resample.

    Args:
        token_probs: Predicted probabilities.
        labels: Binary labels (0/1).
        device: torch.device to perform computations on.
        n_bins: Number of adaptive bins for ECE.
        eps: Clipping bound for probabilities in NLL / BSS reference.
        verbose: Show the reliability diagram drawn by the ECE helper.
        logging: Save that diagram to ``log_dir``.
        log_dir: Directory for logged figures; required when ``logging=True``.

    Returns:
        Dict with ``ece``, the ``calculate_brier_decomposition`` terms
        (``ce_debiased``, ``rel``, ``res``, ``unc``), ``inv_bss``, ``nlll``,
        ``mse``, ``auroc`` and ``accuracy`` as Python floats.
    """
    if logging and not log_dir:
        raise ValueError("logging=True requires log_dir")

    ece_value = calculate_ece_adaptive_bins(
        token_probs,
        labels,
        n_bins=n_bins,
        device=device,
        verbose=verbose,
        logging=logging,
        log_dir=log_dir,
    )

    probs = token_probs.to(device)
    labels_f = labels.to(device=device, dtype=torch.float32)
    probs_clipped = torch.clamp(probs, eps, 1 - eps)

    nlll = torch.nn.functional.binary_cross_entropy(probs_clipped, labels_f)
    mse = torch.nn.functional.mse_loss(probs, labels_f)
    accuracy = torch.mean(labels_f)

    p_ref = labels_f.mean()
    brier_score_ref = torch.mean((p_ref - labels_f) ** 2)
    inv_brier_skill_score = (
        (mse / brier_score_ref).item() if brier_score_ref > eps else float("nan")
    )

    return {
        "ece": ece_value,
        **calculate_brier_decomposition(token_probs, labels, device=device, n_bins=n_bins),
        "inv_bss": inv_brier_skill_score,
        "nlll": nlll.item(),
        "mse": mse.item(),
        "auroc": calculate_roc_auc(probs, labels_f) if 0 < accuracy < 1 else float("nan"),
        "accuracy": accuracy.item(),
    }

def calculate_metrics_bootstrap_ci(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
    n_bins: int = 10,
    eps: float = 1e-6,
    verbose: bool = False,
):
    """
    Percentile bootstrap confidence intervals for the calibration metrics.

    Resamples the evaluation set with replacement ``n_resamples`` times,
    recomputes every metric on each resample, and reports empirical quantiles.
    ECE is a biased statistic with an asymmetric sampling distribution, so
    percentile intervals are preferred over a normal approximation.

    Resamples whose labels are all-zero or all-one make the Brier reference
    vanish and yield ``nan`` for ``inv_bss``; such draws are dropped per
    metric rather than propagated.

    Args:
        token_probs: Predicted probabilities.
        labels: Binary labels (0/1).
        device: torch.device to perform computations on.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage of the reported interval.
        random_seed: Seed for the resampling RNG; ``None`` leaves it unseeded.
        n_bins: Number of adaptive bins for ECE.
        eps: Clipping bound for probabilities in NLL / BSS reference.
        verbose: Show a tqdm progress bar over resamples.

    Returns:
        Dict mapping each metric name to a ``(low, high)`` tuple of floats,
        or ``(nan, nan)`` when every resample produced ``nan``.
    """
    n_samples = labels.shape[0]
    generator = torch.Generator(device="cpu")
    if random_seed is not None:
        generator.manual_seed(random_seed)

    collected = {}
    for _ in tqdm(
        range(n_resamples),
        desc="Bootstrapping metrics...",
        disable=not verbose,
    ):
        resample_ids = torch.randint(
            0, n_samples, (n_samples,), generator=generator
        )

        # Probabilities and labels may live on different devices.
        resample_metrics = calculate_calibration_metrics(
            token_probs[resample_ids.to(token_probs.device)],
            labels[resample_ids.to(labels.device)],
            device=device,
            n_bins=n_bins,
            eps=eps,
        )
        for name, value in resample_metrics.items():
            collected.setdefault(name, []).append(value)

    lower_q = (1 - confidence) / 2
    upper_q = 1 - lower_q

    intervals = {}
    for name, values in collected.items():
        samples = torch.tensor(values, dtype=torch.float32)
        samples = samples[~torch.isnan(samples)]
        if samples.numel() == 0:
            intervals[name] = (float("nan"), float("nan"))
            continue
        intervals[name] = (
            torch.quantile(samples, lower_q).item(),
            torch.quantile(samples, upper_q).item(),
        )

    return intervals

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
