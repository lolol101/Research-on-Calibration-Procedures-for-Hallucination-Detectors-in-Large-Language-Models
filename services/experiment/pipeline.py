from functools import partial
from typing import Optional, Sequence

import torch
from tqdm import tqdm

from ..baseline.calibration_utils import fit_logistic_regression
from ..common.calibration_heads import (
    MLPBetaCalibrationHead,
    MLPCalibrationHead,
    WeightedBetaCalibrationHead,
)
from ..common.datasets import COT_REGIME, CROPPED_REGIME
from ..common.logging_utils import log_data
from ..feature_cache import FeatureCache
from .calibration_utils import (
    find_best_layer_head_hdp,
    find_best_layer_head_roc_auc,
    find_random_layer_head,
    fit_hparameters,
    test_calibration_model,
)
# Aggregated statistics per score column: CoT adds the seven reasoning-span
# statistics of ``calculate_agg_features`` to the answer-token scores.
FEATURES_COUNT = {
    COT_REGIME: 7,
    CROPPED_REGIME: 0,
}

HEAD_SELECTORS = {
    "hdp": find_best_layer_head_hdp,
    "roc-auc": find_best_layer_head_roc_auc,
}

CALIBRATION_HEADS = {
    "mlp": MLPCalibrationHead,
    "mlp+beta": MLPBetaCalibrationHead,
    "weighted_beta": WeightedBetaCalibrationHead,
}

# Inverse L2 strengths searched by the logistic-regression probes.
LOGREG_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def select_heads(
    cache: FeatureCache,
    best_heads_group_size: int,
    hs_size: int,
    device: torch.device,
    split_seed: Optional[int] = None,
):
    """
    Ranks attention heads on the val split by every criterion in ``HEAD_SELECTORS``.

    When the whole val split is used (``hs_size=0``), a ``random`` draw of
    heads is added as an ablation; it does not depend on the selection
    sample, so it is evaluated once rather than for every ``hs_size``.

    Args:
        cache: ``FeatureCache`` of the index to calibrate.
        best_heads_group_size: How many (layer, head) pairs to keep.
        hs_size: Number of leading val records used for selection; 0 uses all.
        device: torch.device to perform computations on.
        split_seed: Seed for shuffling records before the train/val/test
            split; ``None`` keeps the contiguous storage order.

    Returns:
        Dict mapping selector name to ``(best_layers, best_heads)``.
    """
    _, layers_count, heads_count = cache.attention_entropy.shape # [N, L, H]
    head_selection_data = cache.head_selection_data(hs_size, device, split_seed=split_seed)

    selectors = dict(HEAD_SELECTORS)
    if hs_size == 0:
        selectors["random"] = partial(
            find_random_layer_head, seed=0 if split_seed is None else split_seed
        )

    return {
        name: selector(
            data=head_selection_data,
            layers_count=layers_count,
            heads_count=heads_count,
            best_heads_group_size=best_heads_group_size,
        )
        for name, selector in selectors.items()
    }


def run_experiment_calibrations(
    cache: FeatureCache,
    regime: str,
    device: torch.device,
    attn_only: bool = False,
    answer_only: bool = False,
    hs_size: int = 50,
    best_heads_group_size: int = 30,
    heads_group_sizes: Sequence[int] = (1, 3, 5, 7, 10, 15, 20, 30),
    search_trials: int = 20,
    search_seed: Optional[int] = None,
    split_seed: Optional[int] = None,
    l1_reg: bool = True,
    l2_reg: bool = False,
    bootstrap: bool = False,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Runs head selection and attention-based calibration on one index.

    For every head-selection criterion and calibration head, fits on the train split,
    chooses hyperparameters on the val split, and evaluates on the test split
    once per number of selected heads.

    Args:
        cache: ``FeatureCache`` of the index to calibrate.
        regime: ``"cot"`` or ``"cropped"``.
        device: torch.device to perform computations on.
        attn_only: If True, omit final-token confidence features.
        answer_only: CoT only; if True, keep the answer-token scores and drop
            the reasoning-span statistics, so the two feature sets can be
            compared on the same records.
        hs_size: Number of leading val records used for head selection; 0 uses all.
        best_heads_group_size: How many (layer, head) pairs to select.
        heads_group_sizes: Numbers of best heads fed to the calibration head.
        search_trials: Hyperparameter combinations tried per fit.
        search_seed: Seed for sampling hyperparameter combinations; ``None``
            draws a different sample on every call.
        split_seed: Seed for shuffling records before the train/val/test
            split; ``None`` keeps the contiguous storage order.
        l1_reg: Include L1 penalty values in the search grid.
        l2_reg: Include L2 penalty values in the search grid.
        bootstrap: If True, add bootstrap confidence intervals to test metrics.
        verbose: If True, show progress and print metrics.
        logging: If True, write selected heads, training and test logs under ``log_dir``.
        log_dir: Root directory for logs; required when ``logging=True``.

    Returns:
        Nested dict ``{selector: {calibration_head: {heads_count: metrics}}}``
        with the test metrics returned by ``test_calibration_model``.

    Raises:
        ValueError: If ``logging=True`` and ``log_dir`` is not set, a value
            in ``heads_group_sizes`` exceeds ``best_heads_group_size``, or
            ``answer_only`` is set outside the CoT regime.
    """
    if logging and not log_dir:
        raise ValueError("log_dir must be set when logging=True")
    if answer_only and regime != COT_REGIME:
        raise ValueError("answer_only applies to the CoT regime only")
    if max(heads_group_sizes) > best_heads_group_size:
        raise ValueError(
            f"heads_group_sizes {list(heads_group_sizes)} exceed "
            f"best_heads_group_size={best_heads_group_size}"
        )

    features_count = FEATURES_COUNT[regime]
    block_size = best_heads_group_size + (not attn_only)
    # The answer-token scores are the last of the ``features_count + 1`` blocks.
    blocks = [features_count] if answer_only else range(features_count + 1)
    if answer_only:
        features_count = 0

    selected_heads = select_heads(
        cache=cache,
        best_heads_group_size=best_heads_group_size,
        hs_size=hs_size,
        split_seed=split_seed,
        device=device,
    )

    results = {}
    for selector_name, (best_layers, best_heads) in selected_heads.items():
        best_pairs = [(l.item(), h.item()) for l, h in zip(best_layers, best_heads)]
        if verbose:
            print(f"Best Layers and Heads ({selector_name}): {best_pairs}")
        if logging:
            log_data(
                data={rank: pair for rank, pair in enumerate(best_pairs)},
                log_dir=log_dir,
                log_filename=f"best_heads({selector_name}).txt",
            )

        splits = {
            split: cache.experiment_split(
                split,
                best_layers,
                best_heads,
                device,
                attn_only=attn_only,
                split_seed=split_seed,
            )
            for split in ("train", "val", "test")
        }
        test_data = splits["test"].get()

        results[selector_name] = {}
        for head_name, head_class in CALIBRATION_HEADS.items():
            local_log_dir = f"{log_dir}/({head_name})({selector_name})calibration_res/"
            results[selector_name][head_name] = {}

            for group_size in tqdm(
                heads_group_sizes,
                desc=f"{head_name} / {selector_name}: calibrating with various heads",
            ):
                # Leading ``group_size`` score columns of each used block of
                # width ``block_size``.
                feature_ids = torch.cat([
                    torch.arange(
                        k * block_size,
                        k * block_size + (group_size + (not attn_only)),
                        dtype=torch.long,
                    )
                    for k in blocks
                ]) # [len(blocks) * (group_size + (not ATTN_ONLY))]

                fit_results = fit_hparameters(
                    model_class=head_class,
                    train=splits["train"],
                    test=splits["val"],
                    attn_only=attn_only,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    features_count=features_count,
                    feature_ids=feature_ids,
                    heads_count=group_size,
                    search_trials=search_trials,
                    random_seed=search_seed,
                    device=device,
                    logging=logging,
                    log_dir=local_log_dir + f"train#{group_size}" if logging else None,
                )

                model = head_class(
                    in_features=(features_count + 1) * (group_size + (not attn_only)),
                    device=device,
                )
                model.load_state_dict(fit_results["parameters"])
                model.eval()

                test_calibrated_probs = model.calibrate(
                    test_data["features"][:, feature_ids], device
                )
                results[selector_name][head_name][group_size] = test_calibration_model(
                    test_calibrated_probs.to(device=device, dtype=torch.float32),
                    test_data["labels"].to(device=device, dtype=torch.float32),
                    device=device,
                    verbose=verbose,
                    logging=logging,
                    log_dir=local_log_dir + f"test#{group_size}" if logging else None,
                    bootstrap=bootstrap,
                    random_seed=search_seed,
                    save_predictions=True,
                )

    return results


def run_logreg_calibrations(
    cache: FeatureCache,
    device: torch.device,
    hs_size: int = 50,
    best_heads_group_size: int = 30,
    heads_group_sizes: Sequence[int] = (1, 3, 5, 7, 10, 15, 20, 30),
    search_seed: Optional[int] = None,
    split_seed: Optional[int] = None,
    bootstrap: bool = False,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Logistic regression on the inputs of HEAT, without and with attention scores.

    The features are those of the experiment's attn+final mode, so the probes
    differ from HEAT only in the model, and from one another only in the
    attention scores they get. Probes: ``final`` (the final-token score of
    every block alone); each selector of ``select_heads`` with its leading
    ``heads_group_sizes`` heads added; ``all`` with every head added. One L2
    search throughout: ``C`` is chosen on the val split by ``inv_bss``.

    Args:
        cache: ``FeatureCache`` of the index to calibrate.
        device: torch.device to perform computations on.
        hs_size: Number of leading val records used for head selection; 0 uses all.
        best_heads_group_size: How many (layer, head) pairs to select.
        heads_group_sizes: Numbers of selected heads added to the final-token score.
        search_seed: Seed of the bootstrap resampling.
        split_seed: Seed for shuffling records before the train/val/test
            split; ``None`` keeps the contiguous storage order.
        bootstrap: If True, add bootstrap confidence intervals to test metrics.
        verbose: If True, show progress and print metrics.
        logging: If True, write selected heads, training and test logs under ``log_dir``.
        log_dir: Root directory for logs; required when ``logging=True``.

    Returns:
        Nested dict ``{probe: {heads_count: metrics}}`` with the test metrics
        returned by ``test_calibration_model``; ``final`` has heads count 0
        and ``all`` the number of heads of the model.

    Raises:
        ValueError: If ``logging=True`` and ``log_dir`` is not set, or a value
            in ``heads_group_sizes`` exceeds ``best_heads_group_size``.
    """
    if logging and not log_dir:
        raise ValueError("log_dir must be set when logging=True")
    if max(heads_group_sizes) > best_heads_group_size:
        raise ValueError(
            f"heads_group_sizes {list(heads_group_sizes)} exceed "
            f"best_heads_group_size={best_heads_group_size}"
        )

    splits = ("train", "val", "test")
    _, layers_count, heads_count = cache.attention_entropy.shape # [N, L, H]
    selected_heads = select_heads(
        cache=cache,
        best_heads_group_size=best_heads_group_size,
        hs_size=hs_size,
        split_seed=split_seed,
        device=device,
    )

    no_heads = torch.tensor([], dtype=torch.long)
    probes = [("final", 0, (no_heads, no_heads))]
    for selector_name, (best_layers, best_heads) in selected_heads.items():
        if logging:
            log_data(
                data={
                    rank: (l.item(), h.item())
                    for rank, (l, h) in enumerate(zip(best_layers, best_heads))
                },
                log_dir=log_dir,
                log_filename=f"best_heads({selector_name}).txt",
            )
        probes += [
            (selector_name, size, (best_layers[:size], best_heads[:size]))
            for size in heads_group_sizes
        ]
    probes.append((
        "all",
        layers_count * heads_count,
        (
            torch.arange(layers_count).repeat_interleave(heads_count), # [L * H]
            torch.arange(heads_count).repeat(layers_count), # [L * H]
        ),
    ))

    results = {}
    for probe_name, size, heads in tqdm(probes, desc="logreg: calibrating with various heads"):
        data = {
            split: cache.experiment_split(split, *heads, device, split_seed=split_seed).get()
            for split in splits
        } # features: [B, blocks * (1 + size)]

        local_log_dir = f"{log_dir}/(logreg)({probe_name})calibration_res/"
        fit_results = fit_logistic_regression(
            data["train"]["features"],
            data["train"]["labels"],
            data["val"]["features"],
            data["val"]["labels"],
            device=device,
            c_grid=LOGREG_C_GRID,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            logging=logging,
            log_dir=local_log_dir + f"train#{size}" if logging else None,
        )
        test_probs = torch.from_numpy(
            fit_results["model"].predict_proba(data["test"]["features"].cpu().numpy())[:, 1]
        ).to(device=device, dtype=torch.float32)
        results.setdefault(probe_name, {})[size] = test_calibration_model(
            test_probs,
            data["test"]["labels"].to(device=device, dtype=torch.float32),
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=local_log_dir + f"test#{size}" if logging else None,
            bootstrap=bootstrap,
            random_seed=search_seed,
            save_predictions=True,
        )

    return results
