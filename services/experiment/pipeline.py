from functools import partial
from typing import Callable, Optional, Sequence

import torch
from tqdm import tqdm

from ..common.calibration_heads import (
    MLPBetaCalibrationHead,
    MLPCalibrationHead,
    WeightedBetaCalibrationHead,
)
from ..common.datasets import COT_REGIME, CROPPED_REGIME, letter_answer_label
from ..common.logging_utils import log_data
from ..index import Index, IndexDataset
from .calibration_utils import (
    find_best_layer_head_hal_dif_power,
    find_best_layer_head_roc_auc,
    fit_hparameters,
    test_calibration_model,
)
from .cot import data_process_utils as cot_processing
from .cropped import data_process_utils as cropped_processing

PROCESSING = {
    COT_REGIME: cot_processing,
    CROPPED_REGIME: cropped_processing,
}

# Aggregated statistics per score column: CoT adds the seven reasoning-span
# statistics of ``calculate_agg_features`` to the answer-token scores.
FEATURES_COUNT = {
    COT_REGIME: 7,
    CROPPED_REGIME: 0,
}

HEAD_SELECTORS = {
    "hal": find_best_layer_head_hal_dif_power,
    "roc-auc": find_best_layer_head_roc_auc,
}

CALIBRATION_HEADS = {
    "mlp": MLPCalibrationHead,
    "mlp+beta": MLPBetaCalibrationHead,
    "weighted_beta": WeightedBetaCalibrationHead,
}


def infer_attention_shape(index: Index):
    """
    Reads the number of layers and heads from the first stored record.

    Args:
        index: Non-empty ``Index`` with ``attention_entropy`` per token.

    Returns:
        Tuple ``(layers_count, heads_count)``.
    """
    elem = index.load_data(0, 1)[0]
    attention_entropy = torch.stack(
        elem["attention_entropy"], dim=0
    ).squeeze(-1) # [T, L, H]
    return attention_entropy.shape[1], attention_entropy.shape[2]


def select_heads(
    index: Index,
    regime: str,
    layers_count: int,
    heads_count: int,
    best_heads_group_size: int,
    hs_size: int,
    device: torch.device,
    answer_label: Callable[[dict], str] = letter_answer_label,
    verbose: bool = False,
):
    """
    Ranks attention heads on the val split by every criterion in ``HEAD_SELECTORS``.

    Args:
        index: ``Index`` with collected model responses.
        regime: ``"cot"`` or ``"cropped"``.
        layers_count: Number of transformer layers.
        heads_count: Number of heads per layer.
        best_heads_group_size: How many (layer, head) pairs to keep.
        hs_size: Number of leading val records used for selection; 0 uses all.
        device: torch.device to perform computations on.
        answer_label: Callable mapping ``dataset_elem`` to the expected answer.
        verbose: If True, show progress.

    Returns:
        Dict mapping selector name to ``(best_layers, best_heads)``.
    """
    head_selection_dataset = IndexDataset(
        index=index,
        process_elements=partial(
            PROCESSING[regime].process_elements_hal,
            layers_count=layers_count,
            heads_count=heads_count,
            answer_label=answer_label,
            device=device,
        ),
        split="val",
        load_all_data=True,
        verbose=verbose,
    )
    head_selection_data = head_selection_dataset.get(end=hs_size) if hs_size > 0 \
        else head_selection_dataset.get()

    return {
        name: selector(
            data=head_selection_data,
            layers_count=layers_count,
            heads_count=heads_count,
            best_heads_group_size=best_heads_group_size,
        )
        for name, selector in HEAD_SELECTORS.items()
    }


def run_experiment_calibrations(
    index: Index,
    regime: str,
    device: torch.device,
    attn_only: bool = False,
    hs_size: int = 50,
    best_heads_group_size: int = 30,
    heads_group_sizes: Sequence[int] = (1, 3, 5, 7, 10, 15, 20, 30),
    search_trials: int = 20,
    l1_reg: bool = True,
    l2_reg: bool = False,
    answer_label: Callable[[dict], str] = letter_answer_label,
    bootstrap: bool = False,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Runs head selection and attention-based calibration on one index.

    Mirrors the notebooks under ``legacy/experiment_calibrations/``: for every
    head-selection criterion and calibration head, fits on the train split,
    chooses hyperparameters on the val split, and evaluates on the test split
    once per number of selected heads.

    Args:
        index: ``Index`` with collected model responses.
        regime: ``"cot"`` or ``"cropped"``.
        device: torch.device to perform computations on.
        attn_only: If True, omit final-token confidence features.
        hs_size: Number of leading val records used for head selection; 0 uses all.
        best_heads_group_size: How many (layer, head) pairs to select.
        heads_group_sizes: Numbers of best heads fed to the calibration head.
        search_trials: Hyperparameter combinations tried per fit.
        l1_reg: Include L1 penalty values in the search grid.
        l2_reg: Include L2 penalty values in the search grid.
        answer_label: Callable mapping ``dataset_elem`` to the expected answer
            token; pass the matching ``DatasetSpec.answer_label``.
        bootstrap: If True, add bootstrap confidence intervals to test metrics.
        verbose: If True, show progress and print metrics.
        logging: If True, write selected heads, training and test logs under ``log_dir``.
        log_dir: Root directory for logs; required when ``logging=True``.

    Returns:
        Nested dict ``{selector: {calibration_head: {heads_count: metrics}}}``
        with the test metrics returned by ``test_calibration_model``.

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

    features_count = FEATURES_COUNT[regime]
    block_size = best_heads_group_size + (not attn_only)

    layers_count, heads_count = infer_attention_shape(index)
    if verbose:
        print(f"Attention shape: {layers_count} layers x {heads_count} heads")

    selected_heads = select_heads(
        index=index,
        regime=regime,
        layers_count=layers_count,
        heads_count=heads_count,
        best_heads_group_size=best_heads_group_size,
        hs_size=hs_size,
        device=device,
        answer_label=answer_label,
        verbose=verbose,
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
            split: IndexDataset(
                index,
                partial(
                    PROCESSING[regime].process_elements_main,
                    best_layers=best_layers,
                    best_heads=best_heads,
                    attn_only=attn_only,
                    answer_label=answer_label,
                    device=device,
                ),
                split=split,
                load_all_data=True,
                verbose=verbose,
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
                # Leading ``group_size`` score columns of each of the
                # ``features_count + 1`` blocks of width ``block_size``.
                feature_ids = torch.cat([
                    torch.arange(
                        k * block_size,
                        k * block_size + (group_size + (not attn_only)),
                        dtype=torch.long,
                    )
                    for k in range(features_count + 1)
                ]) # [(FEATURES_COUNT + 1) * (group_size + (not ATTN_ONLY))]

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
                )

    return results
