from functools import partial
from typing import Callable, Optional

import torch

from ..common.calibration_heads import BetaCalibrationHead, TemperatureCalibrationHead
from ..common.datasets import letter_answer_label
from ..index import Index, IndexDataset
from .calibration_utils import (
    fit_hparameters_beta,
    fit_hparameters_temp,
    test_calibration_model,
)
from .data_process_utils import process_elements_main

# The baseline works with the final-token probability alone.
FEATURES_COUNT = 0


def run_baseline_calibrations(
    index: Index,
    device: torch.device,
    answer_label: Callable[[dict], str] = letter_answer_label,
    search_trials: int = 20,
    bootstrap: bool = False,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Evaluates the uncalibrated, beta and temperature baselines on one index.

    Mirrors the notebooks under ``tasks/baseline_calibrations/``: the heads are
    fitted on the train split, hyperparameters are chosen on the val split, and
    the best model is evaluated on the test split.

    Args:
        index: ``Index`` with collected model responses.
        device: torch.device to perform computations on.
        answer_label: Callable mapping ``dataset_elem`` to the expected answer
            token; pass the matching ``DatasetSpec.answer_label``.
        search_trials: Hyperparameter combinations tried per calibration head.
        bootstrap: If True, add bootstrap confidence intervals to test metrics.
        verbose: If True, show progress and print metrics.
        logging: If True, write training and test logs under ``log_dir``.
        log_dir: Root directory for logs; required when ``logging=True``.

    Returns:
        Dict mapping ``"raw"``, ``"beta"`` and ``"temperature"`` to the test
        metrics returned by ``test_calibration_model``.

    Raises:
        ValueError: If ``logging=True`` and ``log_dir`` is not set.
    """
    if logging and not log_dir:
        raise ValueError("log_dir must be set when logging=True")

    def method_log_dir(method, stage):
        return f"{log_dir}/({method})calibration_res/{stage}" if logging else None

    splits = {
        split: IndexDataset(
            index,
            partial(process_elements_main, device=device, answer_label=answer_label),
            split=split,
            load_all_data=True,
            verbose=verbose,
        )
        for split in ("train", "val", "test")
    }
    test_data = splits["test"].get()

    results = {}

    raw_test_probs = test_data["features"]
    if raw_test_probs.ndim > 1:
        raw_test_probs = raw_test_probs.squeeze(-1)
    results["raw"] = test_calibration_model(
        raw_test_probs,
        test_data["labels"],
        device=device,
        verbose=verbose,
        logging=logging,
        log_dir=method_log_dir("raw", "test"),
        bootstrap=bootstrap,
    )

    fit_results = fit_hparameters_beta(
        model_class=BetaCalibrationHead,
        train=splits["train"],
        test=splits["val"],
        features_count=FEATURES_COUNT,
        device=device,
        search_trials=search_trials,
        logging=logging,
        log_dir=method_log_dir("beta", "train"),
    )
    model = BetaCalibrationHead(in_features=FEATURES_COUNT + 1, device=device)
    model.load_state_dict(fit_results["parameters"])
    model.eval()
    results["beta"] = test_calibration_model(
        model.calibrate(test_data["features"], device),
        test_data["labels"],
        device=device,
        verbose=verbose,
        logging=logging,
        log_dir=method_log_dir("beta", "test"),
        bootstrap=bootstrap,
    )

    fit_results = fit_hparameters_temp(
        model_class=TemperatureCalibrationHead,
        train=splits["train"],
        test=splits["val"],
        features_count=FEATURES_COUNT,
        device=device,
        search_trials=search_trials,
        logging=logging,
        log_dir=method_log_dir("temperature", "train"),
    )
    model = TemperatureCalibrationHead(in_features=FEATURES_COUNT + 1, device=device)
    model.load_state_dict(fit_results["parameters"])
    model.eval()
    test_calibrated_probs = model.calibrate(test_data["logits"], device) # [B, TOP_K]
    test_calibrated_probs = test_calibrated_probs.gather(
        1, test_data["gen_tok_ids"].unsqueeze(1)
    ).squeeze(1) # [B]
    results["temperature"] = test_calibration_model(
        test_calibrated_probs,
        test_data["labels"],
        device=device,
        verbose=verbose,
        logging=logging,
        log_dir=method_log_dir("temperature", "test"),
        bootstrap=bootstrap,
    )

    return results
