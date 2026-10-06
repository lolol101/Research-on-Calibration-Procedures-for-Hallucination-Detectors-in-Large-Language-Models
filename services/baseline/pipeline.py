from typing import Optional

from sklearn.isotonic import IsotonicRegression
import torch

from ..common.calibration_heads import BetaCalibrationHead, TemperatureCalibrationHead
from ..feature_cache import FeatureCache
from .calibration_utils import (
    fit_hparameters_beta,
    fit_hparameters_temp,
    fit_logistic_regression,
    test_calibration_model,
)

# The baseline works with the final-token probability alone.
FEATURES_COUNT = 0


def run_baseline_calibrations(
    cache: FeatureCache,
    device: torch.device,
    search_trials: int = 20,
    search_seed: Optional[int] = None,
    split_seed: Optional[int] = None,
    bootstrap: bool = False,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
):
    """
    Evaluates the baselines on one index.

    The heads are fitted on the train split,
    hyperparameters are chosen on the val split, and the best model is
    evaluated on the test split. Platt scaling and isotonic regression are
    fitted on the same final-token probability; ``logreg_all_heads`` is an
    L1 logistic regression over the scores of all attention heads, without
    head selection.

    Args:
        cache: ``FeatureCache`` of the index to calibrate.
        device: torch.device to perform computations on.
        search_trials: Hyperparameter combinations tried per calibration head.
        search_seed: Seed for sampling hyperparameter combinations; ``None``
            draws a different sample on every call.
        split_seed: Seed for shuffling records before the train/val/test
            split; ``None`` keeps the contiguous storage order.
        bootstrap: If True, add bootstrap confidence intervals to test metrics.
        verbose: If True, show progress and print metrics.
        logging: If True, write training and test logs under ``log_dir``.
        log_dir: Root directory for logs; required when ``logging=True``.

    Returns:
        Dict mapping ``"raw"``, ``"beta"``, ``"temperature"``, ``"platt"``,
        ``"isotonic"`` and ``"logreg_all_heads"`` to the test metrics returned
        by ``test_calibration_model``.

    Raises:
        ValueError: If ``logging=True`` and ``log_dir`` is not set.
    """
    if logging and not log_dir:
        raise ValueError("log_dir must be set when logging=True")

    def method_log_dir(method, stage):
        return f"{log_dir}/({method})calibration_res/{stage}" if logging else None

    splits = {
        split: cache.baseline_split(split, device, split_seed=split_seed)
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
        random_seed=search_seed,
        save_predictions=True,
    )

    fit_results = fit_hparameters_beta(
        model_class=BetaCalibrationHead,
        train=splits["train"],
        test=splits["val"],
        features_count=FEATURES_COUNT,
        device=device,
        search_trials=search_trials,
        random_seed=search_seed,
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
        random_seed=search_seed,
        save_predictions=True,
    )

    fit_results = fit_hparameters_temp(
        model_class=TemperatureCalibrationHead,
        train=splits["train"],
        test=splits["val"],
        features_count=FEATURES_COUNT,
        device=device,
        search_trials=search_trials,
        random_seed=search_seed,
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
        random_seed=search_seed,
        save_predictions=True,
    )

    def evaluate(method, probs):
        results[method] = test_calibration_model(
            probs,
            test_data["labels"],
            device=device,
            verbose=verbose,
            logging=logging,
            log_dir=method_log_dir(method, "test"),
            bootstrap=bootstrap,
            random_seed=search_seed,
            save_predictions=True,
        )

    def predict(model, features):
        return torch.from_numpy(model.predict_proba(features.cpu().numpy())[:, 1]).to(torch.float32)

    def logit(split_data):
        probs = split_data["features"].reshape(-1, 1).clamp(1e-6, 1 - 1e-6) # [B, 1]
        return torch.log(probs) - torch.log1p(-probs)

    train_data, val_data = splits["train"].get(), splits["val"].get()

    # Platt scaling: logistic regression on the logit of the final-token probability.
    platt = fit_logistic_regression(
        logit(train_data),
        train_data["labels"],
        logit(val_data),
        val_data["labels"],
        device=device,
        c_grid=(1e4,),
        penalty="l2",
        logging=logging,
        log_dir=method_log_dir("platt", "train"),
    )
    evaluate("platt", predict(platt["model"], logit(test_data)))

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(
        train_data["features"].reshape(-1).cpu().numpy(),
        train_data["labels"].cpu().numpy(),
    )
    evaluate(
        "isotonic",
        torch.from_numpy(
            isotonic.predict(test_data["features"].reshape(-1).cpu().numpy())
        ).to(torch.float32),
    )

    # No head selection: every head's score enters an L1 logistic regression.
    _, layers_count, heads_count = cache.attention_entropy.shape # [N, L, H]
    all_layers = torch.arange(layers_count).repeat_interleave(heads_count) # [L * H]
    all_heads = torch.arange(heads_count).repeat(layers_count) # [L * H]
    all_heads_data = {
        split: cache.experiment_split(
            split, all_layers, all_heads, device, split_seed=split_seed
        ).get()
        for split in ("train", "val", "test")
    }
    logreg = fit_logistic_regression(
        all_heads_data["train"]["features"],
        all_heads_data["train"]["labels"],
        all_heads_data["val"]["features"],
        all_heads_data["val"]["labels"],
        device=device,
        penalty="l1",
        logging=logging,
        log_dir=method_log_dir("logreg_all_heads", "train"),
    )
    evaluate("logreg_all_heads", predict(logreg["model"], all_heads_data["test"]["features"]))

    return results
