"""Cross-dataset transfer of calibration methods, as is and with a labelled-data budget.

For one model and response regime, every method is fitted on a source dataset
and evaluated on the test split of every other dataset in three variants:

- ``as_is``: the source fit is applied unchanged (``n = 0``);
- ``finetune``: the source fit is trained further on ``n`` labelled target
  answers, with an L2 penalty that pulls the parameters back to the source
  ones; HEAT is fine-tuned on the source heads and, as ``finetune_reselected``,
  on heads selected anew on the ``n`` answers;
- ``scratch``: the method is fitted on the same ``n`` answers alone, with an
  L2 penalty towards zero (it does not depend on the source).

Methods are HEAT, the logistic regression on the HEAT inputs (the
final-token score alone, with ``PROBE_HEADS`` heads and with all heads) and
one baseline. Penalty strengths
are chosen by cross-validation on the ``n`` answers, since they leave no room
for a validation split; isotonic regression has no parameters to fine-tune.
"""
import glob
import os
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.func import functional_call, vmap

from ..common.calculation_utils import calculate_calibration_metrics
from ..common.calibration_heads import BetaCalibrationHead
from ..feature_cache import FeatureCache
from .calibration_utils import fit_hparameters
from .pipeline import CALIBRATION_HEADS, FEATURES_COUNT, HEAD_SELECTORS

BUDGETS = (25, 50, 100, 200, 500, 1000)
PROBE_HEADS = 10
PROBES = ("logreg_final", f"logreg_k{PROBE_HEADS}", "logreg_all")
BASELINES = ("platt", "beta", "isotonic")
REPORTED_METRICS = ("inv_bss", "auroc", "ece", "ce_debiased", "nlll")
# Penalty strengths tried by cross-validation; the largest keeps a fine-tuned
# model at its source parameters.
STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
CV_FOLDS = 3
FIT_STEPS = 300
FIT_LR = 1e-2


class LinearProbe(nn.Module):
    """Logistic regression on standardized features."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Store the standardization and a zero-initialised linear layer.

        Args:
            mean: Per-feature mean of the data the probe is first fitted on.
            std: Per-feature standard deviation of the same data.
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", torch.where(std > 1e-8, std, torch.ones_like(std)))
        self.linear = nn.Linear(len(mean), 1, device=mean.device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Probability of a correct answer per row."""
        return torch.sigmoid(self.linear((features - self.mean) / self.std).squeeze(-1))


class _Constant:
    """Base-rate forecast for answers that hold one class only."""

    def __init__(self, labels: torch.Tensor):
        self.prob = float((labels.sum() + 1) / (len(labels) + 2))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full((len(features),), self.prob, device=features.device)


def fit_penalized(
    make: Callable[[], nn.Module],
    features: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    strengths: torch.Tensor,
    anchor: Optional[nn.Module],
):
    """Full-batch Adam for several models at once, one per row of ``masks``.

    Model ``m`` minimises the binary cross-entropy over the answers where
    ``masks[m]`` is 1 plus ``strengths[m] * ||theta_m - anchor||^2``. All
    models start from the parameters of ``make()``; they are trained jointly
    with ``torch.func.vmap``, which gives each the result of training it alone.

    Args:
        make: Builds the model, holding its initial parameters.
        features: Inputs, ``[n, F]``.
        labels: Labels (0/1), ``[n]``.
        masks: Training answers of every model, ``[M, n]``.
        strengths: Penalty coefficient of every model, ``[M]``.
        anchor: Model whose parameters the penalty pulls towards; ``None``
            pulls towards zero.

    Returns:
        Tuple ``(model, params)``: the module that ``make`` built and the
        trained parameters of all models, stacked along a leading ``[M]`` axis.
    """
    model = make()
    start = {name: p.detach() for name, p in model.named_parameters()}
    params = {
        name: p.unsqueeze(0).repeat(len(masks), *([1] * p.dim())).requires_grad_()
        for name, p in start.items()
    }
    targets = (
        {name: p.detach() for name, p in anchor.named_parameters()} if anchor is not None
        else {name: torch.zeros_like(p) for name, p in start.items()}
    )
    buffers = dict(model.named_buffers())
    weights = masks / masks.sum(dim=1, keepdim=True) # [M, n]

    def loss(p, w, strength):
        probs = functional_call(model, (p, buffers), (features,)).clamp(1e-6, 1 - 1e-6) # [n]
        bce = -(w * (labels * torch.log(probs) + (1 - labels) * torch.log1p(-probs))).sum()
        return bce + strength * sum(((p[name] - targets[name]) ** 2).sum() for name in p)

    optimizer = torch.optim.Adam(params.values(), lr=FIT_LR)
    batched_loss = vmap(loss)
    model.train()
    for _ in range(FIT_STEPS):
        optimizer.zero_grad()
        batched_loss(params, weights, strengths).sum().backward()
        optimizer.step()
    return model.eval(), {name: p.detach() for name, p in params.items()}


def fit_cv(
    make: Callable[[], nn.Module],
    features: torch.Tensor,
    labels: torch.Tensor,
    anchor: Optional[nn.Module],
    seed: int,
):
    """Penalized fit with the strength of the lowest cross-validated Brier score.

    The fold fits of every strength and the final fits on all answers are
    trained together by ``fit_penalized``. With fewer than ``CV_FOLDS``
    answers of a class no strength can be validated: a fine-tuned model then
    stays at its source parameters and a scratch fit predicts the base rate.

    Args:
        make: Builds the model, see ``fit_penalized``.
        features: Inputs, ``[n, F]``.
        labels: Labels (0/1), ``[n]``.
        anchor: Source model for fine-tuning, ``None`` for a scratch fit.
        seed: Seed of the folds.

    Returns:
        Callable mapping features to probabilities.
    """
    minority = int(min(labels.sum(), len(labels) - labels.sum()))
    if minority < CV_FOLDS:
        return anchor if anchor is not None else _Constant(labels)
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(
        np.zeros(len(labels)), labels.cpu().numpy()
    ))
    held = torch.zeros(len(folds), len(labels), device=features.device) # [folds, n]
    for fold, (_, held_rows) in enumerate(folds):
        held[fold, held_rows] = 1.0
    # Rows: every fold for every strength, then one fit on all answers per strength.
    masks = torch.cat([(1 - held).repeat_interleave(len(STRENGTHS), dim=0),
                       torch.ones(len(STRENGTHS), len(labels), device=features.device)])
    strengths = torch.tensor(STRENGTHS, device=features.device).repeat(len(folds) + 1)
    model, params = fit_penalized(make, features, labels, masks, strengths, anchor)

    buffers = dict(model.named_buffers())
    with torch.no_grad():
        probs = vmap(lambda p: functional_call(model, (p, buffers), (features,)))(params) # [M, n]
        errors = ((probs - labels) ** 2)[: len(folds) * len(STRENGTHS)].reshape(len(folds), len(STRENGTHS), -1)
        brier = (errors * held[:, None, :]).sum(-1) / held.sum(-1, keepdim=True) # [folds, strengths]
        best = len(folds) * len(STRENGTHS) + int(brier.mean(0).argmin())
        for name, p in model.named_parameters():
            p.copy_(params[name][best])
    return model


def predict(model, features: torch.Tensor) -> np.ndarray:
    """Probabilities of a fitted model as a NumPy array."""
    with torch.no_grad():
        return model(features).float().cpu().numpy()


class TransferData:
    """Inputs of every method for the train+val pool and the test split of one index."""

    def __init__(self, cache: FeatureCache, device: torch.device, split_seed: int):
        """Collects the pool (train then val) and the test split.

        Args:
            cache: ``FeatureCache`` of the index.
            device: Device of every tensor and fit.
            split_seed: Seed of the train/val/test shuffle.
        """
        self.cache = cache
        self.device = device
        self.split_seed = split_seed
        _, self.layers_count, self.heads_count = cache.attention_entropy.shape # [N, L, H]

        def stack(getter, key):
            parts = {split: getter(split).get()[key] for split in ("train", "val", "test")}
            return {"pool": torch.cat([parts["train"], parts["val"]]), "test": parts["test"]}

        baseline = lambda split: cache.baseline_split(split, device, split_seed=split_seed)
        no_heads = torch.tensor([], dtype=torch.long)
        final = lambda split: cache.experiment_split(split, no_heads, no_heads, device, split_seed=split_seed)
        entropy = lambda split: cache.attention_entropy_split(split, device, split_seed=split_seed)
        self.labels = {k: v.float() for k, v in stack(baseline, "labels").items()}
        self.probs = {k: v.reshape(-1).float() for k, v in stack(baseline, "features").items()}
        self.final = stack(final, "features")
        self.entropy = stack(entropy, "attention_entropy") # [n, L, H]
        self._all_heads = {}

    def heat_features(self, part: str, layers, heads, attn_only: bool) -> torch.Tensor:
        """HEAT inputs of the given heads, ``[n, blocks * (K + (not attn_only))]``."""
        def split_features(split):
            return self.cache.experiment_split(
                split, layers, heads, self.device, attn_only=attn_only, split_seed=self.split_seed
            ).get()["features"].float()
        if part == "test":
            return split_features("test")
        return torch.cat([split_features("train"), split_features("val")])

    def probe_features(self, part: str, heads=None) -> torch.Tensor:
        """Final-token scores, followed by the attention scores of ``heads``.

        Args:
            part: ``"pool"`` or ``"test"``.
            heads: ``None`` for the final layer alone, ``"all"`` for every
                head (computed once), or a ``(layers, heads)`` pair.
        """
        if heads is None:
            return self.final[part]
        if heads == "all":
            if part not in self._all_heads:
                all_layers = torch.arange(self.layers_count).repeat_interleave(self.heads_count) # [L * H]
                all_heads = torch.arange(self.heads_count).repeat(self.layers_count) # [L * H]
                self._all_heads[part] = torch.cat(
                    [self.final[part], self.heat_features(part, all_layers, all_heads, attn_only=True)], dim=1
                )
            return self._all_heads[part]
        return torch.cat([self.final[part], self.heat_features(part, *heads, attn_only=True)], dim=1)

    def select_heads(self, selector: str, k: int, rows=None):
        """Top-``k`` heads of ``selector`` on the pool answers ``rows`` (all if ``None``)."""
        entropy = self.entropy["pool"] if rows is None else self.entropy["pool"][rows] # [n, L, H]
        labels = self.labels["pool"] if rows is None else self.labels["pool"][rows]
        data = {"labels": (labels > 0.5).long()}
        for l in range(self.layers_count):
            for h in range(self.heads_count):
                data[f"attn_score{l}_{h}"] = entropy[:, l, h]
        layers, heads = HEAD_SELECTORS[selector](
            data=data, layers_count=self.layers_count, heads_count=self.heads_count, best_heads_group_size=k
        )
        return layers.cpu(), heads.cpu()


def heat_config(logs_dir: str, index_name: str) -> Optional[dict]:
    """HEAT configuration with the lowest validation ``inv_bss`` in the logs of ``calibrate.py``.

    Candidates are the attn+final and attn-only runs of the HDP and ROC-AUC
    selectors at every head-selection sample size, as for the reported HEAT.

    Args:
        logs_dir: Root directory of the calibration logs.
        index_name: Index whose runs are searched.

    Returns:
        Dict with ``mode``, ``hs``, ``head``, ``selector`` and ``k``, or
        ``None`` if no run is logged.
    """
    best = None
    pattern = os.path.join(
        logs_dir, index_name, "experiment", "logs_l1_hs_*", "*", "(*)(*)calibration_res",
        "train#*", "best_model_hparameters_*.txt",
    )
    for path in glob.glob(pattern):
        parts = os.path.normpath(path).split(os.sep)
        mode, head_selector, train = parts[-4], parts[-3], parts[-2]
        head, selector = re.match(r"\((.+)\)\((.+)\)calibration_res", head_selector).groups()
        if mode not in ("attn_plus_final", "attn_only") or selector not in HEAD_SELECTORS:
            continue
        with open(path, encoding="utf-8") as f:
            values = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
        val_inv_bss = float(values["inv_bss"])
        if best is None or val_inv_bss < best[0]:
            best = (val_inv_bss, {
                "mode": mode, "hs": int(re.search(r"_hs_(\d+)$", parts[-5]).group(1)),
                "head": head, "selector": selector,
                "k": int(train.split("#")[1]),
            })
    return None if best is None else best[1]


def fit_heat(data: TransferData, regime: str, config: dict, search_seed: int):
    """Refit a HEAT configuration on the train and val splits, as ``run_experiment_calibrations`` does.

    Returns:
        Tuple ``(model, layers, heads)``.
    """
    cache, device = data.cache, data.device
    layers, heads = HEAD_SELECTORS[config["selector"]](
        data=cache.head_selection_data(config["hs"], device, split_seed=data.split_seed),
        layers_count=data.layers_count,
        heads_count=data.heads_count,
        best_heads_group_size=config["k"],
    )
    attn_only = config["mode"] == "attn_only"
    splits = {
        split: cache.experiment_split(split, layers, heads, device, attn_only=attn_only, split_seed=data.split_seed)
        for split in ("train", "val")
    }
    width = splits["train"].get()["features"].shape[1] # blocks * (K + (not attn_only))
    head_class = CALIBRATION_HEADS[config["head"]]
    fit_results = fit_hparameters(
        model_class=head_class,
        train=splits["train"],
        test=splits["val"],
        attn_only=attn_only,
        l1_reg=True,
        features_count=FEATURES_COUNT[regime],
        feature_ids=torch.arange(width),
        heads_count=config["k"],
        random_seed=search_seed,
        device=device,
    )
    model = head_class(in_features=width, device=device)
    model.load_state_dict(fit_results["parameters"])
    return model.eval(), layers.cpu(), heads.cpu()


def heat_maker(config: dict, width: int, device: torch.device, features: torch.Tensor,
               source: Optional[nn.Module], seed: int):
    """Builder of a HEAT head: source parameters, or a seeded initialisation standardized on ``features``."""
    head_class = CALIBRATION_HEADS[config["head"]]

    def make():
        torch.manual_seed(seed)
        model = head_class(in_features=width, device=device)
        if source is not None:
            model.load_state_dict(source.state_dict())
        if source is None or features is not None:
            std = features.std(0)
            model.set_input_scaling(features.mean(0), torch.where(std > 1e-8, std, torch.ones_like(std)))
        return model
    return make


def probe_maker(features: torch.Tensor, source: Optional[nn.Module]):
    """Builder of a ``LinearProbe``: source parameters, or standardized on ``features``."""
    def make():
        probe = LinearProbe(features.mean(0), features.std(0))
        if source is not None:
            probe.load_state_dict(source.state_dict())
        return probe
    return make


def beta_maker(device: torch.device, source: Optional[nn.Module]):
    """Builder of a beta calibration: source parameters, or the identity map."""
    def make():
        model = BetaCalibrationHead(in_features=1, device=device)
        if source is not None:
            model.load_state_dict(source.state_dict())
        return model
    return make


def log_odds(probs: torch.Tensor) -> torch.Tensor:
    """Logit of probabilities clipped to ``[1e-6, 1 - 1e-6]``, as a ``[n, 1]`` column."""
    probs = probs.clamp(1e-6, 1 - 1e-6)
    return (torch.log(probs) - torch.log1p(-probs))[:, None]


def baseline_inputs(baseline: str, probs: torch.Tensor) -> torch.Tensor:
    """Inputs of a baseline: log-odds for Platt scaling, the probability otherwise."""
    return log_odds(probs) if baseline == "platt" else probs


def baseline_maker(baseline: str, data_inputs: torch.Tensor, device: torch.device, source):
    """Builder of the parametric baseline ``platt`` or ``beta``."""
    if baseline == "platt":
        return probe_maker(data_inputs, source)
    return beta_maker(device, source)


def evaluate(probs: np.ndarray, labels: torch.Tensor) -> dict:
    """Test metrics of ``REPORTED_METRICS``."""
    metrics = calculate_calibration_metrics(
        torch.as_tensor(probs, dtype=torch.float32), labels.float().cpu(), device=torch.device("cpu")
    )
    return {name: metrics[name] for name in REPORTED_METRICS}


def budget_draws(pool_size: int, seed: int, budgets: Sequence[int] = BUDGETS):
    """Seeded subsets of the pool: five draws per budget up to 500, three above."""
    for n in budgets:
        if n >= pool_size:
            continue
        for rep in range(5 if n <= 500 else 3):
            yield n, rep, torch.as_tensor(np.random.default_rng([seed, n, rep]).choice(pool_size, n, replace=False))


def fit_source(data: TransferData, regime: str, baseline: str, heat: Optional[dict], seed: int) -> dict:
    """Every method fitted on a whole source pool.

    Returns:
        Dict of method name to a dict with the fitted ``model`` and what the
        method needs to be applied (``heads``, ``config``).
    """
    labels = data.labels["pool"]
    fitted = {}
    inputs = baseline_inputs(baseline, data.probs["pool"])
    if baseline == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        fitted[baseline] = {"model": model.fit(data.probs["pool"].cpu().numpy(), labels.cpu().numpy())}
    else:
        fitted[baseline] = {"model": fit_cv(baseline_maker(baseline, inputs, data.device, None), inputs, labels, None, seed)}

    # The probe's heads are selected on the source, as HEAT selects them on its val split.
    selected = data.select_heads("hdp", PROBE_HEADS)
    for name, heads in zip(PROBES, (None, selected, "all")):
        features = data.probe_features("pool", heads)
        fitted[name] = {"model": fit_cv(probe_maker(features, None), features, labels, None, seed), "heads": heads}

    if heat is not None:
        model, layers, heads = fit_heat(data, regime, heat, seed)
        fitted["heat"] = {"model": model, "heads": (layers, heads), "config": heat}
    return fitted


def run_transfer(
    caches: Dict[str, FeatureCache],
    regime: str,
    device: torch.device,
    baseline: str,
    heat_configs: Optional[Dict[str, dict]] = None,
    seed: int = 42,
    budgets: Sequence[int] = BUDGETS,
    verbose: bool = False,
) -> List[dict]:
    """Transfer evaluation over the datasets of one model and regime.

    Args:
        caches: ``FeatureCache`` per dataset name, same model and regime.
        regime: ``"cot"`` or ``"cropped"``.
        device: Device of every fit.
        baseline: The baseline transferred, one of ``BASELINES``.
        heat_configs: HEAT configuration per dataset name (see
            ``heat_config``); datasets without one are transferred without HEAT.
        seed: Seed of the splits, the draws and the fits.
        budgets: Numbers of labelled target answers.
        verbose: If True, print progress.

    Returns:
        One dict per evaluation with ``source`` (``None`` for ``scratch``),
        ``target``, ``method``, ``variant``, ``n``, ``rep`` and the metrics.
    """
    heat_configs = heat_configs or {}
    data = {name: TransferData(cache, device, seed) for name, cache in caches.items()}
    rows = []

    def record(source, target, method, variant, n, rep, probs):
        rows.append({
            "source": source, "target": target, "method": method, "variant": variant,
            "n": n, "rep": rep, **evaluate(probs, data[target].labels["test"]),
        })

    # Scratch fits on the target answers alone; HEAT uses the target's own configuration.
    for target, t in data.items():
        config = heat_configs.get(target)
        for n, rep, draw in budget_draws(len(t.labels["pool"]), seed, budgets):
            labels, fold_seed = t.labels["pool"][draw], seed + rep
            if baseline == "isotonic":
                model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                model.fit(t.probs["pool"][draw].cpu().numpy(), labels.cpu().numpy())
                probs = model.predict(t.probs["test"].cpu().numpy())
            else:
                inputs = baseline_inputs(baseline, t.probs["pool"][draw])
                model = fit_cv(baseline_maker(baseline, inputs, device, None), inputs, labels, None, fold_seed)
                probs = predict(model, baseline_inputs(baseline, t.probs["test"]))
            record(None, target, baseline, "scratch", n, rep, probs)

            selected = t.select_heads("hdp", PROBE_HEADS, draw)
            for name, heads in zip(PROBES, (None, selected, "all")):
                features = t.probe_features("pool", heads)[draw]
                model = fit_cv(probe_maker(features, None), features, labels, None, fold_seed)
                record(None, target, name, "scratch", n, rep, predict(model, t.probe_features("test", heads)))

            if config is not None:
                attn_only = config["mode"] == "attn_only"
                layers, heads = t.select_heads(config["selector"], config["k"], draw)
                features = t.heat_features("pool", layers, heads, attn_only)[draw]
                make = heat_maker(config, features.shape[1], device, features, None, fold_seed)
                model = fit_cv(make, features, labels, None, fold_seed)
                record(None, target, "heat", "scratch", n, rep,
                       predict(model, t.heat_features("test", layers, heads, attn_only)))
        if verbose:
            print(f"scratch on {target}: done", flush=True)

    for source, s in data.items():
        fitted = fit_source(s, regime, baseline, heat_configs.get(source), seed)
        for target, t in data.items():
            if target == source:
                continue
            draws = list(budget_draws(len(t.labels["pool"]), seed, budgets))
            for method, fit in fitted.items():
                model = fit["model"]
                if method == baseline:
                    if baseline == "isotonic":
                        record(source, target, method, "as_is", 0, 0, model.predict(t.probs["test"].cpu().numpy()))
                        continue
                    pool, test = (baseline_inputs(baseline, t.probs[part]) for part in ("pool", "test"))
                    make = baseline_maker(baseline, pool, device, model)
                    variants = {"finetune": make}
                elif method in PROBES:
                    pool, test = (t.probe_features(part, fit["heads"]) for part in ("pool", "test"))
                    variants = {"finetune": probe_maker(pool, model)}
                else:
                    config = fit["config"]
                    attn_only = config["mode"] == "attn_only"
                    pool, test = (t.heat_features(part, *fit["heads"], attn_only) for part in ("pool", "test"))
                    variants = {"finetune": heat_maker(config, pool.shape[1], device, None, model, seed)}
                record(source, target, method, "as_is", 0, 0, predict(model, test))

                for n, rep, draw in draws:
                    labels, fold_seed = t.labels["pool"][draw], seed + rep
                    for variant, make in variants.items():
                        tuned = fit_cv(make, pool[draw], labels, model, fold_seed)
                        record(source, target, method, variant, n, rep, predict(tuned, test))
                    if method == "heat":
                        layers, heads = t.select_heads(config["selector"], config["k"], draw)
                        features = t.heat_features("pool", layers, heads, attn_only)[draw]
                        make = heat_maker(config, pool.shape[1], device, features, model, seed)
                        tuned = fit_cv(make, features, labels, model, fold_seed)
                        record(source, target, method, "finetune_reselected", n, rep,
                               predict(tuned, t.heat_features("test", layers, heads, attn_only)))
            if verbose:
                print(f"transfer {source} -> {target}: done", flush=True)
    return rows
