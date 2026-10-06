"""Train and evaluate calibration heads on a collected ``Index``.

Parameterised by model, dataset, response regime and method, so that several
runs can go side by side, one process per GPU::

    python tasks/calibrate.py --model Qwen/Qwen3-4B \\
        --dataset hellaswag --regime cot --method experiment --gpu 0

The index is located by the name ``tasks/launch.py`` gives it, or
passed explicitly with ``--index-name``. The number of layers and heads is
read from the stored records, so a new model needs no extra configuration.

Logs follow the notebook layout under
``<logs-dir>/<index_name>/<method>/...``; for ``experiment`` the path also
carries the regularisation, the head-selection sample size and the feature
mode, e.g. ``logs_l1_hs_50/attn_plus_final/``.

A run that finishes writes ``completed.txt`` into its log directory. With
``--skip-done`` such a run is skipped, and the log directory of an interrupted
run is removed before the run starts over, so a queue can be resumed after a
crash without mixing partial and fresh logs.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.common.datasets import DATASETS, REGIMES, get_dataset
from services.common.logging_utils import log_data

METHODS = ("baseline", "experiment")
COMPLETION_MARKER = "completed.txt"


def parse_args():
    """Parse command-line options for one calibration run."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate calibration heads on a collected index.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help='HuggingFace model id the index was collected with, e.g. "Qwen/Qwen3-4B".',
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="Benchmark the index was collected from.",
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=REGIMES,
        help='Response regime of the index: "cot" or "cropped".',
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=METHODS,
        help='"baseline": raw, beta and temperature; "experiment": attention-head calibration.',
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=12000,
        help="Row count the index was collected with; part of its default name (default: 12000).",
    )
    parser.add_argument(
        "--index-dir",
        default=str(REPO_ROOT / "index_data"),
        help="Directory holding the index files (default: <repo>/index_data).",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Index base name; defaults to <model>_<dataset>_<regime>_<iterations>.",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(REPO_ROOT / "logs"),
        help="Root directory for logs (default: <repo>/logs).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU index to expose via CUDA_VISIBLE_DEVICES; omit to use the default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for torch, the train/val/test shuffle, and hyperparameter sampling (default: 42).",
    )
    parser.add_argument(
        "--search-trials",
        type=int,
        default=20,
        help="Hyperparameter combinations tried per fit (default: 20).",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Add bootstrap confidence intervals to the test metrics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show data loading progress and print test metrics.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip the run if it already completed; remove the logs of an interrupted run and start over.",
    )

    experiment = parser.add_argument_group("experiment method")
    experiment.add_argument(
        "--attn-only",
        action="store_true",
        help="Use attention scores only, without the final-token confidence.",
    )
    experiment.add_argument(
        "--answer-only",
        action="store_true",
        help="CoT only: use the answer-token scores without the reasoning-span statistics.",
    )
    experiment.add_argument(
        "--hs-size",
        type=int,
        default=50,
        help="Leading val records used for head selection; 0 uses the whole val split (default: 50).",
    )
    experiment.add_argument(
        "--best-heads",
        type=int,
        default=30,
        help="Number of (layer, head) pairs to select (default: 30).",
    )
    experiment.add_argument(
        "--heads-group-sizes",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 10, 15, 20, 30],
        help="Numbers of best heads fed to the calibration head (default: 1 3 5 7 10 15 20 30).",
    )
    experiment.add_argument(
        "--l1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include L1 penalty values in the search grid (default: on).",
    )
    experiment.add_argument(
        "--l2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include L2 penalty values in the search grid (default: off).",
    )
    return parser.parse_args()


def experiment_log_subdir(args):
    """Return the notebook-style subdirectory naming the experiment settings."""
    regularisation = "_".join(
        name for name, enabled in (("l1", args.l1), ("l2", args.l2)) if enabled
    ) or "default"
    feature_mode = "attn_only" if args.attn_only else "attn_plus_final"
    if args.answer_only:
        feature_mode += "_answer_only"
    return os.path.join(f"logs_{regularisation}_hs_{args.hs_size}", feature_mode)


def main():
    """Run one calibration: load the index, fit heads, and log test metrics."""
    args = parse_args()

    # CUDA_VISIBLE_DEVICES has to be set before torch initialises its runtime,
    # which is why torch and everything importing it are loaded below.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    import torch

    from services.baseline.pipeline import run_baseline_calibrations
    from services.experiment.pipeline import run_experiment_calibrations
    from services.feature_cache import FeatureCache
    from services.index import Index

    spec = get_dataset(args.dataset)
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"Device: {device}")

    model_slug = args.model.split("/")[-1].lower()
    index_name = args.index_name or (
        f"{model_slug}_{spec.name}_{args.regime}_{args.iterations}"
    )
    index = Index(os.path.join(args.index_dir, index_name))
    if len(index) == 0:
        raise SystemExit(f"Index {index_name!r} in {args.index_dir} holds no records.")
    print(f"Index: {index_name} ({len(index)} records)")

    log_dir = os.path.join(args.logs_dir, index_name, args.method)
    if args.method == "experiment":
        log_dir = os.path.join(log_dir, experiment_log_subdir(args))

    if args.skip_done:
        if os.path.exists(os.path.join(log_dir, COMPLETION_MARKER)):
            print(f"Already completed, skipped: {log_dir}")
            return
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
            print(f"Removed logs of an interrupted run: {log_dir}")

    # Built on the first run over this index, read by every later one.
    cache = FeatureCache(index, args.regime, answer_label=spec.answer_label, verbose=args.verbose)
    print(f"Features: {cache.path}")

    if args.method == "baseline":
        run_baseline_calibrations(
            cache=cache,
            device=device,
            search_trials=args.search_trials,
            search_seed=args.seed,
            split_seed=args.seed,
            bootstrap=args.bootstrap,
            verbose=args.verbose,
            logging=True,
            log_dir=log_dir,
        )
    else:
        run_experiment_calibrations(
            cache=cache,
            regime=args.regime,
            device=device,
            attn_only=args.attn_only,
            answer_only=args.answer_only,
            hs_size=args.hs_size,
            best_heads_group_size=args.best_heads,
            heads_group_sizes=args.heads_group_sizes,
            search_trials=args.search_trials,
            search_seed=args.seed,
            split_seed=args.seed,
            l1_reg=args.l1,
            l2_reg=args.l2,
            bootstrap=args.bootstrap,
            verbose=args.verbose,
            logging=True,
            log_dir=log_dir,
        )

    log_data(
        data={"finished": time.strftime("%Y-%m-%d %H:%M:%S")},
        log_dir=log_dir,
        log_filename=COMPLETION_MARKER,
    )
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
