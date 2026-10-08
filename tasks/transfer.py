"""Cross-dataset transfer of calibration methods for one model and regime.

Every method is fitted on one dataset and evaluated on the others of the same
model and regime: as is, fine-tuned on ``n`` labelled target answers, and
fitted on those answers alone (see ``services.experiment.transfer``). HEAT
uses, per dataset, the configuration ``calibrate.py`` selected on validation,
read from ``--logs-dir``; ``--baseline`` is the strongest baseline of the main
table::

    python tasks/transfer.py --model Qwen/Qwen2.5-7B-Instruct --regime cropped --baseline platt --gpu 0

Results go to ``<logs-dir>/transfer/<model>_<regime>.csv``, one row per
evaluation.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.common.datasets import DATASETS, REGIMES, get_dataset
from services.experiment.transfer import BASELINES


def parse_args():
    """Parse command-line options for one transfer run."""
    parser = argparse.ArgumentParser(
        description="Cross-dataset transfer of calibration methods.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help='HuggingFace model id the indices were collected with, e.g. "Qwen/Qwen2.5-7B-Instruct".',
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=REGIMES,
        help='Response regime of the indices: "cot" or "cropped".',
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=BASELINES,
        help="Baseline transferred alongside HEAT and the logistic regression.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["mmlu-pro", "race", "cosmos-qa"],
        help="Datasets transferred between (default: mmlu-pro race cosmos-qa).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=12000,
        help="Row count the indices were collected with; part of their names (default: 12000).",
    )
    parser.add_argument(
        "--index-dir",
        default=str(REPO_ROOT / "index_data"),
        help="Directory holding the index files (default: <repo>/index_data).",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(REPO_ROOT / "logs"),
        help="Calibration logs with the HEAT runs; results go to its transfer/ (default: <repo>/logs).",
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
        help="Seed of the splits, the draws and the fits (default: 42).",
    )
    return parser.parse_args()


def main():
    """Run the transfer evaluation and write its rows."""
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import pandas as pd
    import torch

    from services.experiment.transfer import heat_config, run_transfer
    from services.feature_cache import FeatureCache
    from services.index import Index

    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_slug = args.model.split("/")[-1].lower()
    caches, heat_configs = {}, {}
    for dataset in args.datasets:
        spec = get_dataset(dataset)
        index_name = f"{model_slug}_{spec.name}_{args.regime}_{args.iterations}"
        index = Index(os.path.join(args.index_dir, index_name))
        if len(index) == 0:
            raise SystemExit(f"Index {index_name!r} in {args.index_dir} holds no records.")
        caches[spec.name] = FeatureCache(index, args.regime, answer_label=spec.answer_label)
        heat_configs[spec.name] = heat_config(args.logs_dir, index_name)
        print(f"{index_name}: HEAT {heat_configs[spec.name]}", flush=True)

    start = time.time()
    rows = run_transfer(
        caches, args.regime, device, args.baseline, heat_configs, seed=args.seed, verbose=True
    )

    out_dir = Path(args.logs_dir) / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_slug}_{args.regime}.csv"
    frame = pd.DataFrame(rows)
    frame.insert(0, "regime", args.regime)
    frame.insert(0, "model", model_slug)
    frame.to_csv(out_path, index=False)
    print(f"{len(rows)} rows in {(time.time() - start) / 60:.1f} min: {out_path}")


if __name__ == "__main__":
    main()
