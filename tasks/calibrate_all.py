"""Run every calibration for one model as a sequential queue.

Finds the indices collected for ``--model`` under ``index_data/`` (named the
way ``tasks/launch.py`` names them), then runs ``tasks/calibrate.py`` once per
combination, one after another on a single GPU::

    python tasks/calibrate_all.py --model Qwen/Qwen2.5-7B-Instruct --gpu 0

For every index the queue holds the baseline and, per head-selection sample
size, the experiment in each feature mode of ``--modes`` and the
logistic-regression probes. Several models can run side by side, one queue
per GPU.

Each run is a separate process, so the memory an index occupies is released
before the next run starts. A failed run does not stop the queue; its output
stays in ``<logs-dir>/runs/<model>/`` and the final summary lists it.

Options not recognised here are passed on to every ``calibrate.py`` run, e.g.
``--search-trials 10 --bootstrap``.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.common.datasets import COT_REGIME, DATASETS, REGIMES

CALIBRATE_SCRIPT = REPO_ROOT / "tasks" / "calibrate.py"
METHODS = ("baseline", "experiment", "logreg")
FULL_VAL_SPLIT = 0
# Experiment feature modes and the calibrate.py flags selecting them;
# answer_only applies to CoT indices only.
MODES = {
    "attn_plus_final": [],
    "attn_only": ["--attn-only"],
    "answer_only": ["--answer-only"],
}


def hs_size(value):
    """Parse a head-selection sample size; ``full`` means the whole val split."""
    return FULL_VAL_SPLIT if value == "full" else int(value)


def parse_args():
    """Parse queue options; unrecognised options are kept for ``calibrate.py``."""
    parser = argparse.ArgumentParser(
        description="Run all calibrations for one model sequentially.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help='HuggingFace model id the indices were collected with, e.g. "Qwen/Qwen2.5-7B-Instruct".',
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU index passed to every run; omit to use the default.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Datasets to include (default: all with a collected index).",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=list(REGIMES),
        help="Response regimes to include (default: all).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to include (default: baseline experiment logreg).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
        help=(
            "Experiment feature modes; answer_only (CoT indices only) drops the "
            "reasoning-span statistics (default: all)."
        ),
    )
    parser.add_argument(
        "--hs-sizes",
        nargs="+",
        type=hs_size,
        default=[50, 100, 500, 1000, FULL_VAL_SPLIT],
        help='Head-selection sample sizes; "full" uses the whole val split (default: 50 100 500 1000 full).',
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
        help="Root directory for logs (default: <repo>/logs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue without running it.",
    )
    return parser.parse_known_args()


def build_queue(args, model_slug):
    """List the ``calibrate.py`` runs for every collected index of the model.

    Args:
        args: Parsed queue options.
        model_slug: Lower-cased model name used in index names.

    Returns:
        Tuple ``(queue, missing)``: ``queue`` holds ``(run_name, run_args)``
        pairs, ``missing`` the index names that were not found.
    """
    queue, missing = [], []
    for dataset in args.datasets:
        for regime in args.regimes:
            # Same naming as tasks/launch.py and tasks/calibrate.py.
            index_name = f"{model_slug}_{dataset}_{regime}_{args.iterations}"
            index_file = os.path.join(args.index_dir, f"{index_name}_index.pkl")
            if not os.path.exists(index_file):
                missing.append(index_name)
                continue

            common = ["--model", args.model, "--dataset", dataset, "--regime", regime]
            if "baseline" in args.methods:
                queue.append((f"{index_name}__baseline", common + ["--method", "baseline"]))
            for size in args.hs_sizes:
                size_tag = "full" if size == FULL_VAL_SPLIT else str(size)
                if "experiment" in args.methods:
                    for mode in args.modes:
                        if mode == "answer_only" and regime != COT_REGIME:
                            continue
                        queue.append((
                            f"{index_name}__experiment__hs_{size_tag}__{mode}",
                            common
                            + ["--method", "experiment", "--hs-size", str(size)]
                            + MODES[mode],
                        ))
                if "logreg" in args.methods:
                    queue.append((
                        f"{index_name}__logreg__hs_{size_tag}",
                        common + ["--method", "logreg", "--hs-size", str(size)],
                    ))
    return queue, missing


def main():
    """Build the queue for one model and run it sequentially."""
    args, passthrough = parse_args()
    model_slug = args.model.split("/")[-1].lower()

    queue, missing = build_queue(args, model_slug)
    for index_name in missing:
        print(f"No index {index_name!r}; skipped.")
    if not queue:
        raise SystemExit(f"Nothing to run: no indices for {args.model} in {args.index_dir}.")

    shared = [
        "--iterations", str(args.iterations),
        "--index-dir", args.index_dir,
        "--logs-dir", args.logs_dir,
    ]
    if args.gpu is not None:
        shared += ["--gpu", str(args.gpu)]
    shared += passthrough

    runs_dir = Path(args.logs_dir) / "runs" / model_slug
    print(f"{len(queue)} runs queued; outputs in {runs_dir}")

    if args.dry_run:
        for run_name, run_args in queue:
            print(" ".join([str(CALIBRATE_SCRIPT), *run_args, *shared]))
        return

    runs_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    queue_start = time.monotonic()
    for position, (run_name, run_args) in enumerate(queue, start=1):
        output_path = runs_dir / f"{run_name}.out"
        print(f"[{time.strftime('%F %T')}] {position}/{len(queue)} start {run_name}", flush=True)

        run_start = time.monotonic()
        with open(output_path, "w") as output:
            completed = subprocess.run(
                [sys.executable, str(CALIBRATE_SCRIPT), *run_args, *shared],
                stdout=output,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )
        minutes = (time.monotonic() - run_start) / 60

        status = "done" if completed.returncode == 0 else f"FAIL (exit {completed.returncode})"
        print(
            f"[{time.strftime('%F %T')}] {position}/{len(queue)} {status} "
            f"{run_name} in {minutes:.1f} min",
            flush=True,
        )
        if completed.returncode != 0:
            failed.append((run_name, output_path))

    hours = (time.monotonic() - queue_start) / 3600
    print(f"Finished {len(queue) - len(failed)}/{len(queue)} runs in {hours:.2f} h.")
    for run_name, output_path in failed:
        print(f"Failed: {run_name} -> {output_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
