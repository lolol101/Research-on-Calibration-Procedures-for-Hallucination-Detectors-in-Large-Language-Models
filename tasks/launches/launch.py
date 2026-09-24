"""Collect model responses and per-token scores into an ``Index``.

Command-line equivalent of the per-model launch notebooks in this directory,
parameterised by model, dataset and response regime so that several
collections can run side by side, one process per GPU::

    python tasks/launches/launch.py --model Qwen/Qwen3-4B \\
        --dataset hellaswag --regime cot --gpu 0

Records are written in exactly the layout the notebooks produce, so data
collected here and data already under ``index_data/`` stay interchangeable.

The script also reports how many generations had to be discarded before an
example was accepted. That rate is part of the experimental record: a weaker
model rejected more often is effectively evaluated on a different subset.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.common.datasets import COT_REGIME, REGIMES, DATASETS, get_dataset
from services.common.logging_utils import log_data


def parse_args():
    """Parse command-line options for one collection run."""
    parser = argparse.ArgumentParser(
        description="Collect LLM responses with attention and logit scores.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help='HuggingFace model id, e.g. "Qwen/Qwen3-4B".',
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="Benchmark to collect from.",
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=REGIMES,
        help='"cot" elicits reasoning tokens, "cropped" asks for the answer only.',
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=12000,
        help="Number of dataset rows to collect (default: 12000).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU index to expose via CUDA_VISIBLE_DEVICES; omit to use the default.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "index_data"),
        help="Directory holding the index files (default: <repo>/index_data).",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Index base name; defaults to <model>_<dataset>_<regime>_<iterations>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch RNG seed (default: 42).",
    )
    parser.add_argument(
        "--error-limit",
        type=int,
        default=500,
        help="Generations to retry per row before giving up (default: 500).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Rows between progress-file updates (default: 25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear an existing non-empty index before collecting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing index after its last collected row.",
    )
    return parser.parse_args()


def build_filter(regime, retrieve_answer_token_index):
    """Return the acceptance test for generated responses.

    A response is kept when it carries an answer token that is not the last
    generated position, and the expected answer appears among that token's
    top-k candidates. In the CoT regime the reasoning tags must be present too.

    Args:
        regime: ``"cot"`` or ``"cropped"``.
        retrieve_answer_token_index: Helper locating the answer token.

    Returns:
        Callable ``(model_response, expected_answer) -> bool``.
    """

    def response_is_usable(model_response, expected_answer):
        if regime == COT_REGIME:
            output_text = model_response["output_text"]
            if "<think>" not in output_text or "</think>" not in output_text:
                return False

        answer_token_index = retrieve_answer_token_index(model_response["score_data"])
        if (
            answer_token_index is None
            or answer_token_index >= len(model_response["score_data"]) - 1
        ):
            return False

        token_data = model_response["score_data"][answer_token_index]
        return expected_answer in token_data["top_tokens"]

    return response_is_usable


def resolve_start_row(index, args):
    """Decide which dataset row to start from, honouring --overwrite/--resume.

    Args:
        index: The ``Index`` that will receive records.
        args: Parsed command-line options.

    Returns:
        Index of the first dataset row to collect.

    Raises:
        SystemExit: If the index already holds records and neither
            ``--overwrite`` nor ``--resume`` was given.
    """
    if len(index) == 0:
        return 0

    if args.overwrite:
        index.clear()
        return 0

    if args.resume:
        start_row = max(index.iterations) + 1
        print(f"Resuming after row {start_row - 1} ({len(index)} records present)")
        return start_row

    raise SystemExit(
        f"Index already holds {len(index)} records. "
        f"Pass --resume to continue it or --overwrite to discard it."
    )


def log_progress(log_dir, index_name, stats):
    """Rewrite ``<index_name>_progress.txt`` with the current collection state.

    The file carries the same ``key=value`` fields as the final summary and is
    overwritten in place, so a running collection can be followed from another
    shell once the terminal that started it is gone.

    Args:
        log_dir: Directory holding the index files.
        index_name: Index base name the progress file is named after.
        stats: Mapping of progress field names to values.

    Returns:
        Absolute path to the written file.
    """
    return log_data(
        data=stats,
        log_dir=log_dir,
        log_filename=f"{index_name}_progress.txt",
        separator="=",
    )


def main():
    """Run one collection: load the model, generate, and fill the index."""
    args = parse_args()

    # CUDA_VISIBLE_DEVICES has to be set before torch initialises its runtime,
    # which is why torch and everything importing it are loaded below.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    import torch
    from datasets import load_dataset
    from langchain_core.prompts import ChatPromptTemplate
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from services.common.llm_interface import LLMInterface
    from services.common.logging_utils import log_data
    from services.experiment.cot.data_process_utils import retrieve_answer_token_index
    from services.index import Index

    spec = get_dataset(args.dataset)
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"Device: {device}")

    dataset = load_dataset(spec.hf_path, spec.hf_config, split=spec.hf_split)
    row_count = min(args.iterations, len(dataset))
    if row_count < args.iterations:
        print(
            f"Requested {args.iterations} rows but {spec.name} "
            f"split '{spec.hf_split}' holds {len(dataset)}; collecting {row_count}."
        )

    hf_token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = model.to(device)
    model.eval()

    llm_chain = ChatPromptTemplate.from_messages(
        [
            ("system", spec.system_prompt[args.regime]),
            ("human", spec.user_prompt[args.regime]),
        ]
    ) | LLMInterface(model=model, tokenizer=tokenizer, device=device)

    model_slug = args.model.split("/")[-1].lower()
    index_name = args.index_name or (
        f"{model_slug}_{spec.name}_{args.regime}_{args.iterations}"
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    index = Index(os.path.join(args.output_dir, index_name))
    start_row = resolve_start_row(index, args)

    response_is_usable = build_filter(args.regime, retrieve_answer_token_index)

    accepted_count = 0
    correct_count = 0
    discarded_count = 0
    abandoned_rows = []
    attempted_rows = row_count - start_row
    start_time = time.monotonic()

    progress_bar = tqdm(
        range(start_row, row_count),
        total=row_count - start_row,
        desc=f"{model_slug} / {spec.name} / {args.regime}",
    )

    for row_id in progress_bar:
        dataset_elem = dataset[row_id]
        expected_answer = spec.answer_label(dataset_elem)

        for attempt in range(args.error_limit + 1):
            try:
                response = llm_chain.invoke(spec.build_inputs(dataset_elem))
                response["dataset_elem"] = dataset_elem

                if response_is_usable(response, expected_answer):
                    answer_token_index = retrieve_answer_token_index(
                        response["score_data"]
                    )
                    is_correct = (
                        response["score_data"][answer_token_index]["token"]
                        == expected_answer
                    )
                    accepted_count += 1
                    correct_count += int(is_correct)

                    index.save_data({"iteration": row_id, **response}, row_id)
                    progress_bar.set_postfix(
                        {
                            "accuracy": f"{correct_count / accepted_count:.3f}",
                            "discarded": discarded_count,
                        }
                    )
                    break

                discarded_count += 1

            except Exception as error:  # noqa: BLE001 - collection must not stop
                discarded_count += 1
                if (attempt + 1) % 100 == 0:
                    print(f"Row {row_id}: {attempt + 1} failed attempts. Last: {error}")

            if attempt == args.error_limit:
                abandoned_rows.append(row_id)
                print(f"Row {row_id}: error limit reached, skipping.")

        rows_done = row_id - start_row + 1
        if rows_done % args.progress_every == 0 or rows_done == attempted_rows:
            elapsed = time.monotonic() - start_time
            rows_per_second = rows_done / elapsed if elapsed else 0.0
            log_progress(
                log_dir=args.output_dir,
                index_name=index_name,
                stats={
                    "rows_done": rows_done,
                    "rows_total": attempted_rows,
                    "current_row": row_id,
                    "rows_accepted": accepted_count,
                    "rows_abandoned": len(abandoned_rows),
                    "generations_discarded": discarded_count,
                    "accuracy": (
                        correct_count / accepted_count
                        if accepted_count
                        else float("nan")
                    ),
                    "acceptance_rate": accepted_count / rows_done,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": (
                        round((attempted_rows - rows_done) / rows_per_second, 1)
                        if rows_per_second
                        else float("nan")
                    ),
                },
            )

    summary = {
        "model": args.model,
        "dataset": spec.name,
        "regime": args.regime,
        "rows_attempted": attempted_rows,
        "rows_accepted": accepted_count,
        "rows_abandoned": len(abandoned_rows),
        "generations_discarded": discarded_count,
        "acceptance_rate": (
            accepted_count / attempted_rows if attempted_rows else float("nan")
        ),
        "accuracy": (
            correct_count / accepted_count if accepted_count else float("nan")
        ),
        "index_records": len(index),
    }
    for name, value in summary.items():
        print(f"{name}={value}")

    log_data(
        data=summary,
        log_dir=args.output_dir,
        log_filename=f"{index_name}_summary.txt",
        separator="=",
    )


if __name__ == "__main__":
    main()
