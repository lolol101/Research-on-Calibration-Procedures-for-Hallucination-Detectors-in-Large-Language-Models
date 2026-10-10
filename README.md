# HEAT: Head-Entropy Attention-Aware Calibration of LLM Hallucination Detectors

Code for the paper. HEAT calibrates the confidence of an LLM's answer from the
attention entropy of selected heads together with the final-layer score of the
answer token. The repository collects model responses with per-token scores
and attention entropies, selects heads, trains the calibrators and the
baselines, and evaluates transfer between datasets.

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # torch 2.5.1, CUDA 12.4 (the reported results)
uv sync --no-default-groups --group local   # torch 2.7.1, CUDA 12.8, for RTX 50xx GPUs
```

Gated models need `HF_TOKEN` in the environment.

## Data

Models: `Qwen/Qwen2.5-0.5B-Instruct` (cropped), `Qwen/Qwen2.5-3B-Instruct`,
`Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct` (cropped
and CoT). Datasets: MMLU-Pro, RACE, CosmosQA, 12 000 rows each. Regimes:
`cropped` (the answer only) and `cot` (reasoning, then the answer).

```bash
python tasks/launch.py --model Qwen/Qwen2.5-7B-Instruct --dataset mmlu-pro --regime cot
```

Llama-3 in the cropped regime was collected with `--no-example`. Each run
writes an index to `index_data/<model>_<dataset>_<regime>_12000_{data,index}.pkl`:
per record the generated tokens with their raw-logit top-30 scores and the
attention entropy of every head. A record carries its train/val/test split
(60/20/20 by a seeded hash of the dataset row).

## Calibration

All methods for one model, every index found in `index_data/`:

```bash
python tasks/calibrate_all.py --model Qwen/Qwen2.5-7B-Instruct --gpu 0 \
    --bootstrap --best-heads 200 --heads-group-sizes 1 3 5 7 10 15 20 30 50 100 200
```

The queue runs `tasks/calibrate.py` per index with

- `--method baseline`: Platt, temperature, isotonic and beta calibration of the
  final-layer score;
- `--method logreg`: logistic regression on the final layer alone, with
  selected heads and with all heads;
- `--method experiment`: HEAT with head selection by HDP and ROC AUC on the
  first `--hs-size` validation records (50, 100, 500, 1000, all), with
  attention and the final layer, attention only, and, for CoT, the answer
  token without the reasoning span.

Logs and test metrics go to `logs/<index>/<method>/`. The reported numbers ran
the baseline and the logistic regression on CPU and HEAT on GPU; other devices
agree up to floating-point rounding.

## Transfer

```bash
python tasks/transfer.py --model Qwen/Qwen2.5-7B-Instruct --regime cot --baseline beta
```

Fits each method on one dataset and evaluates it on the test split of the
others as is, fine-tuned on n labelled target answers, and fitted from scratch
on them. Needs the HEAT runs of `calibrate_all.py`; results go to
`logs/transfer/<model>_<regime>.csv`.

## Layout

- `services/common/` — generation with per-token scores and attention entropy,
  metrics (ECE, Brier decomposition, AUROC), calibration heads.
- `services/index.py` — record store and train/val/test split.
- `services/feature_cache.py` — calibration inputs of an index, computed once.
- `services/baseline/`, `services/experiment/` — baselines, head selection,
  HEAT and the logistic regression; `services/experiment/transfer.py`.
- `tasks/` — the entry points above.

## License

MIT.
