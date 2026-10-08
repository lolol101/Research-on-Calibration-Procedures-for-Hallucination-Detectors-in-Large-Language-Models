import os
from typing import Callable, Dict, Optional

import torch
from tqdm import tqdm

from .baseline import data_process_utils as baseline_processing
from .common.datasets import COT_REGIME, CROPPED_REGIME, letter_answer_label
from .common.llm_interface import SCORE_SOURCE
from .experiment.cot import data_process_utils as cot_processing
from .experiment.cropped import data_process_utils as cropped_processing
from .index import Index, split_positions

PROCESSING = {
    COT_REGIME: cot_processing,
    CROPPED_REGIME: cropped_processing,
}

# Bumped whenever the cached quantities change, so stale files are rebuilt.
CACHE_VERSION = 1


class CachedSplit:
    """Split data held in memory, with the ``get`` interface of ``IndexDataset``."""

    def __init__(self, data: Dict[str, torch.Tensor]):
        """Wraps a dict of tensors whose first dimension indexes samples."""
        self.data = data

    def __len__(self):
        """Number of samples in this split."""
        return len(self.data["labels"])

    def get(self, start=0, end=None):
        """Return the samples ``[start:end]`` of every tensor."""
        return {k: v[start:end] for k, v in self.data.items()}


class FeatureCache:
    """
    Calibration inputs of every record, computed once per index for all heads.

    Every run of ``calibrate.py`` used to unpickle the raw index several times
    and derive the same quantities again, only for a different set of selected
    heads. The cache stores them for all (layer, head) pairs, next to the index
    as ``<index>_features.pt``:

    - ``labels``: answer correctness, ``[N]``;
    - ``attention_entropy``: answer-token entropy per head for head selection,
      ``[N, L, H]``;
    - ``scores``: calibration features of the regime for all heads, blocks of
      ``1 + L * H`` columns with the final-token score first, ``[N, B * (1 + L * H)]``;
    - ``baseline``: final-token probability, top-k logits and token ids.

    The tensors are produced by the same ``process_elements_*`` functions the
    ``IndexDataset`` path uses, and every feature column depends on its own
    head only, so selecting columns reproduces the per-selection computation.
    Records those functions skip are skipped here too; ``positions`` maps rows
    back to index positions.

    Indices collected before scores were taken from the raw logits hold
    final-layer scores after the repetition penalty, the temperature and
    top-p; they are refused.
    """

    def __init__(
        self,
        index: Index,
        regime: str,
        answer_label: Callable[[dict], str] = letter_answer_label,
        chunk_size: int = 256,
        verbose: bool = False,
    ):
        """Loads the cache of ``index``, building it first if missing or stale.

        Args:
            index: ``Index`` with collected model responses.
            regime: ``"cot"`` or ``"cropped"``.
            answer_label: Callable mapping ``dataset_elem`` to the expected answer.
            chunk_size: Records processed at once while building.
            verbose: If True, show build progress.

        Raises:
            ValueError: If the index holds final-layer scores taken after the
                sampling processors.
        """
        self.index = index
        self.path = f"{index.base_filename}_features.pt"
        if index.load_records([0])[0].get("score_source") != SCORE_SOURCE:
            raise ValueError(
                f"{index.base_filename} holds final-layer scores taken after the sampling "
                f"processors; collect it again with tasks/launch.py."
            )
        meta = {
            "version": CACHE_VERSION,
            "regime": regime,
            "answer_label": answer_label.__name__,
            "records": len(index),
            "data_bytes": os.path.getsize(index.data_filename),
        }

        cache = None
        if os.path.exists(self.path):
            cache = torch.load(self.path, weights_only=True)
            if cache["meta"] != meta:
                cache = None
        if cache is None:
            cache = self._build(regime, answer_label, chunk_size, verbose)
            cache["meta"] = meta
            # Written to a temporary file first, so an interrupted build never
            # leaves a truncated cache behind.
            torch.save(cache, self.path + ".tmp")
            os.replace(self.path + ".tmp", self.path)

        self.labels = cache["labels"]
        self.attention_entropy = cache["attention_entropy"]
        self.scores = cache["scores"]
        self.baseline = cache["baseline"]
        self._row_of = {int(p): row for row, p in enumerate(cache["positions"])}

    def _build(self, regime, answer_label, chunk_size, verbose):
        """Process the index in chunks and collect the cached tensors."""
        cpu = torch.device("cpu")
        first = self.index.load_records([0])[0]
        layers_count, heads_count = torch.stack(
            first["attention_entropy"], dim=0
        ).squeeze(-1).shape[1:] # [T, L, H]
        all_layers = torch.arange(layers_count).repeat_interleave(heads_count) # [L * H]
        all_heads = torch.arange(heads_count).repeat(layers_count) # [L * H]
        processing = PROCESSING[regime]

        parts = {"positions": [], "labels": [], "attention_entropy": [], "scores": []}
        baseline_parts = {}
        for start in tqdm(
            range(0, len(self.index), chunk_size),
            desc="Building feature cache...",
            disable=not verbose,
        ):
            chunk_positions = list(range(start, min(start + chunk_size, len(self.index))))
            records = self.index.load_records(chunk_positions)

            # Same skip rule as the process_elements_* functions.
            kept = [
                position for position, elem in zip(chunk_positions, records)
                if processing.retrieve_answer_token_index(elem["score_data"])
                != len(elem["score_data"]) - 1
            ]
            if not kept:
                continue

            selection = processing.process_elements_hdp(
                records, layers_count, heads_count, cpu, answer_label=answer_label
            )
            features = processing.process_elements_main(
                records, all_layers, all_heads, cpu, attn_only=False, answer_label=answer_label
            )
            baseline = baseline_processing.process_elements_main(
                records, cpu, answer_label=answer_label
            )

            parts["positions"].append(torch.tensor(kept, dtype=torch.long))
            parts["labels"].append(selection["labels"])
            parts["attention_entropy"].append(
                torch.stack(
                    [
                        selection[f"attn_score{l}_{h}"]
                        for l in range(layers_count)
                        for h in range(heads_count)
                    ],
                    dim=1,
                ).reshape(-1, layers_count, heads_count) # [B, L, H]
            )
            parts["scores"].append(features["features"].to(torch.float32)) # [B, blocks * (1 + L * H)]
            for key in ("features", "logits", "gen_tok_ids", "answer_tok_ids"):
                baseline_parts.setdefault(key, []).append(baseline[key])

        cache = {key: torch.cat(values) for key, values in parts.items()}
        cache["baseline"] = {key: torch.cat(values) for key, values in baseline_parts.items()}
        return cache

    def _rows(self, split: str, split_seed: Optional[int]) -> torch.Tensor:
        """Cache rows of a split, in the order ``IndexDataset`` would yield them."""
        positions = split_positions(self.index, split, split_seed=split_seed)
        return torch.tensor(
            [self._row_of[p] for p in positions if p in self._row_of], dtype=torch.long
        )

    def baseline_split(self, split: str, device: torch.device, split_seed: Optional[int] = None):
        """Baseline inputs of a split, as ``baseline.process_elements_main`` returns them.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            device: Target device for the tensors.
            split_seed: Seed of the train/val/test shuffle; ``None`` keeps order.

        Returns:
            ``CachedSplit`` with ``labels``, ``features``, ``logits``,
            ``gen_tok_ids`` and ``answer_tok_ids``.
        """
        rows = self._rows(split, split_seed)
        data = {"labels": self.labels[rows]}
        data.update({key: value[rows] for key, value in self.baseline.items()})
        return CachedSplit({key: value.to(device) for key, value in data.items()})

    def attention_entropy_split(self, split: str, device: torch.device, split_seed: Optional[int] = None):
        """Answer-token attention entropy of every head for a split, in float32.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            device: Target device for the tensors.
            split_seed: Seed of the train/val/test shuffle; ``None`` keeps order.

        Returns:
            ``CachedSplit`` with ``labels`` and ``attention_entropy``, ``[B, L, H]``.
        """
        rows = self._rows(split, split_seed)
        return CachedSplit({
            "labels": self.labels[rows].to(device),
            "attention_entropy": self.attention_entropy[rows].to(device, torch.float32),
        })

    def head_selection_data(
        self,
        hs_size: int,
        device: torch.device,
        split_seed: Optional[int] = None,
    ) -> dict:
        """Head-selection input from the leading val records, as ``process_elements_hdp`` returns it.

        Args:
            hs_size: Number of leading val records; 0 uses the whole split.
            device: Target device for the tensors.
            split_seed: Seed of the train/val/test shuffle; ``None`` keeps order.

        Returns:
            Dict with ``labels`` and ``attn_score{l}_{h}`` for every head.
        """
        rows = self._rows("val", split_seed)
        if hs_size > 0:
            rows = rows[:hs_size]
        # Stored in bfloat16; selection statistics are averaged in float32, or
        # most heads tie at bfloat16 resolution.
        entropy = self.attention_entropy[rows].to(device, torch.float32) # [B, L, H]
        data = {"labels": self.labels[rows].to(device)}
        for l in range(entropy.shape[1]):
            for h in range(entropy.shape[2]):
                data[f"attn_score{l}_{h}"] = entropy[:, l, h]
        return data

    def experiment_split(
        self,
        split: str,
        best_layers: torch.Tensor,
        best_heads: torch.Tensor,
        device: torch.device,
        attn_only: bool = False,
        split_seed: Optional[int] = None,
    ):
        """Calibration features of a split for the selected heads.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            best_layers: Selected layer indices, shape ``[K]``.
            best_heads: Head indices aligned with ``best_layers``.
            device: Target device for the tensors.
            attn_only: If True, omit final-token confidence features.
            split_seed: Seed of the train/val/test shuffle; ``None`` keeps order.

        Returns:
            ``CachedSplit`` with ``labels`` and ``features``, laid out as the
            regime's ``process_elements_main`` lays them out.
        """
        heads_count = self.attention_entropy.shape[2]
        block_width = 1 + self.attention_entropy.shape[1] * heads_count
        blocks = self.scores.shape[1] // block_width
        head_columns = 1 + best_layers.reshape(-1).cpu() * heads_count + best_heads.reshape(-1).cpu() # [K]
        columns = torch.cat([
            torch.cat([
                torch.tensor([] if attn_only else [0], dtype=torch.long),
                head_columns,
            ]) + block * block_width
            for block in range(blocks)
        ]) # [blocks * (K + (not attn_only))]

        rows = self._rows(split, split_seed)
        return CachedSplit({
            "labels": self.labels[rows].to(device),
            "features": self.scores[rows][:, columns].to(device),
        })
