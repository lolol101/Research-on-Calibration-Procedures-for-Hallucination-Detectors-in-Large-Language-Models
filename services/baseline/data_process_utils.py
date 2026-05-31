import numpy as np
import torch
from tqdm import tqdm

from services.common.calculation_utils import calculate_norm_entropy

def retrieve_answer_token_index(tokens):
    """
    Finds the index of the last digit token in a scored sequence - the answer token in current context.

    Scans ``tokens`` from the end toward the start and returns the index of
    the rightmost entry whose ``"token"`` field is numeric.

    Args:
        tokens: List of per-token score dicts with a ``"token"`` key.

    Returns:
        Index of the answer token, or ``None`` if no digit is found.
    """
    for i in range(len(tokens), 0, -1):
        if tokens[i-1]["token"].isdigit():
            return i - 1

def process_elements_main(
    index_data: np.array, 
    device: torch.device, 
    verbose=False
    ):
    """
    Builds baseline calibration tensors from indexed inference records.

    For each element in ``index_data``, locates the multiple-choice answer
    token, derives binary correctness labels, and stacks final-token
    confidence (``features``) plus top-k logits (``logits``). Elements whose
    answer token is the last scored position are skipped.

    Args:
        index_data: Array of dicts with ``score_data`` (token scores) and
            ``dataset_elem`` (must include ``answer`` as letter in upper register).
        device: torch.device to perform computations on.
        verbose: If True, show a tqdm progress bar over the feature pass.

    Returns:
        Dict with tensors on ``device``:
        - ``labels``: 0/1 correctness per sample, shape ``[B]``.
        - ``gen_tok_ids``: Index of the generated answer token in ``top_tokens``.
        - ``answer_tok_ids``: Index of the factual answer token in ``top_tokens``.
        - ``features``: Clamped final-token probability, shape ``[B, 1]``.
        - ``logits``: Top-k logits at the answer position, shape ``[B, TOP_K]``.
    """
    processed = {}

    labels = []
    answer_tok_ids, gen_tok_ids = [], []
    for elem in index_data:
        answer_token_index = retrieve_answer_token_index(elem["score_data"])

        # TODO: It must be resolved on data collecting stage
        if answer_token_index == len(elem["score_data"]) - 1:
            continue            
        
        answer_token = elem["score_data"][answer_token_index]["token"]
        answer_label = str(ord(elem["dataset_elem"]["answer"]) - ord("A"))
        labels.append(torch.tensor(answer_token == answer_label))
        gen_tok_ids.append(
            torch.tensor(
                elem["score_data"][answer_token_index]["top_tokens"] \
                    .index(answer_token)
            )
        )
        answer_tok_ids.append(
            torch.tensor(
                elem["score_data"][answer_token_index]["top_tokens"] \
                    .index(answer_label)
            )
        )
    processed["labels"] = torch.stack(labels).to(device=device, dtype=torch.long)
    processed["gen_tok_ids"] = torch.stack(gen_tok_ids).to(device=device, dtype=torch.long)
    processed["answer_tok_ids"] = torch.stack(answer_tok_ids).to(device=device, dtype=torch.long)
    
    elem_features = []
    elem_logits = []
    for elem in tqdm(
        index_data,
        desc="Processing data...",
        disable=not verbose,
    ):
        # TODO: It must be resolved on data collecting stage
        answer_token_index = retrieve_answer_token_index(elem["score_data"])
        if answer_token_index == len(elem["score_data"]) - 1:
            continue
        
        captured_idx = retrieve_answer_token_index(elem["score_data"])
                    
        final_token_prob = torch.tensor(
            elem["score_data"][captured_idx]["prob"]
        ).clamp(1e-8).to(device) # [1]        
        elem_features.append(final_token_prob) # [B, 1]

        final_token_top_logits = elem["score_data"][captured_idx]["top_logits"].to(device) # [TOP_K]
        elem_logits.append(final_token_top_logits) # [B, TOP_K]
    
    processed["features"] = torch.stack(
        elem_features
    ).to(device) # [B, 1]

    processed["logits"] = torch.stack(
        elem_logits
    ).clamp(-100).to(device) # [B, TOP_K]
    
    return processed