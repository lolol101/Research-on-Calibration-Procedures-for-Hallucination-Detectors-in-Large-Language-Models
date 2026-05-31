import numpy as np
import torch
from tqdm import tqdm

from services.common.calculation_utils import (
    calculate_norm_entropy,
    calculate_agg_features,
)

def retrieve_answer_token_index(tokens):
    """
    Finds the index of the last digit token in a scored sequence.

    Args:
        tokens: List of per-token score dicts with a ``"token"`` key.

    Returns:
        Zero-based index of the answer token, or ``None`` if no digit is found.
    """
    for i in range(len(tokens), 0, -1):
        if tokens[i-1]["token"].isdigit():
            return i - 1
 
def retrieve_reasoning_tokens_range(tokens, start="<think>", end="</think>"):
    """
    Locates token indices spanning a reasoning block in the output.

    Args:
        tokens: List of per-token dicts with ``"token"`` strings.
        start: Marker substring that ends the search for the range start.
        end: Marker substring that ends the search for the range end.

    Returns:
        Tuple ``(start_index, end_index)``; either may stay ``-1`` if not found.
    """
    start_index, end_index = -1, -1
    tmp_text = ""
    for i in range(len(tokens)):
        if start in tmp_text:
            start_index = i
            break
        tmp_text += tokens[i]["token"]

    tmp_text = ""
    for i in range(len(tokens) - 1, 0, -1):
        tmp_text = tokens[i]["token"] + tmp_text
        if end in tmp_text:
            end_index = i
            break

    return (start_index, end_index)

def process_elements_main(
    index_data: np.array, 
    best_layers: torch.Tensor, 
    best_heads: torch.Tensor,
    device: torch.device, 
    attn_only=False,
    verbose=False
    ):
    """
    Builds CoT calibration features over reasoning tokens plus the answer.

    Aggregates attention (and optional final-token) scores along the reasoning
    span via ``calculate_agg_features``, then concatenates answer-token scores.

    Args:
        index_data: Inference records with ``score_data`` and ``dataset_elem``.
        best_layers: Selected layer indices, shape ``[K]``.
        best_heads: Selected head indices aligned with ``best_layers``.
        device: Target device for tensors.
        attn_only: If True, use attention confidences only (no final-token dim).
        verbose: Show tqdm during processing.

    Returns:
        Dict with ``labels`` ``[B]`` and wide ``features`` per sample.
    """
    processed = {}

    labels = []
    for elem in index_data:
        answer_token_index = retrieve_answer_token_index(elem["score_data"])
        
        # TODO: It must be resolved on data collecting stage
        if answer_token_index == len(elem["score_data"]) - 1:
            continue   
                 
        answer_token = elem["score_data"][answer_token_index]["token"]
        answer_label = str(ord(elem["dataset_elem"]["answer"]) - ord("A"))
        labels.append(torch.tensor(answer_token == answer_label))
    processed["labels"] = torch.stack(labels).to(device=device, dtype=torch.long)

    best_layers = best_layers.reshape(-1).to(device)
    best_heads = best_heads.reshape(-1).to(device)
    
    elem_features = []
    for elem in tqdm(
        index_data,
        desc="Processing data...",
        disable=not verbose,
    ):
        
        # TODO: It must be resolved on data collecting stage
        answer_token_index = retrieve_answer_token_index(elem["score_data"])
        if answer_token_index == len(elem["score_data"]) - 1:
            continue
        
        captured_ids = list(range(*retrieve_reasoning_tokens_range(elem["score_data"]))) + \
            [retrieve_answer_token_index(elem["score_data"])]
        captured_ids = torch.tensor(captured_ids, device=device)
        
        norm_attention_entropy = torch.stack(
            elem["norm_attention_entropy"], dim=0
        ).squeeze(-1).to(device) # [T, L, H]
        norm_attention_entropy = norm_attention_entropy.index_select(0, captured_ids)
        best_heads_norm_attn_entropy = norm_attention_entropy[
            :, best_layers, best_heads
        ] # [T, BEST_H]
        best_heads_norm_attn_confidence = 1 - best_heads_norm_attn_entropy
                    
        score_data = elem["score_data"]
        final_token_top_probs = torch.stack(
            [score_data[int(t)]["top_probs"] for t in captured_ids],
            dim=0,
        ).clamp(1e-8).to(device) # [T, TOP_K]
        
        final_token_scores = 1 - calculate_norm_entropy(final_token_top_probs) # [T]
        
        if attn_only:
            reasoning_scores = best_heads_norm_attn_confidence[:-1] # [T - 1, BEST_H + (not ATTN_ONLY)]
            answer_scores = best_heads_norm_attn_confidence[-1] # [1, BEST_H + (not ATTN_ONLY)]
        else:
            reasoning_scores = torch.cat([
                best_heads_norm_attn_confidence,
                final_token_scores.unsqueeze(-1),
            ], dim=1).to(device)[:-1] # [T - 1, BEST_H + (not ATTN_ONLY)]
            answer_scores = torch.cat([
                best_heads_norm_attn_confidence,
                final_token_scores.unsqueeze(-1),
            ], dim=1).to(device)[-1] # [1, BEST_H + (not ATTN_ONLY)]
        
        elem_features.append(
            torch.cat([
                calculate_agg_features(reasoning_scores),
                answer_scores
            ])
        ) # [B, (FEATURES_COUNT + 1) * (BEST_H + (not ATTN_ONLY))]
    
    processed["features"] = torch.stack(
        elem_features
    ).to(device) # [B, (FEATURES_COUNT + 1) * (BEST_H + (not ATTN_ONLY))]
    
    return processed

def process_elements_hal(
    index_data: np.array, 
    layers_count: int, 
    heads_count: int, 
    device: torch.device, 
    verbose=False
    ):
    """
    Extracts per-(layer, head) attention scores at the answer token for head search.

    Same output layout as the cropped ``process_elements_hal`` helper.

    Args:
        index_data: Inference records with attention entropy fields.
        layers_count: Number of transformer layers.
        heads_count: Number of heads per layer.
        device: Target device for tensors.
        verbose: Show tqdm over layers.

    Returns:
        Dict with ``labels`` and ``attn_score{l}_{h}`` entries.
    """
    processed = {}
    
    # Getting labels of if the answer is correct or not 
    labels = []
    for elem in index_data:
        answer_token_index = retrieve_answer_token_index(elem["score_data"])
        
        # TODO: It must be resolved on data collecting stage
        if answer_token_index == len(elem["score_data"]) - 1:
            continue   
                 
        answer_token = elem["score_data"][answer_token_index]["token"]
        answer_label = str(ord(elem["dataset_elem"]["answer"]) - ord("A"))
        labels.append(torch.tensor(answer_token == answer_label))
    processed["labels"] = torch.stack(labels).to(device=device, dtype=torch.long)
   
    attn_entropy = []
    for elem in index_data:
        answer_token_index = retrieve_answer_token_index(elem["score_data"])
        
        # TODO: It must be resolved on data collecting stage
        if answer_token_index == len(elem["score_data"]) - 1:
            continue   
        
        captured_ids = [retrieve_answer_token_index(elem["score_data"])]
        captured_ids = torch.tensor(captured_ids, device=device)
        
        attn_entropy_lh = torch.stack(
            elem["attention_entropy"], dim=0
        ).squeeze(-1).to(device) # [1, L, H]
        
        attn_entropy.append(
            attn_entropy_lh.index_select(0, captured_ids)
        ) # [B, 1, L, H]
        
    attn_entropy = torch.stack(attn_entropy, dim=0).squeeze(1)

    for l in tqdm(
        range(layers_count),
        desc="Processing data...",
        disable=not verbose,
    ):
        for h in range(heads_count):
            processed[f"attn_score{l}_{h}"] = attn_entropy[:, l, h].to(device)        
            
    return processed