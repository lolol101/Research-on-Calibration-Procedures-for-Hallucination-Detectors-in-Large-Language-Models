import numpy as np
import torch
from tqdm import tqdm

from services.common.calculation_utils import calculate_norm_entropy

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

def process_elements_main(
    index_data: np.array, 
    best_layers: torch.Tensor, 
    best_heads: torch.Tensor,
    device: torch.device, 
    attn_only=False,
    verbose=False
    ):
    """
    Builds cropped-attention calibration features for the answer token only.

    Stacks normalized attention confidence at selected (layer, head) pairs and
    optionally appends final-token confidence from top-k probabilities.

    Args:
        index_data: Inference records with ``score_data`` and ``dataset_elem``.
        best_layers: Layer indices from head selection, shape ``[K]``.
        best_heads: Head indices aligned with ``best_layers``.
        device: Target device for output tensors.
        attn_only: If True, omit final-token confidence features.
        verbose: Show tqdm during the feature pass.

    Returns:
        Dict with ``labels`` ``[B]`` and ``features`` ``[B, K or K+1]``.
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

    best_layers = best_layers.detach().reshape(-1).to(device)
    best_heads = best_heads.detach().reshape(-1).to(device)
    
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
        
        captured_ids = [retrieve_answer_token_index(elem["score_data"])]
        captured_ids = torch.tensor(captured_ids, device=device)
        
        norm_attention_entropy = torch.stack(
            elem["norm_attention_entropy"], dim=0
        ).squeeze(-1).to(device, torch.float32) # [1, L, H]
        norm_attention_entropy = norm_attention_entropy.index_select(0, captured_ids)
        best_heads_norm_attn_entropy = norm_attention_entropy[
            :, best_layers, best_heads
        ] # [1, BEST_H]
        best_heads_norm_attn_confidence = 1 - best_heads_norm_attn_entropy
                    
        score_data = elem["score_data"]
        final_token_top_probs = torch.stack(
            [score_data[int(t)]["top_probs"] for t in captured_ids],
            dim=0,
        ).clamp(1e-8).to(device) # [1, TOP_K]
        
        final_token_scores = 1 - calculate_norm_entropy(final_token_top_probs) # [1]
        
        if attn_only:
            answer_scores = best_heads_norm_attn_confidence # [1, BEST_H + (not ATTN_ONLY)]
        else:
            answer_scores = torch.cat([
                final_token_scores.unsqueeze(-1),
                best_heads_norm_attn_confidence,
            ], dim=1).to(device, torch.float32) # [1, BEST_H + (not ATTN_ONLY)]
        
        elem_features.append(answer_scores.squeeze(0)) # [B, (BEST_H + (not ATTN_ONLY))]
    
    processed["features"] = torch.stack(
        elem_features
    ).to(device) # [B, BEST_H + (not ATTN_ONLY)]
    
    return processed

def process_elements_hal(
    index_data: np.array, 
    layers_count: int, 
    heads_count: int, 
    device: torch.device, 
    verbose=False
    ):
    """
    Extracts per-(layer, head) attention scores for head selection.

    Used before ``find_best_layer_head_*`` to populate ``attn_score{l}_{h}``
    keys and binary ``labels``.

    Args:
        index_data: Inference records with attention entropy fields.
        layers_count: Number of layers to iterate.
        heads_count: Number of heads per layer.
        device: Target device for score tensors.
        verbose: Show tqdm over layers.

    Returns:
        Dict with ``labels`` and ``attn_score{l}_{h}`` for each layer/head.
    """
    processed = {}
    
    # Getting labels of if the answer is correct or not 
    labels = []
    for elem in index_data:
        
        # TODO: It must be resolved on data collecting stage
        answer_token_index = retrieve_answer_token_index(elem["score_data"])
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

        
    attn_entropy = torch.stack(attn_entropy, dim=0).squeeze(1) # [B, L, H]

    for l in tqdm(
        range(layers_count),
        desc="Processing data...",
        disable=not verbose,
    ):
        for h in range(heads_count):
            processed[f"attn_score{l}_{h}"] = attn_entropy[:, l, h].to(device)
        
    return processed