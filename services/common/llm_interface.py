from typing import Any, List

import torch
from langchain_core.runnables import Runnable, RunnableConfig

from .calculation_utils import calculate_entropy, calculate_norm_entropy

DEFAULT_TOPK = 30
DEFAULT_LOWER_PROB_LIMIT = 1e-8
DEFAULT_LOWER_LOGIT_LIMIT = -1000
DEFAULT_UPPER_LOGIT_LIMIT = 1000


class LLMInterface(Runnable):
    def __init__(
        self,
        model,
        tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        *,
        topk: int = DEFAULT_TOPK,
        lower_prob_limit: float = DEFAULT_LOWER_PROB_LIMIT,
        lower_logit_limit: float = DEFAULT_LOWER_LOGIT_LIMIT,
        upper_logit_limit: float = DEFAULT_UPPER_LOGIT_LIMIT,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_handles = []
        self.device = device
        self.topk = topk
        self.lower_prob_limit = lower_prob_limit
        self.lower_logit_limit = lower_logit_limit
        self.upper_logit_limit = upper_logit_limit

        self.model = self.model.to(device)

    def invoke(
        self,
        messages: List[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        hf_messages = [
            {"role": "system", "content": messages.messages[0].content},
            {"role": "user", "content": messages.messages[1].content},
        ]

        inputs = self.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
        ).to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            response = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.05,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        token_data = []
        generated_ids = response.sequences[:, model_inputs["input_ids"].shape[1] :].squeeze(0)
        for i, logits in enumerate(response.scores):
            logits = logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)

            top_logits, top_tok_ids = torch.topk(logits, self.topk, dim=-1)
            top_logits = torch.clamp(
                top_logits,
                min=self.lower_logit_limit,
                max=self.upper_logit_limit,
            )

            top_probs, _ = torch.topk(probs, self.topk, dim=-1)
            top_probs = top_probs.cpu()

            top_tokens = []
            for j in range(top_tok_ids.shape[-1]):
                top_tokens.append(self.tokenizer.decode(top_tok_ids[j].item()))

            token_id = generated_ids[i]
            token_data.append(
                {
                    "token": self.tokenizer.decode(token_id),
                    "prob": probs[token_id].cpu().item(),
                    "logit": logits[token_id].cpu().item(),
                    "top_tokens": top_tokens,
                    "top_logits": top_logits.cpu(),
                    "top_probs": top_probs,
                }
            )

        output_attetntions = [
            torch.stack(token_attention).squeeze(1).clamp(min=self.lower_prob_limit).cpu()
            for token_attention in response.attentions[1:]
        ]

        attn_entropy = [calculate_entropy(x) for x in output_attetntions]

        norm_attn_entropy = [calculate_norm_entropy(x) for x in output_attetntions]

        return {
            "input_text": self.tokenizer.decode(
                response.sequences[:, : model_inputs["input_ids"].shape[1]].cpu()[0],
                skip_special_tokens=True,
            ),
            "output_text": self.tokenizer.decode(
                response.sequences[:, model_inputs["input_ids"].shape[1] :].cpu()[0],
                skip_special_tokens=True,
            ),
            "score_data": token_data,
            "attention_entropy": attn_entropy,
            "norm_attention_entropy": norm_attn_entropy,
        }
