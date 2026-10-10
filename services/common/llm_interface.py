from typing import Any, List

import torch
from langchain_core.runnables import Runnable, RunnableConfig

from .calculation_utils import calculate_entropy, calculate_norm_entropy

DEFAULT_TOPK = 30
DEFAULT_LOWER_PROB_LIMIT = 1e-8
DEFAULT_LOWER_LOGIT_LIMIT = -1000
DEFAULT_UPPER_LOGIT_LIMIT = 1000

# Sampling settings of the collection. They only decide which response is
# generated: the recorded scores come from the raw logits, before the
# repetition penalty, the temperature and top-p, which sets every token
# outside the nucleus to -inf.
GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
    "top_p": 0.8,
}


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
        """
        Initialization of a class.

        Args:
            model: Causal LM with ``generate`` supporting logits and attentions.
            tokenizer: Chat-template-capable tokenizer.
            device: torch.device to perform computations on.
            topk: Number of top logits/probs stored per generated token.
            lower_prob_limit: Minimum clamp for attention probabilities.
            lower_logit_limit: Minimum clamp for stored top logits.
            upper_logit_limit: Maximum clamp for stored top logits.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hook_handles = []
        self.device = device
        self.topk = topk
        self.lower_prob_limit = lower_prob_limit
        self.lower_logit_limit = lower_logit_limit
        self.upper_logit_limit = upper_logit_limit

        self.model = self.model.to(device)

    def score_tokens(self, logits, token_ids):
        """Per-token score dicts from raw next-token logits.

        All tokens are scored at once on the logits' device, so a long
        response costs a few kernel launches rather than several per token.

        Args:
            logits: Next-token logits, one ``[V]`` row per scored token.
            token_ids: Ids of the scored tokens, aligned with ``logits``.

        Returns:
            List of dicts with ``token``, ``prob``, ``logit``, ``top_tokens``,
            ``top_logits`` and ``top_probs`` per token.
        """
        logits = torch.stack([row.reshape(-1) for row in logits]).float() # [T, V]
        token_ids = torch.as_tensor(token_ids, device=logits.device).reshape(-1, 1) # [T, 1]
        probs = torch.softmax(logits, dim=-1) # [T, V]

        top_logits, top_tok_ids = torch.topk(logits, self.topk, dim=-1) # [T, TOP_K]
        top_logits = torch.clamp(
            top_logits,
            min=self.lower_logit_limit,
            max=self.upper_logit_limit,
        ).cpu()
        top_probs = torch.topk(probs, self.topk, dim=-1).values.cpu() # [T, TOP_K]

        tokens = self.tokenizer.batch_decode(token_ids.tolist())
        token_probs = probs.gather(1, token_ids).squeeze(1).tolist()
        token_logits = logits.gather(1, token_ids).squeeze(1).tolist()
        top_tokens = self.tokenizer.batch_decode(top_tok_ids.reshape(-1, 1).tolist())

        # Rows are cloned: a pickled view would carry the storage of all tokens.
        return [
            {
                "token": tokens[t],
                "prob": token_probs[t],
                "logit": token_logits[t],
                "top_tokens": top_tokens[t * self.topk : (t + 1) * self.topk],
                "top_logits": top_logits[t].clone(),
                "top_probs": top_probs[t].clone(),
            }
            for t in range(len(tokens))
        ]

    def invoke(
        self,
        messages: List[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        """Generate a response and collect per-token logits, probs, and attention stats.

        Args:
            messages: LangChain message list (system + user expected).
            config: Optional LangChain runnable config.
            **kwargs: Extra generation.

        Returns:
            Dict with ``input_text``, ``output_text``, ``score_data`` (per-token dicts),
            ``attention_entropy``, and ``norm_attention_entropy``.
        """
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
                output_logits=True,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **GENERATION_KWARGS,
            )

        generated_ids = response.sequences[:, model_inputs["input_ids"].shape[1] :].squeeze(0)
        token_data = self.score_tokens(response.logits, generated_ids)

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
