"""RexReranker-0.6B / Qwen3-Reranker 生成式重排打分器（本地 transformers，GPU/CPU）。

打分逻辑（对齐模型卡 README）：
- chat 模板：system 裁判指令 + user "<Instruct>/<Query>/<Document>"；
- 追加 assistant 后缀 "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"；
- 取最后非 padding token 的 "yes"/"no" logit，score = exp(yes)/(exp(yes)+exp(no))。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def is_generation_reranker(model_name: str) -> bool:
    """判断是否为 Qwen3 生成式重排（RexReranker / Qwen3-Reranker）。"""
    n = (model_name or "").lower()
    return "rex" in n or "qwen3-reranker" in n


class RexRerankerScorer:
    """生成式重排打分器：score_pairs(pairs) -> list[float]（与 FlagReranker 同接口）。"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 1024,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self.device = self._resolve_device(device, torch)
        logger.info("[rex] loading %s on %s ...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.yes_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        logger.info("[rex] loaded: %s (%s)", model_name, self.device)

    @staticmethod
    def _resolve_device(device: str, torch) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    def _messages(self, query: str, doc: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"<Instruct>: {_INSTRUCTION}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]

    def score_pairs(
        self, pairs: list[tuple[str, str]], batch_size: int | None = None
    ) -> list[float]:
        """批处理打分：返回与 pairs 等长的 score 列表（0~1，越高越相关）。"""
        bs = batch_size or self.batch_size
        out: list[float] = []
        for start in range(0, len(pairs), bs):
            batch = pairs[start : start + bs]
            texts = [
                self.tokenizer.apply_chat_template(
                    self._messages(q, d), tokenize=False, add_generation_prompt=False
                )
                + _SUFFIX
                for q, d in batch
            ]
            inputs = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                logits = self.model(**inputs).logits
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            last_logits = logits[self._torch.arange(len(batch), device=self.device), last_idx]
            yes = last_logits[:, self.yes_id].float().exp()
            no = last_logits[:, self.no_id].float().exp()
            out.extend((yes / (yes + no)).cpu().tolist())
        return out
