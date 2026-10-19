"""Qwen3-Embedding and Qwen3-Reranker run on this machine, in the backend's own process
(`EMBEDDING_PROVIDER=local`, `RERANK_PROVIDER=local`), instead of on SiliconFlow: the same
models, so the reranker's calibrated thresholds hold, with no key and no bill.

PyTorch and the rest come with the optional `local` extra (`uv sync --extra local`) and
are imported only here, as a model is first used, so deployments without them, and
commands that never embed, do not pay for them. Each model is downloaded from Hugging
Face on first use and loaded once, on the GPU if there is one (in half precision there).
Inference runs in a worker thread, one call at a time per model: a call given up on (a
query out of time) still holds its model until it is done."""

import asyncio
import gc
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

M = TypeVar("M")
R = TypeVar("R")

EMBEDDING_BATCH = 16
"""Texts embedded in one forward pass."""
RERANK_BATCH = 8
"""Query and document pairs scored in one forward pass: passages run long."""


class LocalModel(Generic[M]):
    """A model `load` makes on first use, run in a worker thread, one call at a time."""

    def __init__(self, load: Callable[[], M]) -> None:
        self._load = load
        self._model: M | None = None
        self._lock = threading.Lock()

    async def run(self, work: Callable[[M], R]) -> R:
        return await asyncio.to_thread(self._run, work)

    async def aclose(self) -> None:
        """Let the model go and give its memory back, once the call running is done."""
        await asyncio.to_thread(self._unload)

    def _run(self, work: Callable[[M], R]) -> R:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return work(self._model)

    def _unload(self) -> None:
        with self._lock:
            self._model = None
        gc.collect()
        if (torch := sys.modules.get("torch")) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _device_and_dtype() -> tuple[str, Any]:
    import torch

    return ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)


class Qwen3Embedding:
    """Qwen3-Embedding through sentence-transformers, which pools on each text's last token
    as the model card says; vectors come out normalized, as SiliconFlow's do."""

    def __init__(self, name: str) -> None:
        from sentence_transformers import SentenceTransformer

        device, dtype = _device_and_dtype()
        self._model = SentenceTransformer(name, device=device, model_kwargs={"dtype": dtype})

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=EMBEDDING_BATCH, normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()


class Qwen3Reranker:
    """Qwen3-Reranker scored as its model card has it: each document and the query in the
    model's chat template, the score the probability it answers "yes" rather than "no"."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the"
        ' Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
    MAX_LENGTH = 8192

    def __init__(self, name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device, dtype = _device_and_dtype()
        self._tokenizer: Any = AutoTokenizer.from_pretrained(name, padding_side="left")
        model: Any = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self._model = model.to(self._device).eval()
        self._yes = self._tokenizer.convert_tokens_to_ids("yes")
        self._no = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix = self._tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self._suffix = self._tokenizer.encode(self.SUFFIX, add_special_tokens=False)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Each document's relevance to `query`, 0 to 1, in their order."""
        pairs = [
            f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        scores: list[float] = []
        for start in range(0, len(pairs), RERANK_BATCH):
            scores.extend(self._score(pairs[start : start + RERANK_BATCH]))
        return scores

    def _score(self, pairs: list[str]) -> list[float]:
        torch = self._torch
        room = self.MAX_LENGTH - len(self._prefix) - len(self._suffix)
        inputs = self._tokenizer(
            pairs, padding=False, truncation="longest_first", return_attention_mask=False,
            max_length=room,
        )
        inputs["input_ids"] = [self._prefix + ids + self._suffix for ids in inputs["input_ids"]]
        batch = self._tokenizer.pad(inputs, padding=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            last = self._model(**batch).logits[:, -1, :]
        no_yes = torch.stack([last[:, self._no], last[:, self._yes]], dim=1).float()
        return torch.nn.functional.log_softmax(no_yes, dim=1)[:, 1].exp().tolist()
