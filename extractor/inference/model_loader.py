"""Load the 4-bit base + LoRA adapter once and wrap it for serving.

The FastAPI app loads a single :class:`FineTunedGenerator` at startup and reuses
it across requests (model loading is expensive; generation is cheap). All heavy
imports (``unsloth``/``torch``/``transformers``) are inside function bodies so
this module imports cleanly on CPU — the API tests inject a mock generator and
never touch a real model.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from extractor.inference.stream import stream_tokens

logger = logging.getLogger(__name__)

# Default location the training driver saves the adapter to (gitignored).
DEFAULT_ADAPTER_PATH = "checkpoints/contract-extractor/final-adapter"
DEFAULT_MAX_SEQ_LENGTH = 8192


class FineTunedGenerator:
    """Holds a tokenizer + model and runs chat-template generation.

    Greedy decoding (``do_sample=False``) keeps outputs deterministic and
    matches how the model was evaluated. The two public methods are what the
    API depends on; they can be reproduced by a ``unittest.mock`` in tests.
    """

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, messages: list[dict], max_new_tokens: int = 2048) -> tuple[str, int]:
        """Greedy-decode the assistant turn. Returns ``(text, n_new_tokens)``."""
        import torch  # noqa: PLC0415

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, int(new_tokens.shape[0])

    def stream(self, messages: list[dict], max_new_tokens: int = 2048) -> Iterator[str]:
        """Yield text chunks as they are generated (for SSE)."""
        return stream_tokens(self.tokenizer, self.model, messages, max_new_tokens)


def load_generator(
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> FineTunedGenerator:
    """Load the base + adapter via Unsloth and return a ready generator.

    Resolves the base model from the adapter's ``adapter_config.json`` and sets
    the model up for fast inference. GPU-only (bitsandbytes + CUDA). Lazy-imports
    ``unsloth`` so importing this module stays CPU-safe.
    """
    from unsloth import FastLanguageModel  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    logger.info("Loaded fine-tuned generator from %s", adapter_path)
    return FineTunedGenerator(tokenizer, model)
