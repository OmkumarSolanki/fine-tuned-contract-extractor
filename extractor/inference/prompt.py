"""Prompt construction for inference — must match training byte-for-byte.

The fine-tuned model was trained on a fixed system turn plus the
``"Extract structured clauses from this contract:\\n\\n<contract>"`` user turn
(see ``training/prepare_dataset.py``). Serving the model with any other prompt
would silently tank accuracy, so we **import** the training constants here
rather than redefining them — there is no second source of truth to drift.

``tests/test_api.py`` additionally asserts these are the same objects as the
training module, so an accidental redefinition fails CI.
"""

from __future__ import annotations

# Single source of truth: the exact strings the model was trained on.
from training.prepare_dataset import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

__all__ = ["SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "build_messages"]


def build_messages(contract_text: str) -> list[dict]:
    """Build the ChatML messages for one contract, in training format.

    Returns the system + user turns (no assistant turn — the model generates
    that). The caller renders these through the tokenizer's chat template with
    ``add_generation_prompt=True``.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(contract_text=contract_text)},
    ]
