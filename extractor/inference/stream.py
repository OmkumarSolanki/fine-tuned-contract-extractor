"""Token-streaming helper built on HuggingFace ``TextIteratorStreamer``.

Used by the ``POST /extract/stream`` endpoint to emit partial generation as it
is produced, rather than waiting for the full extraction. Heavy imports are
inside the function so the module is CPU-importable for tests.
"""

from __future__ import annotations

from typing import Any, Iterator


def stream_tokens(
    tokenizer: Any,
    model: Any,
    messages: list[dict],
    max_new_tokens: int = 2048,
) -> Iterator[str]:
    """Yield decoded text chunks as the model generates them.

    Renders ``messages`` through the chat template (with a generation prompt),
    runs ``model.generate`` on a background thread feeding a
    ``TextIteratorStreamer``, and yields the streamer's text pieces on the
    calling thread. Greedy decoding keeps it deterministic.
    """
    import threading  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import TextIteratorStreamer  # noqa: PLC0415

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    def _run() -> None:
        with torch.no_grad():
            model.generate(**generation_kwargs)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for text in streamer:
        if text:
            yield text
    thread.join()
