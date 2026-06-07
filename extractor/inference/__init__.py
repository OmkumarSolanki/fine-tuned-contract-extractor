"""Inference-time serving helpers for the fine-tuned contract extractor.

Modules:

- :mod:`extractor.inference.prompt` — builds the chat messages using the
  **exact** training-time system/user prompts (imported from
  ``training/prepare_dataset.py`` so they can never drift).
- :mod:`extractor.inference.model_loader` — loads the 4-bit base + LoRA adapter
  once and wraps it in a :class:`FineTunedGenerator` (lazy heavy imports).
- :mod:`extractor.inference.stream` — token-streaming helper built on
  HuggingFace ``TextIteratorStreamer``.

All heavy imports (``unsloth``/``torch``/``transformers``) live inside function
bodies so this package imports cleanly on CPU for the test suite.
"""
