"""Pytest bootstrap: ensure the repository root is importable.

The packaged modules (``extractor``, ``training``, ``evaluation``) resolve via
the editable install, but ``scripts/`` is a standalone, non-packaged directory.
Inserting the repo root on ``sys.path`` lets ``tests/test_scripts.py`` import
``scripts.*`` regardless of how pytest is invoked (``pytest`` vs ``python -m
pytest``).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
