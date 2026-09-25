"""Skip a test that needs scikit-learn when Windows refuses to load it.

Windows 11's Smart App Control lets an unsigned file run only if Microsoft's reputation service
knows it, and it refuses some of scikit-learn 1.9.1's compiled files: the import then fails with
"An Application Control policy has blocked this file" (PROGRESS.md §7). Only that failure skips a
test. CI and any other machine still run it, and a scikit-learn that is missing or broken for any
other reason still fails loudly.
"""

from __future__ import annotations

import pytest


def _blocked_by_windows() -> str | None:
    try:
        import sklearn.linear_model  # noqa: F401
        import sklearn.metrics  # noqa: F401
        import sklearn.preprocessing  # noqa: F401
    except ImportError as error:
        if "Application Control" in str(error):
            return str(error).splitlines()[0]
    return None


BLOCKED = _blocked_by_windows()
"""The loader's message when Windows blocks scikit-learn here, else ``None``."""

needs_sklearn = pytest.mark.skipif(
    BLOCKED is not None,
    reason=f"Windows Smart App Control blocks scikit-learn here (PROGRESS.md §7): {BLOCKED}",
)
"""For a test that fits a model with scikit-learn, or runs a script that imports it."""
