"""Lazy TabFM adapter — loads TabFM only when needed.

TabFM is an optional dependency with a non-commercial license.
If not installed, a clear install message is shown instead of an
import error crashing ``hmc``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TABFM_AVAILABLE = None  # tri-state: None = unchecked, True/False = cached


def _check_tabfm() -> bool:
    """Return True if TabFM is importable."""
    global _TABFM_AVAILABLE  # pylint: disable=global-statement
    if _TABFM_AVAILABLE is None:
        try:
            import tabfm  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import,import-error

            _TABFM_AVAILABLE = True
        except ImportError:
            _TABFM_AVAILABLE = False
    return _TABFM_AVAILABLE


def require_tabfm():
    """Raise :class:`ImportError` with a clear install message if TabFM
    is not available."""
    if not _check_tabfm():
        raise ImportError(
            "TabFM is required but not installed. "
            "Install with: pip install tabfm[pytorch]\n"
            "Note: TabFM v1.0 weights have a non-commercial license. "
            "See https://github.com/liam-sbhoo/tabfm for details."
        )


class TabFMAdapter:
    """Lazy wrapper around TabFM for conditional node classification.

    Parameters
    ----------
    model_name : str
        TabFM checkpoint name (e.g. ``"tabfm-v1"``).
    device : str
        PyTorch device string.
    kwargs :
        Passed to the TabFM constructor.
    """

    def __init__(
        self,
        model_name: str = "tabfm-v1",
        device: str = "cpu",
        **kwargs,
    ):
        require_tabfm()
        self.model_name = model_name
        self.device = device
        self._kwargs = kwargs
        self._model: Any = None
        self._loaded = False

    def load(self) -> TabFMAdapter:
        """Load the TabFM model (call once before training or inference)."""
        if self._loaded:
            return self

        import tabfm  # pylint: disable=import-outside-toplevel,import-error

        logger.info("Loading TabFM model '%s' on %s ...", self.model_name, self.device)
        self._model = tabfm.TabFM(
            model_name=self.model_name,
            device=self.device,
            **self._kwargs,
        )
        self._loaded = True
        return self

    @property
    def model(self):
        """Access the underlying TabFM model (auto-loads on first access)."""
        if not self._loaded:
            self.load()
        return self._model

    def encode(self, X, context: Any | None = None):
        """Encode tabular features with optional context."""
        return self.model.encode(X, context=context)

    def predict(self, X, context: Any | None = None):
        """Return class probabilities for the given features."""
        return self.model.predict(X, context=context)
