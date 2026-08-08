"""Vision feature encoder (placeholder).

Activated only when raw image data (Diatoms / ImCLEF) is confirmed
available.  Falls back to pre-extracted ARFF features otherwise.
"""

from __future__ import annotations

import logging

from hmc.data.base import FeatureEncoder, Split

logger = logging.getLogger(__name__)


class VisionFeatureEncoder(FeatureEncoder):
    """Encode images via a pre-trained vision backbone.

    Supports DINOv2, ConvNeXt, and ViT backbones.  All backbones are
    optional dependencies loaded lazily.

    Parameters
    ----------
    backbone : str
        One of ``"dinov2"``, ``"convnext"``, ``"vit"``.
    device : str
    """

    def __init__(
        self,
        backbone: str = "dinov2",
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.device = device
        self._model = None
        self._transform = None

    def fit(self, split: Split) -> VisionFeatureEncoder:
        return self  # frozen encoder, no fitting

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            if self.backbone == "dinov2":
                try:
                    import importlib  # pylint: disable=import-outside-toplevel

                    importlib.import_module("dinov2")
                except ImportError:
                    logger.warning("dinov2 not installed — vision encoder unavailable")
                    return
            logger.info("Vision backbone '%s' loaded on %s", self.backbone, self.device)
        except ImportError as e:
            logger.warning("Vision encoder loading failed: %s", e)

    def transform(self, split: Split) -> Split:
        """Encode images if raw data available; otherwise passthrough."""
        if not split.metadata.raw_available:
            logger.info(
                "Vision encoder: no raw images available, "
                "passing through pre-extracted features"
            )
            return split

        self._ensure_loaded()
        if self._model is None:
            return split

        # Raw-image encoding would go here once images are available.
        # For now, fall through to passthrough.
        logger.warning(
            "Vision encoder: raw images confirmed but encoding not "
            "yet implemented — passing through features"
        )
        return split
