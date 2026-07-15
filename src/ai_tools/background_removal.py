"""AI-powered background removal using ONNX-based segmentation.

Uses rembg (U2Net ONNX model) for fast, accurate background segmentation
without requiring GPU acceleration — runs efficiently on CPU even on
integrated graphics. Output is an RGBA image with transparency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Remove backgrounds from images using AI segmentation."""

    def __init__(self):
        """Initialize the background remover (lazy-loads the model on first use)."""
        self._session = None

    @property
    def session(self):
        """Lazy-load the rembg session on first access."""
        if self._session is None:
            try:
                from rembg import new_session

                logger.info("Initializing background removal model (first use, may take 30s)...")
                self._session = new_session()
            except ImportError:
                raise ImportError(
                    "rembg is not installed. Run: pip install rembg"
                )
        return self._session

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove the background from an image.

        Args:
            image: PIL Image in any mode (converted to RGB for processing).

        Returns:
            RGBA PIL Image with transparent background.
        """
        from rembg import remove

        # Ensure we're working with RGB for the model.
        if image.mode != "RGB":
            image = image.convert("RGB")

        # rembg.remove handles the segmentation and alpha blending.
        # The session parameter is optional but can speed up repeated calls.
        result = remove(image, session=self.session)

        # Ensure output is RGBA.
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        return result

    def remove_background_array(self, image_array: np.ndarray) -> np.ndarray:
        """Remove background from a numpy RGB array.

        Args:
            image_array: RGB uint8 array (H, W, 3).

        Returns:
            RGBA uint8 array (H, W, 4) with transparent background.
        """
        # Convert array to PIL, process, convert back.
        pil_image = Image.fromarray(image_array)
        result = self.remove_background(pil_image)
        return np.asarray(result)
