"""Panoramic reel maker: turn 3-7 stills into a scrolling 9:16 video.

Builds one giant horizontal panorama from the input images, then animates
a 1080x1920 viewport sliding left-to-right across it — the "seamless
carousel" effect popular on Instagram Reels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy import VideoClip

from src.engine.universal_loader import UniversalLoader

logger = logging.getLogger(__name__)

REEL_WIDTH = 1080
REEL_HEIGHT = 1920


class ScrlReelMaker:
    """Create scrolling panoramic reels from a set of images."""

    def __init__(self, loader: UniversalLoader | None = None):
        self.loader = loader or UniversalLoader()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def create_panoramic_reel(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
        duration: float = 10.0,
        fps: int = 30,
    ) -> Path:
        """Render a 9:16 reel that pans across a stitched panorama.

        Args:
            image_paths: 3 to 7 image files (any supported format).
            output_path: Destination .mp4 path.
            duration: Total video length in seconds.
            fps: Output frame rate.

        Returns:
            The output path.
        """
        if not 3 <= len(image_paths) <= 7:
            raise ValueError(
                f"Panoramic reels need 3-7 images, got {len(image_paths)}."
            )

        panorama = self._build_panorama(image_paths)
        pano_w = panorama.shape[1]

        if pano_w <= REEL_WIDTH:
            raise ValueError(
                "Panorama is narrower than the 1080px viewport — "
                "use wider or more images."
            )

        max_offset = pano_w - REEL_WIDTH

        def make_frame(t: float) -> np.ndarray:
            # Smoothstep easing: gentle start and stop instead of a
            # mechanical linear slide.
            progress = min(max(t / duration, 0.0), 1.0)
            eased = progress * progress * (3.0 - 2.0 * progress)
            x = int(round(eased * max_offset))
            return panorama[:, x : x + REEL_WIDTH, :]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clip = VideoClip(make_frame, duration=duration)
        clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio=False,
            preset="medium",
            logger=None,
        )
        clip.close()

        logger.info("Reel written: %s", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_panorama(self, image_paths: list[str | Path]) -> np.ndarray:
        """Resize all images to reel height and concatenate horizontally."""
        strips: list[np.ndarray] = []
        for p in image_paths:
            image = self.loader.load_full(p)
            if image is None:
                raise IOError(f"Could not load image: {p}")

            # Scale to exactly REEL_HEIGHT, preserving aspect ratio.
            scale = REEL_HEIGHT / image.height
            new_w = max(int(round(image.width * scale)), 1)
            resized = image.resize((new_w, REEL_HEIGHT), Image.Resampling.LANCZOS)
            strips.append(np.asarray(resized))

        return np.concatenate(strips, axis=1)
