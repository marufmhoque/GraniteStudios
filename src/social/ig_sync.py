"""Instagram formatting: pad any image to exact IG aspect ratios.

Fits an image inside a 4:5 (feed) or 9:16 (story/reel) canvas without
cropping, filling the letterbox with a heavily blurred, darkened version
of the image itself — the standard professional look.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance

from src.engine.universal_loader import UniversalLoader

logger = logging.getLogger(__name__)

# Canonical Instagram output sizes.
IG_FORMATS = {
    "feed": (1080, 1350),   # 4:5
    "story": (1080, 1920),  # 9:16
}


class IgFormatter:
    """Pad images with a blurred background to exact Instagram ratios."""

    def __init__(self, loader: UniversalLoader | None = None, blur_radius: int = 40):
        self.loader = loader or UniversalLoader()
        self.blur_radius = blur_radius

    def format_image(
        self,
        image_path: str | Path,
        target: str = "feed",
    ) -> Image.Image:
        """Return an IG-ready image padded to exactly 4:5 or 9:16.

        Args:
            image_path: Any supported image file.
            target: "feed" (1080x1350, 4:5) or "story" (1080x1920, 9:16).
        """
        if target not in IG_FORMATS:
            raise ValueError(
                f"Unknown target {target!r}. Choose from: {', '.join(IG_FORMATS)}"
            )

        image = self.loader.load_full(image_path)
        if image is None:
            raise IOError(f"Could not load image: {image_path}")

        canvas_w, canvas_h = IG_FORMATS[target]

        # --- Background: cover-fit, blurred, darkened ------------------- #
        bg_scale = max(canvas_w / image.width, canvas_h / image.height)
        bg = image.resize(
            (int(round(image.width * bg_scale)), int(round(image.height * bg_scale))),
            Image.Resampling.LANCZOS,
        )
        left = (bg.width - canvas_w) // 2
        top = (bg.height - canvas_h) // 2
        bg = bg.crop((left, top, left + canvas_w, top + canvas_h))
        bg = bg.filter(ImageFilter.GaussianBlur(self.blur_radius))
        bg = ImageEnhance.Brightness(bg).enhance(0.6)

        # --- Foreground: contain-fit, centered -------------------------- #
        fg_scale = min(canvas_w / image.width, canvas_h / image.height)
        fg = image.resize(
            (int(round(image.width * fg_scale)), int(round(image.height * fg_scale))),
            Image.Resampling.LANCZOS,
        )
        offset = ((canvas_w - fg.width) // 2, (canvas_h - fg.height) // 2)
        bg.paste(fg, offset)

        return bg

    def export(
        self,
        image_path: str | Path,
        output_path: str | Path,
        target: str = "feed",
        quality: int = 92,
    ) -> Path:
        """Format and save an image as a high-quality JPEG."""
        formatted = self.format_image(image_path, target=target)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatted.save(output_path, "JPEG", quality=quality, optimize=True)
        logger.info("IG %s export written: %s", target, output_path)
        return output_path

    def export_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path,
        target: str = "feed",
        quality: int = 92,
    ) -> list[Path]:
        """Format a batch of images into ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for p in image_paths:
            src = Path(p)
            dest = output_dir / f"{src.stem}_ig_{target}.jpg"
            try:
                written.append(self.export(src, dest, target=target, quality=quality))
            except Exception:
                logger.exception("Failed to format %s", src)
        return written
