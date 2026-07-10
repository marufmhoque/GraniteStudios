"""Static web gallery publisher.

Resizes images for the web (max 2000px wide) and generates a fully
responsive Masonry-grid ``index.html`` styled with Tailwind CSS (CDN).
The output folder is self-contained and ready to drop on any static host.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from src.engine.universal_loader import UniversalLoader

logger = logging.getLogger(__name__)

MAX_WEB_WIDTH = 2000
THUMB_WIDTH = 800

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .masonry {{ column-gap: 1rem; column-count: 1; }}
    @media (min-width: 640px)  {{ .masonry {{ column-count: 2; }} }}
    @media (min-width: 1024px) {{ .masonry {{ column-count: 3; }} }}
    @media (min-width: 1536px) {{ .masonry {{ column-count: 4; }} }}
    .masonry-item {{ break-inside: avoid; }}
  </style>
</head>
<body class="bg-neutral-950 text-neutral-100 min-h-screen">
  <header class="max-w-7xl mx-auto px-6 pt-12 pb-8">
    <h1 class="text-3xl md:text-4xl font-bold tracking-tight">{title}</h1>
    <p class="mt-2 text-neutral-400">{count} photographs &middot; Published with Granite Studios</p>
  </header>

  <main class="max-w-7xl mx-auto px-6 pb-16">
    <div class="masonry">
{items}
    </div>
  </main>

  <footer class="border-t border-neutral-800 py-8 text-center text-sm text-neutral-500">
    Built with Granite Studios
  </footer>
</body>
</html>
"""

_ITEM_TEMPLATE = """      <div class="masonry-item mb-4">
        <a href="full/{full_name}" target="_blank" rel="noopener">
          <img src="thumbs/{thumb_name}" alt="{alt}" loading="lazy"
               class="w-full rounded-xl shadow-lg shadow-black/40 hover:opacity-90 hover:scale-[1.01] transition duration-200">
        </a>
      </div>
"""


def generate_static_gallery(
    image_paths: list[str | Path],
    output_dir: str | Path,
    title: str = "Granite Studios Gallery",
    quality: int = 88,
) -> Path:
    """Build a self-contained static gallery site.

    Layout of the output directory::

        output_dir/
          index.html
          full/     — display images, max 2000px wide
          thumbs/   — grid thumbnails, max 800px wide

    Returns the path to ``index.html``.
    """
    loader = UniversalLoader()
    output_dir = Path(output_dir)
    full_dir = output_dir / "full"
    thumb_dir = output_dir / "thumbs"
    full_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    items_html: list[str] = []

    for p in image_paths:
        src = Path(p)
        image = loader.load_full(src)
        if image is None:
            logger.warning("Skipping unreadable image: %s", src)
            continue

        base_name = f"{src.stem}.jpg"

        # Full-size web image (max 2000px wide).
        full_img = image
        if full_img.width > MAX_WEB_WIDTH:
            scale = MAX_WEB_WIDTH / full_img.width
            full_img = full_img.resize(
                (MAX_WEB_WIDTH, max(int(round(full_img.height * scale)), 1)),
            )
        full_img.save(full_dir / base_name, "JPEG", quality=quality, optimize=True)

        # Grid thumbnail (max 800px wide).
        thumb = image
        if thumb.width > THUMB_WIDTH:
            scale = THUMB_WIDTH / thumb.width
            thumb = thumb.resize(
                (THUMB_WIDTH, max(int(round(thumb.height * scale)), 1)),
            )
        thumb.save(thumb_dir / base_name, "JPEG", quality=82, optimize=True)

        items_html.append(
            _ITEM_TEMPLATE.format(
                full_name=html.escape(base_name),
                thumb_name=html.escape(base_name),
                alt=html.escape(src.stem.replace("_", " ").replace("-", " ")),
            )
        )
        logger.info("Published: %s", base_name)

    index_path = output_dir / "index.html"
    index_path.write_text(
        _PAGE_TEMPLATE.format(
            title=html.escape(title),
            count=len(items_html),
            items="".join(items_html),
        ),
        encoding="utf-8",
    )
    logger.info("Gallery index written: %s (%d images)", index_path, len(items_html))
    return index_path
