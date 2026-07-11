"""Static web gallery publisher.

Generates a self-contained, responsive gallery site styled with Tailwind CSS
(CDN). Photos are organized into category tabs (from the labels assigned in
the UI), each rendered as a Masonry grid. Camera + lens info is printed under
every photo, and images can optionally be auto-edited on the way out.

This module is intentionally a pure function over whatever ``gallery`` dict
it's handed — it does no selection of its own. The caller (the Web Export
page in the UI) is responsible for curating that input: only images the
user has manually checked "Publish to Web", sourced from that folder's
``-edit`` subfolder (see ``UniversalLoader.get_edit_folder`` /
``export_keeper``) rather than the original RAW/source files.

The output folder is ready to drop onto any static host.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from src.ai_tools.color_engine import ColorEngine
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
  <header class="max-w-7xl mx-auto px-6 pt-12 pb-6">
    <h1 class="text-3xl md:text-4xl font-bold tracking-tight">{title}</h1>
    <p class="mt-2 text-neutral-400">{count} photographs &middot; {category_count} categories &middot; Published with Granite Studios</p>
  </header>

  <nav class="max-w-7xl mx-auto px-6 pb-8 flex flex-wrap gap-2">
{tabs}
  </nav>

  <main class="max-w-7xl mx-auto px-6 pb-16">
{panels}
  </main>

  <footer class="border-t border-neutral-800 py-8 text-center text-sm text-neutral-500">
    Built with Granite Studios
  </footer>

  <script>
    function showCategory(slug, btn) {{
      document.querySelectorAll('.category-panel').forEach(function (p) {{ p.classList.add('hidden'); }});
      var panel = document.getElementById('cat-' + slug);
      if (panel) panel.classList.remove('hidden');
      document.querySelectorAll('.tab-btn').forEach(function (b) {{
        b.classList.remove('bg-blue-600', 'text-white');
        b.classList.add('bg-neutral-800', 'text-neutral-300');
      }});
      btn.classList.remove('bg-neutral-800', 'text-neutral-300');
      btn.classList.add('bg-blue-600', 'text-white');
    }}
  </script>
</body>
</html>
"""

_TAB_TEMPLATE = (
    '    <button class="tab-btn px-4 py-2 rounded-full text-sm font-medium '
    'transition {active_classes}" onclick="showCategory(\'{slug}\', this)">'
    "{label} <span class=\"opacity-60\">({count})</span></button>\n"
)

_PANEL_TEMPLATE = """    <section id="cat-{slug}" class="category-panel {hidden}">
      <div class="masonry">
{items}
      </div>
    </section>
"""

_ITEM_TEMPLATE = """        <figure class="masonry-item mb-4">
          <a href="{full_href}" target="_blank" rel="noopener">
            <img src="{thumb_href}" alt="{alt}" loading="lazy"
                 class="w-full rounded-xl shadow-lg shadow-black/40 hover:opacity-90 hover:scale-[1.01] transition duration-200">
          </a>
{caption}        </figure>
"""

_CAPTION_TEMPLATE = (
    '          <figcaption class="mt-2 px-1 text-xs text-neutral-400">{gear}</figcaption>\n'
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "category"


def _normalize_item(item) -> dict:
    """Coerce an input item into ``{path, gear}``.

    Accepts a plain path, a ``(path, metadata)`` tuple, or a dict with
    ``path`` and optional ``camera`` / ``lens`` / ``gear`` keys.
    """
    if isinstance(item, dict):
        gear = (item.get("gear") or "").strip()
        if not gear:
            gear = " | ".join(
                p for p in ((item.get("camera") or "").strip(), (item.get("lens") or "").strip()) if p
            )
        return {"path": Path(item["path"]), "gear": gear}

    if isinstance(item, (tuple, list)) and item:
        path = item[0]
        meta = item[1] if len(item) > 1 else None
        gear = getattr(meta, "gear_label", "") if meta is not None else ""
        return {"path": Path(path), "gear": gear}

    return {"path": Path(item), "gear": ""}


def _normalize_gallery(gallery) -> dict[str, list[dict]]:
    """Coerce input into ``{category_name: [ {path, gear}, ... ]}``."""
    if isinstance(gallery, dict):
        source = gallery
    else:  # a flat list of paths / items
        source = {"All Photos": list(gallery)}

    normalized: dict[str, list[dict]] = {}
    for category, items in source.items():
        entries = [_normalize_item(i) for i in items]
        if entries:
            normalized[str(category)] = entries
    return normalized


def generate_static_gallery(
    gallery,
    output_dir: str | Path,
    title: str = "Granite Studios Gallery",
    quality: int = 88,
    auto_fix: bool = False,
) -> Path:
    """Build a self-contained, category-tabbed static gallery site.

    Args:
        gallery: Either a flat list of image paths (rendered as a single
            "All Photos" tab) or a dict mapping category/label names to lists
            of items. Each item may be a path, a ``(path, ImageMetadata)``
            tuple, or a ``{"path", "camera", "lens", "gear"}`` dict.
        output_dir: Destination folder for the generated site.
        title: Page heading.
        quality: JPEG quality for the full-size web images.
        auto_fix: When True, apply :meth:`ColorEngine.smart_auto_fix`
            (white-balance + histogram equalization) to each image.

    Output layout::

        output_dir/
          index.html
          full/<category-slug>/   — display images, max 2000px wide
          thumbs/<category-slug>/ — grid thumbnails, max 800px wide

    Returns the path to ``index.html``.
    """
    loader = UniversalLoader()
    color_engine = ColorEngine() if auto_fix else None

    categories = _normalize_gallery(gallery)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tabs_html: list[str] = []
    panels_html: list[str] = []
    total_images = 0

    for index, (category, items) in enumerate(categories.items()):
        slug = _slugify(category)
        full_dir = output_dir / "full" / slug
        thumb_dir = output_dir / "thumbs" / slug
        full_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        item_html: list[str] = []
        for entry in items:
            src = Path(entry["path"])
            image = loader.load_array(src) if auto_fix else loader.load_full(src)
            if image is None:
                logger.warning("Skipping unreadable image: %s", src)
                continue

            if auto_fix and color_engine is not None:
                from PIL import Image as _Image

                image = _Image.fromarray(color_engine.smart_auto_fix(image))

            base_name = f"{src.stem}.jpg"

            # Full-size web image (max 2000px wide).
            full_img = image
            if full_img.width > MAX_WEB_WIDTH:
                scale = MAX_WEB_WIDTH / full_img.width
                full_img = full_img.resize(
                    (MAX_WEB_WIDTH, max(int(round(full_img.height * scale)), 1))
                )
            full_img.save(full_dir / base_name, "JPEG", quality=quality, optimize=True)

            # Grid thumbnail (max 800px wide).
            thumb = image
            if thumb.width > THUMB_WIDTH:
                scale = THUMB_WIDTH / thumb.width
                thumb = thumb.resize(
                    (THUMB_WIDTH, max(int(round(thumb.height * scale)), 1))
                )
            thumb.save(thumb_dir / base_name, "JPEG", quality=82, optimize=True)

            gear = entry.get("gear", "")
            caption = _CAPTION_TEMPLATE.format(gear=html.escape(gear)) if gear else ""

            item_html.append(
                _ITEM_TEMPLATE.format(
                    full_href=f"full/{html.escape(slug)}/{html.escape(base_name)}",
                    thumb_href=f"thumbs/{html.escape(slug)}/{html.escape(base_name)}",
                    alt=html.escape(src.stem.replace("_", " ").replace("-", " ")),
                    caption=caption,
                )
            )
            total_images += 1
            logger.info("Published [%s]: %s", category, base_name)

        is_first = index == 0
        tabs_html.append(
            _TAB_TEMPLATE.format(
                slug=slug,
                label=html.escape(category),
                count=len(item_html),
                active_classes=(
                    "bg-blue-600 text-white" if is_first else "bg-neutral-800 text-neutral-300"
                ),
            )
        )
        panels_html.append(
            _PANEL_TEMPLATE.format(
                slug=slug,
                hidden="" if is_first else "hidden",
                items="".join(item_html),
            )
        )

    index_path = output_dir / "index.html"
    index_path.write_text(
        _PAGE_TEMPLATE.format(
            title=html.escape(title),
            count=total_images,
            category_count=len(categories),
            tabs="".join(tabs_html),
            panels="".join(panels_html),
        ),
        encoding="utf-8",
    )
    logger.info(
        "Gallery index written: %s (%d images, %d categories)",
        index_path,
        total_images,
        len(categories),
    )
    return index_path
