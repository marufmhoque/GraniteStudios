# Granite Studios

### [⬇️ Download the installer](https://github.com/marufmhoque/GraniteStudios/releases/latest/download/Install-GraniteStudios.bat)

One file. Download it, double-click it, done — it fetches the latest version
and installs it for you. Re-run the same file any time to update.
[![Publish installer](https://github.com/marufmhoque/GraniteStudios/actions/workflows/installer.yml/badge.svg)](https://github.com/marufmhoque/GraniteStudios/actions/workflows/installer.yml)

> **📋 [Product Requirements Document (PRD.md)](PRD.md)** — who it is for, the invariants it must never break (originals are never modified or deleted), every requirement with its measured evidence, known limitations, and the roadmap.

---

**Local-first AI photo and video editing suite** for photographers and content creators. Designed to run smoothly on modest hardware (i5-8350U + 32GB RAM), but scales to powerful machines with CUDA or Apple Silicon.

- 📸 **Universal input**: RAW (Nikon, Canon, Sony, etc.), HEIC, JPEG, PNG, and more
- 🤖 **Smart culling**: Perceptual-hash duplicate detection, burst grouping, sharpness scoring
- 🎨 **AI color grading**: Selective color isolation, cinematic matrix filters, one-click auto-edit
- 📹 **Reel maker**: Turn 3–7 images into a scrolling 9:16 panoramic video
- 🌐 **Web publisher**: Responsive Masonry gallery with category tabs and camera metadata
- 💾 **Edit-safe workflow**: Edits and "keepers" are written to an `-edit` subfolder; your originals are never modified or moved

## Quick Start

### Windows (recommended)

1. **[Download `Install-GraniteStudios.bat`](https://github.com/marufmhoque/GraniteStudios/releases/latest/download/Install-GraniteStudios.bat)**
2. Double-click it.
3. Launch Granite Studios from the Desktop shortcut or the Start Menu.

That single file is all you need — it downloads the app itself, so there is no
repository to clone and nothing else to fetch. **Run it again whenever you want
to update**: it pulls the latest version and installs over the old one, leaving
your photos and `-edit` folders alone.

The download link always serves the current version — a GitHub Action
re-publishes it on every change to `main`, and the file fetches `main` when it
runs, so it cannot go stale.

<details>
<summary>Already have the repository cloned?</summary>

Open the `install` folder and double-click **`Install Granite Studios.bat`**
instead — it installs from your local copy rather than downloading. Or run the
installer directly against any source tree:

```powershell
powershell -ExecutionPolicy Bypass -File install\install.ps1 -SourceDir C:\path\to\GraniteStudios
```

</details>

The installer copies the app to `%LOCALAPPDATA%\Granite Studios`, builds it a
private Python environment so it cannot disturb any other Python on the
machine, pre-downloads the background-removal model, registers the app in
**Settings > Apps**, and finishes by actually starting the app headlessly to
confirm the install works rather than just assuming it did.

It needs Python 3.10+ installed once; if it cannot find one it says so and
links to the download instead of failing cryptically. To remove the app, use
**Settings > Apps > Granite Studios** — your photos, videos and `-edit`
folders are never touched.

> **Why an installer rather than running it in place:** a checkout often lives
> in a synced folder (Google Drive, OneDrive, Dropbox). Those paths disappear
> when sync pauses, are slow to read, and break `pip`. Installing to a local
> directory makes launching deterministic.

### If the install fails

Every run writes a full log, including pip's output and any error:

```
%TEMP%\GraniteStudios-install.log   <- what the installer did
%TEMP%\GraniteStudios-pip.log       <- full dependency install detail
```

On failure the installer prints that path, opens the file in Notepad, and
holds the window open, so the error cannot scroll away or vanish when the
window closes. Send that log if you want help reading it.

Two things it checks for and reports in plain language:

- **Granite Studios is already running.** A running copy holds its own files
  open, so replacing them fails. Close the app's window and run the installer
  again.
- **No Python 3.10+.** It links to the download rather than failing
  cryptically. Tick *Add python.exe to PATH* in the Python installer.

See [README_INSTALL.md](README_INSTALL.md) for SmartScreen notes.

### macOS
1. Download the latest `.dmg` from [Releases](https://github.com/marufmhoque/GraniteStudios/releases)
2. Drag **Granite Studios** into Applications
3. Right-click the app > **Open** (bypasses Gatekeeper on first launch)

### From Source
```bash
git clone https://github.com/marufmhoque/GraniteStudios.git
cd GraniteStudios
pip install -r requirements.txt
python main.py
```

### In the Browser
A hosted web version runs the same engines with no install — upload photos or
video, edit, and download the results. See [web/README.md](web/README.md) for the
architecture and deployment steps.

## Features

### Blur triage
- **Find Blurry** scores every frame and separates *motion blur* from *missed focus*: a smear destroys detail along one axis, so the gradients bunch in one direction, whereas defocus softens every direction equally. Motion direction is reported too (±15° across 15 synthetic test cases)
- **Strictness** — *Aggressive* catches mild softness for reclaiming storage, *Conservative* only flags clearly unusable frames
- Blurry frames are **marked rejected, not deleted**, so they land in the same review drawer and can be put back
- **Sharpen Recoverable** is offered only for frames with detail left to lift. Past that point sharpening adds halos rather than restoring detail — the Laplacian score rises either way, which is exactly why the score alone does not decide it

### Auto-levelling
- **Auto-Level Selected** straightens a whole batch in one pass. Two engines run, because neither covers every photograph
- *Which way up* comes from face detection — a detector will not fire on an upside-down face, so each quarter turn is tried. All 90/180/270 turns were recovered correctly
- When no face is visible (a subject facing away, a face too small), the **body** answers instead: the direction from the hips to the shoulders is read straight off the pose landmarks and rounded to a quarter turn. With the face engine switched off entirely, all four turns were still recovered
- *How far off level* comes from the straight lines in the scene. Round-tripping known tilts leaves **under 1° residual**
- When the frame has no usable lines — a portrait against a plain wall — the **shoulder and hip lines** carry it instead. Over a ±14° sweep that left **0.6° mean error, 1.7° worst**
- Lines are preferred over the body wherever there are any: lines measure the camera against the world, whereas a person can genuinely be leaning. In practice the two rarely compete — the test portraits yielded no usable lines at all, and the frames with strong lines had no usable body
- **Use body** is optional and needs a one-off ~17 MB model download, which the app asks about before fetching. Everything is verified against a pinned fingerprint and nothing is uploaded
- **Check Posture** reports what the subjects are doing — how much of the body is in frame, whether the camera was rolled, whether the subject is leaning — and changes nothing
- Rotating leaves wedge corners; **Crop corners** cuts back to the largest clean rectangle, or leave it off to keep them black. There is no generative fill
- **Head roll is still deliberately not used.** See the caveat below

### Reviewing and rejecting
- **Select All / None / Invert**, scoped to whatever the view filter is showing — selecting frames you cannot see would be a trap
- **Reject Selected** marks photos as out. **Rejected never means deleted:** the original file is never moved, renamed or removed. It simply stops being copied into the `-edit` subfolder
- **Review Rejected** opens every rejected frame as thumbnails before anything is acted on, with a tick per photo so you can put any of them back
- **Reject Non-Picks** rejects everything a cull did not choose, in one click, reversibly
- **Show:** filters the grid to *Keepers*, *AI picks*, *Duplicates* or *Rejected*, so you can actually see what the cull decided instead of reading it in a report

### Library & Culling
- **Universal loader**: One-click folder import. Embedded RAW thumbnails cached for instant preview scrolling
- **Smart cull**: Laplacian-variance sharpness scoring + timestamp-based burst grouping. Auto-stars the top 2 sharpest of each burst
- **Blink detection**: YuNet (OpenCV's DNN face detector) finds faces, then each eye is scored for openness. With **Skip blinks** enabled, an open-eyed frame outranks a *sharper* one where everybody blinked — a crisp photo of closed eyes is not a keeper. Degrades to sharpness-only if the model can't be fetched
- **Cull to target**: Say how many keepers you want ("the best 400 of these 3,000") instead of guessing at a sharpness threshold. The quota is spread across bursts, so a sharp stretch of the day cannot swallow the whole selection and leave other moments unrepresented
- **XMP sidecar export**: Writes star ratings and colour labels to `.xmp` files beside each image, so Lightroom Classic, Capture One, Photo Mechanic and Bridge open the folder with your picks already applied. Existing sidecars are *merged*, never overwritten — another tool's develop settings, keywords and crops survive. Originals are never modified
- **Metadata overlays**: Camera, lens, ISO, aperture, shutter speed displayed on hover

### Color & Grading
- **Selective color**: Isolate one hue (e.g., "red dress") and desaturate everything else
- **Cinematic filters**: 6 matrix-based grades (teal-orange, bleach bypass, noir, etc.) with lift/gamma tone curves
- **Smart auto-fix**: One-click white-balance + CLAHE histogram equalization for web readiness

### Web Workflow
- **Edit-safe subfolder**: Load `Vacation2024` → all edits save to `Vacation2024-edit` → originals stay pristine
- **Selective publishing**: Check "Publish to Web" on individual thumbnails. Only checked images export
- **Labeled galleries**: Assign categories (Graduation, Nature, Travel, etc.) to thumbnails, rendered as category tabs in the published HTML
- **Gallery naming**: Custom title for each batch you export
- **Keeper JPEG export**: RAW files are automatically flattened to JPEG in the edit folder on first export (idempotent, so repeated calls are instant)

### Instagram & Social
- **Panoramic reels**: 3–7 images stitched into a massive panorama, with a 1080×1920 viewport sliding left-to-right over 10 seconds (MP4)
- **IG formatting**: Auto-pad any image to exact 4:5 (feed) or 9:16 (story) with a blurred, darkened background

### Background Removal
- **AI segmentation**: One-click background removal via `rembg` (U2Net ONNX model), CPU-friendly and GPU-independent
- **Transparent PNG export**: Outputs `*_nobg.png` (RGBA) to the edit folder, ready for compositing

### Video Editor
A full FFmpeg + OpenCV video engine, ships its own FFmpeg binary (`imageio-ffmpeg`) so it works with no system install:
- **Timing**: trim, reverse, freeze-frame, constant/ramped speed changes, time-lapse condensation (e.g. a 30-minute file → a 30-second highlight)
- **Framing**: aspect-ratio conversion (16:9/9:16/1:1/4:5/4:3, crop or letterbox), explicit bounding-box crop, subject-aware auto-crop (Haar-cascade face detection with a center-crop fallback), OpenCV point-feature stabilization, static-region redaction
- **Structure**: split/trim, multi-clip concat, crossfade/wipe transitions (`xfade`/`acrossfade`), J-cut/L-cut audio offsetting
- **Audio**: track mixing, EBU R128 loudness normalization, sidechain auto-ducking, FFT noise reduction (`afftdn`), volume adjustment
- **Color**: brightness/contrast/saturation/gamma correction, temperature-based white balance, 3D LUT (`.cube`) application
- **Export presets**: Web MP4 (H.264/faststart), high-quality MOV, WebM (VP9/Opus), Instagram Reel (1080×1920)
- **Universal ingestion**: MP4, MOV, AVI, MKV, WebM, FLV — any container/codec FFmpeg can decode
- **Local REST API**: optional FastAPI server (`src/video/api.py`, toggled from the Video Editor tab) exposes `/probe`, `/jobs`, `/concat`, `/transition` so external scripts can drive the engine over HTTP
- **Hardware-sized job pool**: multi-step pipelines run through a background thread pool sized to leave CPU headroom, matching the same conservative philosophy as the photo side's RAM budgeting

## Hardware Support

Granite Studios detects your hardware and picks the best engine:
- **Nvidia GPU** (CUDA): Full acceleration for RAW demosaicing and ML inference
- **Apple Silicon** (M1/M2/M3): CoreML/MPS for native performance
- **Intel iGPU** (integrated graphics): OpenVINO inference
- **CPU fallback**: Works everywhere, just slower

RAM is automatically profiled at startup. Large operations (batch exports) are sized conservatively to never starve the OS.

## Project Structure

```
.
├── main.py                    # Entry point: splash screen, hardware diagnostics, shortcut creation
├── requirements.txt           # Python dependencies
├── src/
│   ├── engine/
│   │   ├── hardware.py       # GPU/CPU detection, RAM profiling
│   │   ├── universal_loader.py # RAW/HEIC/JPEG loader, edit-subfolder management
│   │   ├── resources.py      # Frozen-bundle path resolution (sys._MEIPASS)
│   │   └── create_shortcut.py # First-run Desktop/Start Menu shortcuts
│   ├── ai_tools/
│   │   ├── culling.py        # Sharpness scoring, burst grouping, perceptual hashing
│   │   ├── color_engine.py   # Selective color, cinematic filters, auto-edit
│   │   └── background_removal.py # AI background removal (rembg / U2Net)
│   ├── social/
│   │   ├── reel_maker.py     # Panoramic video stitcher (moviepy)
│   │   └── ig_sync.py        # Instagram feed/story formatter
│   ├── web/
│   │   └── publisher.py      # Static Tailwind gallery generator
│   ├── video/                # FFmpeg + OpenCV video editing engine
│   │   ├── ffmpeg_utils.py   # Bundled-ffmpeg resolution, probing, subprocess wrapper
│   │   ├── normalize.py      # fps/sample-rate/resolution normalization pass
│   │   ├── timing.py         # Trim, reverse, freeze-frame, speed ramp, time-lapse
│   │   ├── spatial.py        # Aspect conversion, crop, auto-crop, stabilization
│   │   ├── structure.py      # Split, concat, transitions, J/L cuts
│   │   ├── audio.py          # Mixing, loudness norm, auto-duck, denoise
│   │   ├── color.py          # Color correction, 3D LUT application
│   │   ├── export.py         # Container/codec/bitrate export presets
│   │   ├── pipeline.py       # VideoJob/JobManager — multi-step async pipelines
│   │   └── api.py            # Optional local FastAPI REST server
│   └── ui/
│       └── main_window.py    # PyQt6 dark-theme professional interface
├── web/                      # Browser version (see web/README.md)
│   ├── backend/              # FastAPI service wrapping src/ engines (Docker → Render)
│   └── frontend/             # React + Vite + Tailwind UI (→ Vercel)
├── render.yaml               # Render blueprint for the web backend
├── package_windows.py        # PyInstaller builder for .exe (Pillow icon conversion)
└── .github/workflows/build.yml # CI: macOS .dmg builder
```

## Usage Workflow

> **One pane, not three.** Import, culling, rejection and grading all happen
> on the **Photos** page against a single grid that stays on screen. They used
> to be three separate tabs, which hid the photographs at exactly the moment
> you needed to see what an action had done.

### 1. Load & Cull
- Click **Load Folder** on the Photos page
- Edit subfolder `<FolderName>-edit` is created automatically
- Run **Smart Cull** to score sharpness and group similar frames → stars the best of each burst

### 2. Grade & Edit
- Select images, choose a **Cinematic Filter** or **Selective Color** mode
- Click **Apply to Selected** → graded `*_graded.jpg` files land in the edit folder
- Or use **Auto-Edit** on export for instant white-balance + contrast

### 3. Publish to Web
- Check "**Publish to Web**" on the thumbnails you want to include
- Assign **Labels** (Graduation, Nature, etc.) for category tabs
- Enter a **Gallery Name** (e.g., "Vacation 2024 Highlights")
- Click **Generate Gallery** → responsive HTML + images ready to upload to any web host

### 4. Export Socials
- **Reel Maker**: Select 3–7 images → choose where to save → outputs a 1080×1920 MP4 panoramic scroll (selecting fewer than 3 or more than 7 is rejected)
- **IG Formatter**: Select images → choose an export folder → each is padded to exactly 1080×1350 (feed) or 1080×1920 (story) over a blurred, darkened copy of itself

### 5. Edit Video
- Click **Load Video** in the Video Editor tab (any MP4/MOV/AVI/MKV/WebM/FLV file)
- Run operations in any order — trim, speed, reframe, stabilize, color-correct, denoise. Each one saves into `<video-name>-edit/` and becomes the input to the next, so they chain; **Reset to Original** starts over
- Pick an **Export preset** and click **Export** — Web MP4 (H.264, faststart, capped at 1920 wide), MOV (high-quality H.264), WebM (VP9/Opus), or Instagram Reel, which fits and pads the clip to exactly 1080×1920 whatever the source shape
- Optionally click **Start Local API Server** to drive the same engine from an external script over HTTP (`http://127.0.0.1:8756/docs`)

## Development

### Building from Source
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pyinstaller

# Windows
python package_windows.py
# Output: dist/GraniteStudios/GraniteStudios.exe

# macOS (from a Mac)
pyinstaller main.py --name "Granite Studios" --windowed --onedir \
  --add-data "src:src" --icon=icon.icns --hidden-import=rawpy
# Output: dist/Granite Studios.app
```

### Testing
There is no automated test suite yet. Verification is done by exercising the
real engines against generated media, and by driving the PyQt6 window itself
with `QT_QPA_PLATFORM=offscreen` so every button's handler runs headlessly.

What that covers today:

| Area | Verified |
| --- | --- |
| `hardware.py` | RAM profiling; export budget = half available, capped at 8GB |
| `universal_loader.py` | Folder scan, RAW/HEIC/JPEG/PNG support, preview + array decode, EXIF, `-edit` folder creation |
| `culling.py` | Sharpness ranks sharp above blurred; perceptual hash flags near-duplicates; burst grouping |
| `color_engine.py` | All 6 cinematic filters; all 8 selective-colour hues (isolated hue stays saturated, rest goes mono); auto-fix; brightness/contrast/saturation |
| `background_removal.py` | Returns RGBA at source size with a non-uniform alpha channel |
| `ig_sync.py` | Exact 1080×1350 feed and 1080×1920 story output |
| `reel_maker.py` | 1080×1920 MP4 at the requested duration; rejects fewer than 3 or more than 7 images |
| `publisher.py` | Category tabs, custom title, responsive masonry, images emitted |
| `xmp_sidecar.py` | Ratings 0-5 and -1 round-trip; all five Lightroom labels; clamping; **merge preserves another tool's `crs:` develop settings, `dc:title` and keywords**; a corrupt sidecar degrades instead of raising; batch write over a real cull |
| `cull_to_target` | Quota spread evenly across bursts, picks each burst's sharpest first; clamps above the image count; rejects a negative target |
| `create_shortcut.py` | Writes a `.lnk` whose argument is quoted, targets `pythonw.exe`, and launches successfully from a path containing a space |
| `blur_analysis.py` | Sharp frames not flagged across 3 reference images; defocus classified correctly at 2 blur radii each; **15/15 motion cases detected with direction within 15°**; strictness levels verified to change the verdict on the same frame; the rescue gate refuses frames past recovery |
| `orientation.py` | Upright frames left alone; **all 90/180/270 turns recovered** (and still recovered with the face engine disabled, using the body); known tilts round-tripped to under 1° residual; blank padding no longer read as scene structure; crop leaves no black corners; batch skips already-level frames |
| `pose_analysis.py` | Landmarks verified against the image (skeleton lands on the body); **body tilt tracks a known roll 1:1 across ±24°**; levelling a full-body frame leaves **0.53° mean / 1.33° worst** residual; shoulders-only leaves 2.90° / 9.54° and is therefore never applied; a half-out-of-frame body is refused rather than guessed; deterministic across repeat runs; degrades to lines-only with the models absent |
| `model_store.py` | Pinned digests reject a substituted or truncated download; a cached model that fails verification is refetched; a missing model disables its feature instead of raising |
| `face_analysis.py` | Model download checksum-pinned and verified; detects two real faces and **no false positives** across six non-face sample images; stable from 0.4x to 2x scale; grayscale, RGBA, empty and `None` inputs; landmarks mapped back to full-resolution coordinates after downscaling; **degrades to unavailable rather than raising when the model is absent** |
| blink-aware culling | With eye checking off, the sharpest frame wins even with closed eyes; with it on, the same burst picks the softer open-eyed frame instead — verified on bursts built so sharpness and open eyes disagree |
| `video/` | Every timing, spatial, structure, audio, and colour operation; all 4 export presets; `.cube` LUT; multi-step `JobManager` pipelines including the failure path |
| `video/api.py` | `/probe`, `/jobs`, `/jobs/{id}`, `/concat`, `/transition`, plus 400/404 handling |
| `ui/main_window.py` | Window construction, folder load, video load, all ten video operations chained through their real handlers, plus **Select Top N**, **Skip blinks** and **Write XMP Sidecars** driven through their real handlers |

Contributions that add a proper `pytest` suite are very welcome — see
[Contributing](#contributing).

### Key Dependencies
- **PyQt6**: Desktop UI (dark theme, fade transitions, hardware info display)
- **rawpy**: RAW demosaicing + embedded thumbnail extraction
- **Pillow / pillow-heif**: JPEG, PNG, HEIC support
- **OpenCV**: Sharpness analysis, selective color masking, video stabilization/auto-crop
- **numpy**: Image arrays, matrix math for color grading
- **moviepy**: Reel rendering
- **rembg**: AI background removal (U2Net ONNX model)
- **imageio-ffmpeg**: Bundled FFmpeg binary for the video engine — no system install required
- **fastapi / uvicorn**: Optional local REST API for the video engine
- **psutil**: RAM and CPU core detection

## Architecture Highlights

### Edit-Safe Workflow
Loading a folder creates an `-edit` folder *inside* it, named after the folder,
and edits land there instead of next to your originals:

```
Vacation2024/                 # your originals — never modified or moved
├── DSC_0001.NEF
├── DSC_0002.NEF
└── Vacation2024-edit/        # created on load
    ├── DSC_0001_graded.jpg   # cinematic filter / selective colour / auto-fix
    ├── DSC_0002_nobg.png     # background removal
    └── DSC_0001.jpg          # keeper flatten of the RAW
```

The video editor follows the same rule per file — loading `clip.mp4` creates
`clip-edit/` beside it, and each operation writes a new file there rather than
overwriting the last, which is what lets operations chain and **Reset to
Original** work.

The three *publishing* actions — panoramic reel, Instagram export, and
**Generate Gallery** — ask you where to save instead, since those outputs
usually belong somewhere else entirely. The gallery still reads its images
exclusively from the edit folder.

### Hardware Adaptation
At startup, the app logs detected CPU cores, RAM, ONNX execution provider, and calculates a safe export budget (half of available RAM, capped at 8GB). This ensures batch operations never OOM or freeze the OS.

### Splash Screen & Diagnostics
- Shows app icon centered on launch
- Heavy imports (rawpy, cv2, onnxruntime) happen while the splash is visible, so the delay is hidden
- Hardware diagnostics logged to stdout (great for debugging on end-user machines)

### Thread Safety
- Thumbnail previews and metadata decode off-thread → UI never freezes during folder load
- Long-running tasks (cull, color, export) run in background workers, with a dialog if you try to start two at once
- LRU cache (thread-safe) bounds RAM usage during grid scrolling

## Packaging

### Windows Distribution
- **PyInstaller** bundle (--onedir for fast startup)
- Icon conversion: `icon.png` → `icon.ico` (7 sizes)
- First-run creates Desktop + Start Menu shortcut via **pyshortcuts**
- Taskbar uses custom AppUserModelID so your icon appears instead of the generic Python logo

### macOS Distribution
- **PyInstaller** app bundle
- Icon conversion: `icon.png` → `icon.icns`
- Packaged into a `.dmg` with drag-to-Applications instructions
- Distributed via GitHub Releases

## License

[See LICENSE file](LICENSE)

## Support

### First-Run Issues
See [README_INSTALL.md](README_INSTALL.md) for Windows SmartScreen and macOS Gatekeeper bypass steps.

### Hardware Requirements
- **Minimum**: 8GB RAM, dual-core CPU (i5-8350U integrated GPU supported)
- **Recommended**: 32GB RAM, 6+ cores, dedicated GPU (Nvidia/AMD or Apple Silicon)
- **Storage**: ~2GB for the app + space for your image library

### Performance Tips
- Preview caching is LRU-bounded (256 images in RAM by default) — scroll smoothly even on integrated graphics
- Smart Cull uses downscaled previews for scoring, so batch analysis is fast
- Edit-folder workflow means no re-reading originals for each export
- Web publisher resizes once per batch, so large exports finish in seconds

## Roadmap

The full specification — requirements with acceptance criteria, non-functional requirements, architecture, and the target state with each item's actual blocker — lives in **[PRD.md](PRD.md)**. The tables below are the short version.

Granite Studios is being measured against a reference spec for a
high-volume photographer workflow (AI culling, AI editing, AI retouching,
client galleries). That spec describes a mature commercial product; this
table is an honest account of where the app actually stands against it, so
nothing here reads as a promise the code does not keep.

### Culling

| Requirement | Status |
| --- | --- |
| Native RAW (CR3/NEF/ARW/DNG), JPEG, TIFF with no pre-conversion | **Done** |
| Focus / blur quality scoring | **Done** — Laplacian variance |
| Duplicate & burst sequence detection | **Done** — perceptual hash + capture-time grouping |
| Cull to a target count | **Done** — quota spread across bursts |
| Star ratings & colour labels to `.xmp` sidecars | **Done** — merges, never clobbers |
| Closed-eye / blink detection | **Done** — YuNet detection + per-eye openness scoring, wired into burst ranking. Heuristic, not a trained classifier (see below) |
| Identifying the same person across a shoot | **Not built.** Face *detection* now exists, but telling one person from another needs recognition embeddings (e.g. SFace) on top |
| Loupe / Survey / spray-can review UI | **Not built.** Review today is the Library grid plus the cull report |

### Editing & retouching

| Requirement | Status |
| --- | --- |
| Manual RAW adjustment sliders | **Done** — exposure/contrast/saturation, white balance, 6 film grades |
| Subject / background masking | **Partial** — background removal produces a subject mask; it is not yet exposed as an editable mask |
| Skeletal posture / body-aware levelling | **Done** — BlazePose (optional ~17 MB download) reports shoulders, spine and hips; levels full-body frames to 0.5° and gets back-turned subjects the right way up |
| Background replacement | **Partial** — cutout exists, compositing a new background does not |
| Skin smoothing, blemish & wrinkle reduction | **Not built.** Face detection and pose both exist now, so classical frequency-separation retouching targeted at the face box is the next tractable step here |
| Teeth whitening, glare removal, stray-hair removal, clothing de-wrinkling | **Not built.** Each needs its own trained segmentation model |
| AI profiles trained from a photographer's Lightroom catalogue | **Not built.** This is a training pipeline and a model architecture, not a feature flag — the single largest item in the spec |
| Marketplace of pre-built styles | **Not built.** Needs server infrastructure and curated content |

### Client delivery

| Requirement | Status |
| --- | --- |
| Static web gallery with category tabs | **Done** — self-hosted HTML output |
| Client proofing, favourites, face search, print ordering, 4K downloads | **Not built.** These need a hosted backend, accounts and payments; the desktop app only generates static galleries |

### Other near-term items

- [ ] ONNX subject detection for selective masks (auto-star hero shots)
- [ ] EXIF-driven smart grouping (camera + lens combo)
- [ ] Batch watermarking
- [ ] Direct Instagram upload (currently exports, you upload)
- [ ] Keyboard shortcuts for rapid culling

### Why auto-levelling ignores head roll

The obvious approach — level the line between the detected eyes — does not
work with the bundled face model, and the measurements are recorded here so
the idea is not retried blind:

- Rotating a portrait through **24°** moved the measured eye line by about
  **5°**. YuNet regresses its landmarks toward an upright configuration, so
  the eye line is close to horizontal whatever the head is doing.
- Searching for the rotation of peak detection confidence fails for the same
  underlying reason: the detector is deliberately tolerant of in-plane
  rotation, so confidence stayed **flat across the entire ±20° sweep**.

The pose model behaves the same way when swept — its confidence sat at
**1.00 at all four quarter turns** — which is exactly why the body signal
reads the landmark *geometry* instead of hunting for a confidence peak. A
landmark moves with the body; a confidence score does not.

### Reading posture: what the body engine can and cannot do

Pose estimation is now bundled (optional download), so shoulders, spine and
hips are available. What that does and does not buy:

**Camera roll versus a leaning subject.** A tilted shoulder line on its own
is ambiguous — it happens when the camera was rolled and when the subject
shifted their weight. The hips settle it: camera roll turns the whole body
together, a lean moves the shoulders against the hips. A correction is only
offered when the two agree.

The agreement threshold is measured, not chosen by taste. A subject standing
square still reads a shoulder/hip spread of **4–7°** frame to frame, because
the two lines are regressed independently — an initial 6° gate refused
ordinary upright photographs, so it is 12°. A genuine lean is far wider: the
worst case in testing, a body half out of frame, spread **57°** and was
correctly refused.

**Shoulders without hips are reported but never applied.** Correcting a
frame by the body angle and re-measuring what is left over:

| Visible | Mean residual | Worst |
|---|---|---|
| Shoulders **and** hips | **0.53°** | 1.33° |
| Shoulders only | 2.90° | **9.54°** |

A ten-degree error is worse than the tilt most people would have left alone,
so a head-and-shoulders crop gets a reading and no automatic correction.

**Spine lean is reported, never corrected.** How far the shoulder-to-hip
axis departs from vertical is a property of the subject, not the camera.
Straightening it would be editing the pose, not levelling the photograph.

**Still not built:** generative corner fill, which needs an inpainting model.
Rotating leaves wedge-shaped gaps, and the two honest options are to crop
them away or leave them black — both of which the app offers.

**A note on what gets measured.** `line_tilt` previously read the black
wedges left by an earlier rotation as scene structure: their borders are
flawless straight lines sitting at precisely the angle that produced them, so
a corrected frame looked strongly tilted. Blank regions are now masked out
before edge detection. This surfaced only because a test fed it a padded
frame — any already-rotated, letterboxed or scanned photograph would have
hit it.

### How blink detection actually works, and what it is not

YuNet returns five landmarks per face, two of which are eye *centres*. It
gives no eye contour, so the standard Eye Aspect Ratio — which needs six
points per eye — cannot be computed from it. Instead a patch is cropped
around each eye centre and the **vertical extent of the dark region** inside
it is measured: an open eye shows the iris as a tall dark blob, a closed lid
shows only a thin horizontal lash line.

This is a **heuristic, not a trained classifier.** Be aware of what that
means in practice:

- The decision threshold (`EYE_OPEN_THRESHOLD`, 0.42) was placed using two
  real faces with open eyes and synthetically closed versions of them. Open
  eyes measured 0.52-0.78 and closed 0.15-0.23, so the gap is wide — but
  two faces is **not** a validated dataset.
- Sunglasses, heavy shadow, strong side-lighting, motion blur across the eye,
  and faces turned far from camera have not been tested at all.
- `eyes_closed` deliberately requires **both** eyes below threshold. One eye
  alone is far more often a highlight, a stray hair or a head turn than a
  real blink, and flagging those would make a cull noisy.
- Every frame also exposes a raw `min_eye_openness` score, so you can judge
  borderline frames yourself rather than trusting the boolean.

Treat it as a strong first pass worth reviewing, not an oracle. If it
mis-reads your images, `EYE_OPEN_THRESHOLD` is the one number to tune.

### A note on the performance target

The spec asks for 1,000 RAW files culled in under five minutes. Measured so
far, on this hardware:

| Workload | Per frame | Extrapolated to 1,000 |
| --- | --- | --- |
| Blink detection, 512px JPEG | ~19 ms | ~0.3 min |
| Blink detection, 36MP JPEG | ~341 ms | **~5.7 min** |

So with blink detection enabled on full-resolution files the target is
**not** met — and these numbers are from JPEGs, whereas RAW decoding is
slower still. The cost is dominated by decoding the file, not by the
detector. Culling without `check_eyes` is far quicker. This has still never
been benchmarked on genuine camera RAW files, so treat the RAW case as
unmeasured.

## Contributing

Bug reports, feature suggestions, and PRs welcome. Please open an issue first so we can discuss.

---

Built with ❤️ by photographers, for photographers.  
**Granite Studios** — your RAW, edited locally. Never in the cloud.
