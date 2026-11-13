# Granite Studios

### [⬇️ Download the installer](https://github.com/marufmhoque/GraniteStudios/releases/latest/download/Install-GraniteStudios.bat)

One file. Download it, double-click it, done — it fetches the latest version
and installs it for you. Re-run the same file any time to update.
[![Publish installer](https://github.com/marufmhoque/GraniteStudios/actions/workflows/installer.yml/badge.svg)](https://github.com/marufmhoque/GraniteStudios/actions/workflows/installer.yml)

> **📋 [Product Requirements Document (PRD.md)](PRD.md)** — who it is for, the invariants it must never break (originals are never modified or deleted), every requirement with its measured evidence, known limitations, and the roadmap.

---

**Local-first AI photo and video editing suite** for photographers and content creators. Designed to run smoothly on modest hardware (i5-8350U + 32GB RAM), but scales to powerful machines with CUDA or Apple Silicon.

Six workspaces over one folder: **Cull**, **Develop**, **Retouch**, **Design**, **Export** and **Video**.

- 📸 **Universal input**: ten camera RAW formats, phone DNG, HEIC (including 10-bit), Ultra HDR, Motion Photos, Live Photos, JPEG, PNG, TIFF, and ProRes/DNxHD video
- ⚡ **Fast culling**: 1,000 camera RAWs scored for focus, duplicates, blinks and smiles in **42–104 seconds** on an i5-8350U laptop ([measured](#a-note-on-the-performance-target))
- 👁️ **Faces**: a 478-point face mesh measures each eye's aspect ratio and each smile; faces are grouped by person, entirely on your computer
- 🎚️ **Develop**: a 16-bit, float develop engine with tone, curves, HSL, colour grading, dehaze, lens profiles, LUTs, straighten and keystone
- 🖌️ **Retouch**: layers with seven blend modes, AI masks (subject, skin, teeth, eyes, sky…), healing, cloning, skin smoothing that keeps pores, and inpainting that fills straightened corners
- ✍️ **Design**: vector text, shapes and watermarks that print sharp at any size and export to SVG and PDF
- 💾 **Non-destructive**: every edit is a small settings file. Nothing is rendered until you export, and your originals are never modified, moved or deleted

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

### The six workspaces

The bar across the top switches between them (**Ctrl+1** to **Ctrl+6**). They
all look at the same open folder: a selection, rating or rejection made in one
is seen in all of them, and the editing workspaces show the folder as a
filmstrip along the bottom.

| Workspace | What it is for |
| --- | --- |
| **Cull** | The grid. AI cull, blur triage, auto-levelling, ratings and labels, rejecting, survey and face loupe, grouping by person |
| **Develop** | One frame, every tone and colour setting, live preview, before and after |
| **Retouch** | Layers and masks, healing and cloning, skin, teeth and eyes |
| **Design** | Text, shapes and watermarks over the finished frame |
| **Export** | Everything that writes files: presets, watermarks, Instagram, reels, web gallery, background removal |
| **Video** | The FFmpeg video editor |

### Cull

- **Keyboard first.** Arrows move; **1–5** set stars (0 clears); **6–9** set red, yellow, green, blue labels; **P** picks, **X** rejects, **U** clears both; **Enter** opens the frame in Develop. Marks apply to the selection, or to the current frame when nothing is selected, and are saved in the edit folder so they are there next time
- **Smart cull** decodes each file once. For a RAW that is the camera's own embedded JPEG, not a full development of the sensor data; sharpness, a duplicate fingerprint, the capture time and every face are all measured from that one decode, on all your cores
- **The keeper score** ranks frames inside each burst: sharpness *of the faces* (40%), eyes open (35%), smile (15%) and a simple composition check (10%). A frame focused on the wall behind someone no longer wins. Frames with no people are ranked on overall sharpness
- **Cull to target**: say how many keepers you want ("the best 400 of these 3,000"); the quota is spread across bursts so no part of the day is dropped wholesale
- **Survey (S)** puts 2–6 selected frames side by side with zoom and pan locked together. **Face Loupe (F)** shows every face in the selection at full size, to check focus and eyes
- **Group by Person** recognises faces across the folder and groups photographs by who is in them, so you can show only one person's frames and name them ("Bride"). It all happens on your computer; **Forget Faces** deletes the stored face data
- **XMP sidecars** carry ratings and labels to Lightroom Classic, Capture One, Photo Mechanic and Bridge, merging with whatever those tools already wrote

### Blur triage
- **Find Blurry** separates *motion blur* from *missed focus*: a smear destroys detail along one axis, defocus softens every direction equally. Motion direction is reported too (±15° across 15 synthetic test cases)
- **Strictness** — *Aggressive* catches mild softness, *Conservative* only flags clearly unusable frames
- Blurry frames are **marked rejected, not deleted**, so they land in the same review drawer and can be put back
- **Sharpen Recoverable** is offered only for frames with detail left to lift, and adds sharpening to their edits rather than writing new files

### Auto-levelling
- **Auto-Level Selected** straightens a batch by setting each frame's rotation. Two engines run, because neither covers every photograph
- *Which way up* comes from face detection — a detector will not fire on an upside-down face, so each quarter turn is tried — and from the **body** when no face is visible (hips-to-shoulders direction). All 90/180/270 turns were recovered, with or without the face engine
- *How far off level* comes from the straight lines in the scene (**under 1° residual** on known tilts), or from the **shoulder and hip lines** when the frame has none (**0.6° mean, 1.7° worst** over ±14°)
- **Corners**: *Crop* cuts back to the largest clean rectangle; *Fill* paints the corners in at export so the full frame is kept; *Leave black* does neither
- **Check Posture** reports how much of each body is in frame, whether the camera was rolled and whether the subject is leaning, and changes nothing
- **Head roll is still deliberately not used to level.** See the caveat below

### Reviewing and rejecting
- **Select All / None / Invert**, scoped to whatever the view filter is showing
- **Rejected never means deleted.** The original file is never moved, renamed or removed; it is simply left out of every export
- **Review Rejected** shows every rejected frame before anything is acted on, with a tick per photo to put it back
- **Show:** filters the grid to *Keepers*, *AI picks*, *Duplicates*, *Rejected*, *Blurry*, *Motion blur* or *Out of focus*, and to one person after **Group by Person**

### Develop
- **Basic**: exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, texture, clarity, dehaze, vibrance, saturation
- **Tone curve** for the whole image and per channel, drawn over the histogram
- **Colour**: eight-band HSL (hue, saturation, luminance), three-way colour grading (shadows, midtones, highlights) with balance and blending
- **Looks and LUTs**: the six original grades as adjustable looks, and any 3D `.cube` LUT, each with a strength slider
- **Detail**: sharpening and noise reduction. **Lens**: automatic distortion, colour-fringe and vignetting correction from the lens database when the camera and lens are known, manual sliders when not
- **Crop and straighten**: quarter turns, flip, straighten, vertical and horizontal keystone, aspect crops
- **Before / After (\\)**, **clipping view (J)** showing lost shadows in blue and blown highlights in red, a live RGB histogram, **undo and redo** (Ctrl+Z / Ctrl+Y), **copy and paste settings**, and **Sync** to copy the settings to every selected frame (crops and retouching stay with their own photo)
- RAW files develop from 16-bit linear sensor data in 32-bit float. Highlights roll off rather than clip when exposure is pushed

### Retouch
- **Layers** with opacity and seven blend modes (normal, multiply, screen, overlay, soft light, colour dodge, luminosity), shown in a panel you can reorder and hide
- **Adjustment layers** limited by a mask: **AI** (subject, background, person, skin, teeth, eyes, iris, lips; hair and sky are approximate and labelled so), **click-to-select** an object, a **gradient**, or one you **paint** with the brush
- **Heal** removes blemishes and dust with a click; **Clone** copies a patch from one place to another. Both are stored as positions, so they render at full resolution on export
- **Smooth Skin** evens out blotches while keeping pores (frequency separation, measured on the texture bands); **Whiten Teeth** takes out the yellow without making them glow; **Brighten Eyes** adds clarity to the irises

### Design
- **Text** in any installed font with size, bold, italic, kerning, tracking, leading, alignment, outline and drop shadow; **rectangles** (with rounded corners), **ellipses**, **stars** and **bezier paths**
- **Watermarks**: a signature, a round photographer's seal, and a strip that fills in each photo's own camera, lens and exposure from `{camera}`, `{lens}`, `{settings}`, `{date}` tokens
- **Snap** to a grid, with guide lines when an item lines up with another or with the middle of the frame
- **Export SVG** (the design alone, as vector) or **Export PDF** (the photograph with the design, as a print page)

### Export
- **Presets**: Web JPEG (2048 px), full-size JPEG, 16-bit TIFF, print at 300 dpi, 16-bit PNG — each rendering every selected frame with its edits, layers and overlays
- **Batch watermark** with any of the three templates, filled in per photograph
- **Instagram** 4:5 feed and 9:16 story, the **panoramic reel**, the **static web gallery** with category tabs, and **background removal** to transparent PNG all use the edited version of each frame
- Rejected frames are always left out

### Phones and video
- **HEIC** at its full bit depth; **Ultra HDR** gain maps read and applied; **Motion Photo** clips found and extracted; **Live Photo** stills paired with their clips; ProRAW DNGs are recognised (their tone-map table is not yet applied — see the PRD)
- The **Video** workspace is the full FFmpeg editor: trim, speed, reverse, time-lapse, reframe, stabilise, crop, redact, concat, transitions, J/L cuts, loudness, ducking, denoise, colour and LUTs, four export presets and an optional local REST API. Ingests MP4, MOV, MKV, WebM, AVI and ProRes / DNxHD, and ships its own FFmpeg

## Hardware Support

AI models that run through ONNX Runtime use the best accelerator the machine has:

- **Windows**: DirectML, on any DirectX 12 graphics card (Intel, AMD or NVIDIA), no vendor install needed
- **macOS**: Core ML (Apple Neural Engine)
- **Elsewhere**: CUDA or OpenVINO when present
- **CPU** always works, and is used automatically if the accelerator will not start

Measured on Intel UHD 620 integrated graphics: background removal **1.36× faster** on DirectML, with output matching the CPU to within one level in 255. The very first run on a machine takes about a minute while the graphics driver prepares; after that it starts in under a second. The face, pose, mesh, segmentation and inpainting models run through OpenCV on the CPU and are unaffected.

RAM is profiled at startup and batch work is sized so it never starves the system.

## Project Structure

```
.
├── main.py                       # Entry point: splash screen, diagnostics, shortcuts
├── src/
│   ├── engine/
│   │   ├── fast_preview.py       # Embedded RAW previews, reduced and greyscale decodes
│   │   ├── universal_loader.py   # Formats, EXIF, edit folders
│   │   ├── phone_media.py        # Ultra HDR, Motion/Live Photos, deep HEIC
│   │   ├── parametric_recipe.py  # The edit recipe and its sidecar files
│   │   ├── develop_pipeline.py   # Float develop engine, geometry, lens, LUTs, export
│   │   ├── layer_stack.py        # Layers, masks, blend modes, heal and clone
│   │   ├── export_presets.py     # Export presets and batch rendering
│   │   ├── session.py            # The open folder, shared by every workspace
│   │   └── hardware.py           # Accelerator selection, RAM budgets
│   ├── ai_tools/
│   │   ├── culling.py            # One-pass scoring, keeper score, bursts, target cull
│   │   ├── face_analysis.py      # YuNet face detection
│   │   ├── face_mesh.py          # 478-point mesh: eye aspect ratio, smile, head roll
│   │   ├── person_clustering.py  # Face recognition and grouping by person
│   │   ├── segmentation.py       # Subject, skin, teeth, eyes, sky, click masks
│   │   ├── retouch.py            # Skin, teeth, eyes, healing, inpainting
│   │   ├── blur_analysis.py, orientation.py, pose_analysis.py
│   │   ├── color_engine.py, background_removal.py, xmp_sidecar.py
│   │   └── model_store.py        # Pinned, verified model downloads
│   ├── graphics/vector_canvas.py # Text, shapes, watermarks; raster, SVG, PDF
│   ├── ui/
│   │   ├── main_window.py        # The workspace bar
│   │   ├── workspaces/           # cull, develop, retouch, design, export, video
│   │   └── widgets/              # canvas, histogram, curves, filmstrip, renderer
│   ├── social/, web/, video/     # Reels and Instagram, web gallery, video engine
├── tests/                        # pytest suite (see Testing)
├── web/                          # Browser version (see web/README.md)
└── install/, package_windows.py  # Installer and packaging
```

## Usage Workflow

### 1. Cull
- **Load Folder** (top right). A `<FolderName>-edit` subfolder is created for everything the app writes
- **Select Top N** with *Skip blinks* to pick keepers, or **Run Smart Cull** to star the best of each burst. Rate and label with the keyboard; **Survey** and **Face Loupe** to compare
- **Reject Non-Picks**, then **Review Rejected** to check before anything is left out

### 2. Develop and retouch
- Double-click a frame (or press **Enter**) to open it in **Develop**; arrow keys move through the folder
- **Sync** your settings to the rest of the selection
- **Retouch** for skin, teeth, eyes, spots and local adjustments

### 3. Design
- Add a watermark or caption; it is drawn over the final, cropped frame at export

### 4. Export
- Choose a **preset** and optional **watermark**, then **Export Selected**. Or make an Instagram set, a reel, or a web gallery from the same edits

### 5. Video
- **Load Video** in the Video workspace and chain operations; each saves into `<video-name>-edit/`, and **Reset to Original** starts over

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

```bash
pip install pytest
python tests/fetch_media.py   # optional: CC0 camera RAW samples and reference faces, ~190 MB
pytest
```

**493 tests.** They drive the real engines and the real window (headless,
with `QT_QPA_PLATFORM=offscreen`); nothing is mocked except file dialogs.
Tests that need a downloaded model or sample file are marked `model` or
`media` and skip themselves when it is absent.

The first file, `tests/test_invariants.py`, hashes every original before and
after each action that reads the library (cull, reject, blur triage, grade,
level, sidecars, export, background removal, Instagram, reel, gallery, video)
and fails on any change, and checks that a rejected frame never reaches an
export. Every new feature extends it.

`tests/bench_cull.py` times a 1,000-frame cull per camera and records the
results in its docstring.

### Key Dependencies
- **PyQt6**: Desktop UI (dark theme, fade transitions, hardware info display)
- **rawpy**: RAW demosaicing + embedded thumbnail extraction
- **Pillow / pillow-heif**: JPEG, PNG, HEIC support
- **OpenCV**: Sharpness analysis, selective color masking, video stabilization/auto-crop
- **numpy**: Image arrays, matrix math for color grading
- **moviepy**: Reel rendering
- **rembg**: AI background removal (U2Net ONNX model)
- **imageio-ffmpeg**: Bundled FFmpeg binary for the video engine — no system install required
- **onnxruntime-directml** (Windows) / **onnxruntime**: AI model runtime
- **lensfunpy**: Lens distortion, fringe and vignetting profiles
- **scipy, tifffile**: Curve interpolation; 16-bit TIFF export
- **fastapi / uvicorn**: Optional local REST API for the video engine
- **psutil**: RAM and CPU core detection

## Architecture Highlights

### Edit-Safe Workflow
Loading a folder creates an `-edit` folder *inside* it. Your edits are
**settings**, kept there as small files; nothing is rendered until you export:

```
Vacation2024/                       # your originals — never modified or moved
├── DSC_0001.NEF
├── DSC_0002.NEF
└── Vacation2024-edit/              # created on load
    ├── DSC_0001.NEF.granite.json   # this photo's edit recipe
    ├── session.granite.json        # ratings, labels, picks, rejects
    ├── people.granite.json         # who appears where (if you grouped by person)
    └── DSC_0002_nobg.png           # background removal output
```

Because an edit is a list of settings, any of it can be changed or undone
later, in any order, and there is no chain of re-saved JPEGs to lose quality
along. A damaged settings file never stops a photo opening; it opens unedited
and says why. Files written by version 1 (`*_graded.jpg` and so on) are left
alone, and the web gallery still reads them.

The video editor works differently: each operation writes a new clip into
`clip-edit/`, which is what lets operations chain and **Reset to Original**
work. Exports, Instagram sets, reels and galleries are saved where you choose.

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
- Culling reads the camera's embedded JPEG from each RAW instead of developing it, and uses every core
- The Develop preview renders from a 1600 px proxy with exactly the export's pipeline; simple slider moves take about 15 ms on an i5-8350U
- Two cameras in the test set (an iPhone XS DNG and a Hasselblad H3D) carry no usable embedded preview, so they are developed at half size and cull several times slower

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
| Native RAW, phone DNG, HEIC, JPEG, TIFF with no pre-conversion | **Done** |
| 1,000 RAW culled fast | **Done for 5 minutes on all nine cameras tested; 90 seconds on five of nine** — see below |
| Focus scoring on the faces, not the background | **Done** |
| Blink detection by eye aspect ratio | **Done** — 478-point mesh. Checked on four faces with eyes painted shut, not on real blinks (see below) |
| Smile and open-mouth detection | **Done** — from the mesh's expression coefficients |
| Duplicate and burst detection, cull to a target | **Done** |
| Identify the same person across a shoot | **Done** — local face recognition and clustering. Checked for self-consistency, not measured on a real event |
| Survey, face loupe, keyboard culling | **Done** |
| Ratings and labels to `.xmp` sidecars | **Done** — merges, never clobbers |

### Editing & retouching

| Requirement | Status |
| --- | --- |
| Parametric RAW develop (tone, curves, HSL, grading, dehaze, lens, LUTs, geometry) | **Done** |
| Layers, blend modes, masks | **Done** |
| Subject, skin, teeth, eyes masks | **Done** |
| Hair and sky masks | **Partial** — work, but unmeasured; offered as a starting point |
| Skin smoothing, healing, cloning, teeth, eyes | **Done** |
| Inpainting and corner fill | **Done** — fast fill always; neural fill with a 93 MB model. Quality not yet judged on real prints |
| Vector text, shapes, watermarks, SVG and PDF | **Done** |
| ProRAW local tone mapping | **Not built** — detected and reported, not applied |
| AI profiles trained from a photographer's Lightroom catalogue | **Not built** — a training pipeline, the largest item in the spec |
| Marketplace of styles | **Not built** |

### Client delivery

| Requirement | Status |
| --- | --- |
| Static web gallery with category tabs | **Done** |
| Client proofing, favourites, face search, print ordering | **Not built.** These need a hosted backend, accounts and payments |

### Why auto-levelling ignores head roll

Version 1 could not measure head roll at all: the five-point face detector
pulls its landmarks toward an upright face, so rotating a portrait **24°**
moved the measured eye line about **5°**, and detection confidence stayed
**flat across ±20°**.

The 478-point face mesh in 2.0 *can* measure it: the eye-corner line tracked
a known rotation to within about **1° across ±25°** (5° off at 40°). It is
still not used to level the photograph, for the same reason spine lean is
not: a tilted head is usually the subject's pose, and "correcting" it would
rotate the whole frame away from a level horizon. Levelling stays with the
scene's lines and the body.

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

**Corner fill** is now built: choose *Fill* and the wedges straightening
leaves are painted in at export, by a fast fill or, with its model, by LaMa.

**A note on what gets measured.** `line_tilt` previously read the black
wedges left by an earlier rotation as scene structure: their borders are
flawless straight lines sitting at precisely the angle that produced them, so
a corrected frame looked strongly tilted. Blank regions are now masked out
before edge detection. This surfaced only because a test fed it a padded
frame — any already-rotated, letterboxed or scanned photograph would have
hit it.

### How blink detection actually works, and what it is not

A 478-point face mesh is fitted to every detected face, and each eye's
**aspect ratio** is measured: its height over its width,
`(|p2 − p6| + |p3 − p5|) / (2 |p1 − p4|)`. That is a ratio of distances on
the face itself, so unlike version 1's "how dark is the eye patch" it does
not depend on iris colour or lighting.

Two meshes were tried, and the measurement decided between them:

| Mesh | Open eyes (4 faces) | Same faces, eyes painted shut |
| --- | --- | --- |
| 468-point (older) | 0.22–0.32 | 0.17–0.26 — overlaps, unusable |
| **478-point (used)** | **0.155–0.282** | **0.071–0.142** |

The model's own "blink" output was also tried and did worse than the plain
ratio, so it is not used for blinks. Be clear about the limits:

- The "shut" faces are the open ones with the eyes painted over, because no
  labelled photographs of real blinks were available. That shows the
  measurement responds; it is not validation on real closed eyes.
- Four faces is four faces. Nothing is established across skin tones, ages,
  glasses or lighting.
- A small face looking down reads low: one reference face (about 30 px
  across) measured 0.155 open and 0.142 shut, both beside the 0.135
  threshold. It is recorded as a known limit in the tests.
- A frame counts as a blink only when a main face's **more open** eye is
  below the threshold; one low eye is far more often a wink, hair or a turned
  head. Without the mesh model, the version 1 heuristic is used and the frame
  says so.

### A note on the performance target

The spec asks for 1,000 RAW files culled in under 90 seconds. Measured on an
**i5-8350U laptop** (4 cores, integrated graphics), 1,000 frames per camera,
with eye checking on:

| Camera | Seconds per 1,000 |
| --- | --- |
| Sony a6400 (ARW) | **42** |
| Samsung S10 (DNG) | **51** |
| Olympus E-M1 II (ORF) | **53** |
| Canon EOS R6 (CR3) | **61** |
| Panasonic GH5S (RW2) | **85** |
| Fujifilm X-S10 (RAF) | 91 |
| Nikon Z50 II (NEF) | 96 |
| Canon 5D IV (CR2) | 100 |
| Pentax K-1 II (PEF) | 104 |

So **under 90 seconds for five of nine cameras, and under 5 minutes for all
nine.** The first build took 95–258 seconds; decoding a greyscale copy for
sharpness and a small colour copy for faces is what closed most of the gap.

Read the numbers for what they are: each run repeats one sample file a
thousand times, so after the first read it comes from memory. It measures
decoding and analysis, not reading a thousand different files off a card.
Two cameras with no usable embedded preview (iPhone XS DNG, Hasselblad H3D)
have to be developed instead and took 291 and 616 seconds in the first build.
The full method is in `tests/bench_cull.py`.

## Contributing

Bug reports, feature suggestions, and PRs welcome. Please open an issue first so we can discuss.

---

Built with ❤️ by photographers, for photographers.  
**Granite Studios** — your RAW, edited locally. Never in the cloud.
