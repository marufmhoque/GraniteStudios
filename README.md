# Granite Studios

**Local-first AI photo and video editing suite** for photographers and content creators. Designed to run smoothly on modest hardware (i5-8350U + 32GB RAM), but scales to powerful machines with CUDA or Apple Silicon.

- 📸 **Universal input**: RAW (Nikon, Canon, Sony, etc.), HEIC, JPEG, PNG, and more
- 🤖 **Smart culling**: Perceptual-hash duplicate detection, burst grouping, sharpness scoring
- 🎨 **AI color grading**: Selective color isolation, cinematic matrix filters, one-click auto-edit
- 📹 **Reel maker**: Turn 3–7 images into a scrolling 9:16 panoramic video
- 🌐 **Web publisher**: Responsive Masonry gallery with category tabs and camera metadata
- 💾 **Edit-safe workflow**: All edits, exports, and "keepers" live in a separate `-edit` subfolder; originals never touched

## Quick Start

### Windows
1. Download the latest `GraniteStudios.exe` from [Releases](https://github.com/marufmhoque/GraniteStudios/releases)
2. Run it — a Desktop shortcut is created automatically on first launch
3. See [README_INSTALL.md](README_INSTALL.md) for SmartScreen bypass steps

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

## Features

### Library & Culling
- **Universal loader**: One-click folder import. Embedded RAW thumbnails cached for instant preview scrolling
- **Smart cull**: Laplacian-variance sharpness scoring + timestamp-based burst grouping. Auto-stars the top 2 sharpest of each burst
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
├── package_windows.py        # PyInstaller builder for .exe (Pillow icon conversion)
└── .github/workflows/build.yml # CI: macOS .dmg builder
```

## Usage Workflow

### 1. Load & Cull
- Click **Load Folder** in the Library tab
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
- **Reel Maker**: Select 3–7 images → outputs an MP4 panoramic scroll
- **IG Formatter**: Select images → pads them to 4:5 or 9:16 with a blurred background

### 5. Edit Video
- Click **Load Video** in the Video Editor tab (any MP4/MOV/AVI/MKV/WebM/FLV file)
- Run operations in any order — trim, speed, reframe, stabilize, color-correct, denoise. Each one saves into `<video-name>-edit/` and becomes the input to the next, so they chain; **Reset to Original** starts over
- Pick an **Export preset** (Web MP4, MOV, WebM, Instagram Reel) and click **Export**
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
All core modules compile and have been smoke-tested:
- `hardware.py`: CPU/GPU detection, RAM profiling
- `universal_loader.py`: Preview caching, RAW thumbnails, EXIF extraction
- `culling.py`: Sharpness scoring, perceptual hashing, burst grouping
- `color_engine.py`: Selective color, cinematic filters
- `publisher.py`: Category-tabbed gallery generation

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
Every project gets a sibling `-edit` subfolder:
- **Original folder**: `Vacation2024/` (unchanged)
- **Edit folder**: `Vacation2024-edit/` (graded JPEGs, keeper flattens, resizes)
- **Web exports** pull exclusively from the edit folder

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

- [ ] ONNX subject detection for selective masks (auto-star hero shots)
- [ ] EXIF-driven smart grouping (camera + lens combo)
- [ ] Batch watermarking
- [ ] Direct Instagram upload (currently exports, you upload)
- [ ] Keyboard shortcuts for rapid culling

## Contributing

Bug reports, feature suggestions, and PRs welcome. Please open an issue first so we can discuss.

---

Built with ❤️ by photographers, for photographers.  
**Granite Studios** — your RAW, edited locally. Never in the cloud.
