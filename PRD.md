# Granite Studios — Product Requirements Document

| | |
| --- | --- |
| **Version** | 1.0 |
| **Date** | 12 November 2026 |
| **Status** | Living document — tracks `main` |
| **Applies to** | commit `6531d3a` and later |
| **Owner** | Maruf Hoque |
| **Repository** | https://github.com/marufmhoque/GraniteStudios |

---

## How to read this document

`README.md` is the **user guide**: how to install the app and what the buttons
do. This document is the **specification**: who the product is for, what it
must never do, what each capability has to satisfy to count as working, how
well it actually works, and what it should become.

### The evidence rule

A product requirements document normally carries market sizing, user research
and competitive analysis. **None of that exists for this project, and this
document does not invent it.**

Three conventions make that explicit, and they hold throughout:

- **Measured** — a number that traces to a recorded test run, a module
  docstring, or the verification tables in `README.md`. Cited with its
  conditions and sample size.
- **Assumption** — a belief about users, market or value that has *not* been
  validated. Flagged inline. Sections [2.4](#24-target-users) and
  [11](#11-success-metrics) are assumption-heavy by nature and say so.
- **Unmeasured** — a thing we believe works but have not tested. Named rather
  than quietly counted as working.

Where a conventional PRD would cite data we do not have, this one says so.
That restraint is deliberate: the project's engineering discipline has been to
measure before claiming, and several features exist in their current shape
precisely *because* a measurement contradicted the obvious design. A document
that papered over that would misrepresent the product.

### Status vocabulary

| Term | Meaning |
| --- | --- |
| **Done** | Implemented and verified. Evidence cited. |
| **Partial** | Implemented in part. The gap is stated explicitly. |
| **Not built** | Not implemented. The actual blocker is named. |

---

## Table of contents

1. [Product overview](#1-product-overview)
2. [Users and positioning](#2-users-and-positioning)
3. [Product invariants](#3-product-invariants)
4. [Functional requirements](#4-functional-requirements)
5. [Non-functional requirements](#5-non-functional-requirements)
6. [Architecture](#6-architecture)
7. [Quality and verification](#7-quality-and-verification)
8. [Known limitations](#8-known-limitations)
9. [Target state](#9-target-state)
10. [Release and distribution](#10-release-and-distribution)
11. [Success metrics](#11-success-metrics)
12. [Risks and open questions](#12-risks-and-open-questions)
13. [Appendix](#13-appendix)

---

## 1. Product overview

### 1.1 Problem statement

A photographer returns from a wedding or an event with two to five thousand
frames. Before any creative work begins, that set has to be reduced: throw out
the misfires, pick one frame from each burst of near-identical shots, discard
the ones where somebody blinked, straighten what was shot handheld. This is
mechanical, repetitive, and it routinely takes longer than the editing that
follows it.

Software exists to automate it. The established options share two properties
that this project rejects:

1. **They upload your work.** Cloud culling means transferring a client's
   unreleased photographs to a third party.
2. **They rent, rather than sell.** Access ends when payment does, including
   access to the catalogue decisions already made.

*(Assumption — the significance of these two properties to photographers is a
design belief, not a validated finding. See [2.4](#24-target-users).)*

### 1.2 Vision

**Granite Studios is a local-first photo and video suite that does the
mechanical work of a shoot without the photographs ever leaving the machine.**

Three commitments follow from that sentence, and they are enforced in code
rather than asserted in marketing — see [3. Product invariants](#3-product-invariants):

- Nothing is uploaded. The only outbound network traffic in the entire desktop
  application is downloading AI models, each verified against a pinned digest.
- Original files are never modified, moved or deleted. Not on reject, not on
  edit, not on export, not on uninstall.
- It runs on modest hardware. The development target is an i5-8350U with
  integrated graphics; a dedicated GPU is an accelerant, never a requirement.

### 1.3 Scope

| In scope | Out of scope |
| --- | --- |
| Ingestion of RAW, HEIC and standard formats | Being a digital asset manager (no catalogue database) |
| Automated culling, blur triage, levelling | Cloud sync, accounts, or multi-device state |
| Colour grading and background removal | Replacing Lightroom's full develop module |
| Static web gallery generation | Hosted client proofing, print ordering, payments |
| A complete FFmpeg-backed video editor | Non-linear multi-track timeline editing |
| Self-updating Windows installation | App-store distribution; code signing |

---

## 2. Users and positioning

> **This entire section is assumption, not research.** No user interviews,
> surveys or market analysis have been conducted. It records the beliefs the
> product was designed around so they can be tested and corrected later.

### 2.1 Primary persona (assumed)

**The high-volume event photographer.** Shoots weddings, graduations or
corporate events. Returns with 2,000–5,000 RAW frames per job. Already owns
Lightroom and has a develop workflow they like. Their pain is the hours of
culling before that workflow starts. They are the reason the cull, blur-triage
and levelling features exist and the reason XMP sidecar interop matters more
than a built-in develop module: the app should hand off to the tool they
already use rather than try to replace it.

### 2.2 Secondary persona (assumed)

**The content creator.** Shoots stills and short video for social. Needs
reframing to 9:16 and 4:5, quick grading, and output that can be posted
without a round trip through a separate editor. The reel maker, IG formatter
and video engine serve this persona.

### 2.3 Tertiary persona (assumed)

**The hobbyist on modest hardware.** Has a laptop with integrated graphics and
8–16 GB of RAM, and is used to creative software being unusable on it. Served
by the hardware budgeting, the LRU preview cache and the decision never to
require a GPU.

### 2.4 Target users

These three personas are **unvalidated**. The product is currently used by its
author. The most valuable near-term research would be observing one
high-volume photographer complete a real cull, because several design
decisions — the strictness tiers, the rejection-review drawer, the target-count
cull — encode guesses about how people actually work that have never been
checked against anyone.

### 2.5 Positioning

Framed as **design intent**, not competitive research. No benchmarking against
these products has been performed.

| Product | How Granite Studios is positioned relative to it |
| --- | --- |
| Aftershoot | Same problem (AI culling + editing), opposite architecture: local and owned rather than cloud and subscribed. Granite Studios is far less mature — see [9. Target state](#9-target-state) for the honest gap. |
| Lightroom Classic | Not a competitor. Granite Studios aims to hand off *to* it via XMP sidecars that merge rather than overwrite. |
| Photo Mechanic | Overlaps on fast culling and sidecar output. Granite Studios adds automated judgement (blur class, blink, levelling) that Photo Mechanic leaves to the user. |

### 2.6 Non-goals

- **Not a DAM.** No catalogue, no database, no persistent library state. The
  filesystem is the source of truth.
- **Not a cloud service.** The hosted browser build exists for evaluation and
  convenience; it is not the product's centre of gravity.
- **Not a Lightroom replacement.** Grading exists to make images web-ready, not
  to replace a develop module.
- **Not a destructive tool.** There is no code path that deletes a user's
  original photograph. See [FR-6.4](#fr-6-review-and-rejection).

---

## 3. Product invariants

These are the rules the software must never break. They outrank feature
requests. Each is enforced somewhere specific, named here so the claim can be
checked rather than trusted.

### INV-1 — Originals are never modified, moved or deleted

**"Reject" means the file does not get copied into the `-edit` subfolder.** It
does not mean delete, and there is no code path in the application that deletes
a user's original photograph.

- **Enforcement:** `_drop_rejected()` in `src/ui/main_window.py`, applied at
  all six export points — grading, background removal, panoramic reel,
  Instagram export, web gallery, auto-level. Any new export path must call it;
  that is the whole of the enforcement, so omitting it is the one way this
  invariant can be broken.
- **Verification:** every destructive-looking operation is tested by hashing
  the source directory before and after the run and asserting the hashes are
  unchanged. Most recently: *ORIGINALS UNCHANGED: True | 4 of 4* through the
  real auto-level handler.
- **Consequence:** the review drawer ([FR-6.3](#fr-6-review-and-rejection))
  is reversible by construction — nothing has been destroyed to undo.

### INV-2 — Images never leave the machine

The desktop application makes **no outbound network request containing user
data**. The only outbound traffic is model downloads from OpenCV's model zoo.

- **Enforcement:** `src/ai_tools/model_store.py` is the sole network client in
  the photo pipeline.
- **Note:** the hosted browser build ([FR-13](#fr-13-hosted-browser-version))
  necessarily uploads to its own backend. That is a different deployment with a
  different privacy posture and must always be described as such.

### INV-3 — Degrade, never guess

A missing, corrupt or unfetchable model disables the feature that needs it. It
does not raise, and it does not fall back to invented output.

- **Enforcement:** `ensure_model()` returns `None` rather than raising;
  `FaceAnalyzer.available` / `PoseAnalyzer.available` gate every caller.
- **Verification:** with the model directory redirected to an empty path, the
  pose analyzer reports `available: False`, analysis returns an empty report
  with `model_available=False`, the UI builds with the body engine unchecked,
  and levelling still works via scene lines.

### INV-4 — Never claim unmeasured accuracy

Where a measurement shows a signal is not good enough, the feature reports it
and declines to act on it, rather than acting with a caveat in the docs.

- **Example:** shoulders-only body levelling. Measured residual 2.90° mean /
  9.54° worst, against 0.53° / 1.33° when hips are also visible. It is reported
  in the posture view and **never auto-applied** — enforced by
  `MIN_APPLY_CONFIDENCE = 0.55`, which a shoulders-only reading cannot reach by
  construction (it scores at most 0.45).
- **Example:** two approaches to face-based head-roll levelling were built,
  measured, found not to work, and **deleted**. The measurements are preserved
  in `src/ai_tools/orientation.py` so the idea is not retried blind.

### INV-5 — Downloaded models are content-addressed

Every model is pinned by SHA-256 and byte size taken from the upstream git-lfs
pointer. A download that does not match is rejected rather than handed to the
DNN loader. A cached file that fails verification is deleted and refetched.

- **Enforcement:** `_matches()` / `verify()` in `src/ai_tools/model_store.py`.

### INV-6 — One pane, not three

Import, culling, rejection and grading happen against a single grid that stays
on screen. These were formerly three separate tabs, which hid the photographs
at exactly the moment the user needed to see what an action had done.

- **Enforcement:** `_build_photos_page()` in `src/ui/main_window.py`;
  navigation is a four-item `QStackedWidget` (`NAV_ITEMS`).

---

## 4. Functional requirements

### FR-1 Ingestion and library

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-1.1 | Open a folder of mixed-format images with no pre-conversion | All of `SUPPORTED_EXTENSIONS` load: 8 RAW (`.nef .cr2 .cr3 .arw .dng .raf .orf .rw2`), 2 HEIF (`.heic .heif`), 7 standard (`.jpg .jpeg .png .tif .tiff .webp .bmp`) | **Done** |
| FR-1.2 | Preview scrolling must stay smooth on integrated graphics | Embedded RAW thumbnails used where present; previews LRU-cached, bounded at 256 images; decode off the UI thread | **Done** |
| FR-1.3 | Surface capture metadata | Camera make/model, lens, ISO, aperture, shutter from EXIF, shown on hover | **Done** |
| FR-1.4 | Create the edit-safe subfolder on load | Loading `Vacation2024/` creates `Vacation2024/Vacation2024-edit/`; idempotent | **Done** |
| FR-1.5 | Flatten RAW keepers to JPEG on first export | Idempotent — repeat calls are instant | **Done** |

**Module:** `src/engine/universal_loader.py`.
**Verified:** folder scan, RAW/HEIC/JPEG/PNG support, preview and array decode,
EXIF extraction, `-edit` folder creation.

### FR-2 Culling

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-2.1 | Score every frame for focus quality | Laplacian variance over a downscaled preview (`analysis_max_px = 1200`); a sharp frame must rank above a blurred one | **Done** |
| FR-2.2 | Group bursts by capture time | EXIF timestamp, default `time_threshold = 2.0 s` | **Done** |
| FR-2.3 | Detect near-duplicates | Perceptual hash + Hamming distance, default `hash_threshold = 12` | **Done** |
| FR-2.4 | Auto-star the best of each burst | Default `top_count = 2`; all other frames in the group marked duplicate | **Done** |
| FR-2.5 | Cull to a target count | "Best 400 of 3,000" — quota spread round-robin across bursts so one sharp stretch of the day cannot consume the whole selection. Clamps above the image count; rejects a negative target | **Done** |
| FR-2.6 | Detect closed eyes and rank accordingly | With **Skip blinks** on, an open-eyed frame outranks a *sharper* frame where everyone blinked | **Done** — heuristic, see [8.2](#82-blink-detection-is-a-heuristic) |
| FR-2.7 | Blink detection must require both eyes closed | One eye below threshold is far more often a highlight, stray hair or head turn than a blink | **Done** |

**Modules:** `src/ai_tools/culling.py`, `src/ai_tools/face_analysis.py`.

**Measured:** blink-aware ranking verified on bursts constructed so that
sharpness and open eyes *disagree* — with eye checking off the sharpest frame
wins despite closed eyes; with it on the softer open-eyed frame is chosen
instead. Face detection: two real faces detected, **no false positives across
six non-face sample images**, stable from 0.4× to 2× scale.

### FR-3 Blur triage

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-3.1 | Distinguish motion blur from missed focus | Motion smears along one axis (gradients bunch directionally); defocus attenuates all directions equally | **Done** |
| FR-3.2 | Report the direction of a motion smear | Angle of travel reported with the verdict | **Done** |
| FR-3.3 | Offer strictness tiers | `aggressive` / `balanced` (default) / `conservative`; the same frame must be able to receive different verdicts across tiers | **Done** |
| FR-3.4 | Blurry frames are rejected, not deleted | Routed to the same review drawer as any other rejection; individually restorable | **Done** (INV-1) |
| FR-3.5 | Offer sharpening only where detail remains | Frames below `rescue_floor` are not offered. UI must state that it lifts edge contrast rather than reconstructing detail | **Done** |

**Module:** `src/ai_tools/blur_analysis.py`.

**Measured:** sharp frames not flagged across 3 reference images; defocus
classified correctly at 2 blur radii each; **15/15 motion cases detected with
direction within 15°**; strictness tiers verified to change the verdict on the
same frame.

**Design note (FR-3.5).** The rescue gate cannot be decided from the sharpness
score, because sharpening raises the Laplacian score roughly **2.4×** even on
unrecoverable frames — that is halo contrast, not recovered detail. The gate is
therefore an absolute floor on remaining detail, not a delta.

**Design note (FR-3.1).** Directionality alone is insufficient: a *sharp*
photograph of a fence or horizon is strongly directional, and in testing one
sharp reference frame scored more directional than its own defocused copy.
Directionality is consulted only once overall sharpness is already low enough
for the frame to be in question.

### FR-4 Orientation and levelling

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-4.1 | Recover 90/180/270 rotations | Face detection does not fire on an upside-down face, so each quarter turn is tried | **Done** |
| FR-4.2 | Recover rotation with no face visible | Hip→shoulder direction read off pose landmarks, rounded to a quarter turn | **Done** |
| FR-4.3 | Level from scene structure | Long straight edges folded to "distance from square", length-weighted median | **Done** |
| FR-4.4 | Level a frame with no usable lines | Shoulder and hip lines carry it | **Done** |
| FR-4.5 | Prefer lines over body where both exist | Lines measure the camera against the world; a person can genuinely be leaning | **Done** |
| FR-4.6 | Never read blank padding as scene structure | Near-black regions masked before edge detection | **Done** |
| FR-4.7 | Handle rotation corners honestly | Crop to largest inscribed rectangle, or leave black. **No generative fill** | **Done** |
| FR-4.8 | Skip corrections not worth making | `MIN_CORRECTION_DEG = 0.75`; implausible readings above `MAX_FINE_CORRECTION_DEG = 20.0` discarded | **Done** |

**Module:** `src/ai_tools/orientation.py`.

**Measured:** all 90/180/270 turns recovered — and still recovered with the
face engine disabled entirely, using the body. Known tilts round-trip to under
1° residual (max 0.04° across the 7-frame scene-line set). Body-only levelling
on a line-free portrait: **0.6° mean error, 1.7° worst** across a ±14° sweep.

**Design note (FR-4.6).** This was a real defect, not a hypothetical. The black
wedges left by an earlier rotation are flawless straight lines sitting at
precisely the angle that produced them, so a *corrected* frame measured as
strongly tilted. Any already-rotated, letterboxed or scanned photograph would
have hit it. It surfaced only because a test fed the detector a padded frame
and the result was suspiciously perfect.

### FR-5 Posture analysis

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-5.1 | Report shoulders, spine and hips | 33-point BlazePose topology, landmarks in original-image coordinates | **Done** |
| FR-5.2 | Distinguish camera roll from a leaning subject | Roll turns shoulders and hips together; a lean moves them against each other. Correction offered only when they agree within `MAX_ASYMMETRY_DEG = 12.0` | **Done** |
| FR-5.3 | Never auto-level from shoulders alone | `MIN_APPLY_CONFIDENCE = 0.55`; shoulders-only scores at most 0.45 | **Done** (INV-4) |
| FR-5.4 | Report spine lean, never correct it | Lean belongs to the subject, not the camera. Straightening it would be editing the pose | **Done** |
| FR-5.5 | Report framing honestly | `full body` / `three-quarter` / `head and shoulders` / `partial`, from landmark visibility ≥ `MIN_VISIBILITY = 0.55` | **Done** |
| FR-5.6 | Pick the right subject in a group | Largest confident body wins — a bystander at frame edge must not decide the rotation | **Done** |
| FR-5.7 | Ask before downloading models | ~17 MB, consent dialog, digest-verified, nothing uploaded | **Done** |
| FR-5.8 | **Check Posture** must change nothing | Read-only reporting | **Done** |

**Module:** `src/ai_tools/pose_analysis.py`.

**Measured:** landmarks verified visually against the image (skeleton lands on
shoulders, elbows, wrists, hips, knees, ankles). **Body tilt tracks a known
roll 1:1 across ±24°.** Levelling residual 0.53° mean / 1.33° worst with hips
visible. A half-out-of-frame body spread 57° between shoulder and hip lines and
was correctly refused. Deterministic across five repeat runs on the same frame.

**Design note (FR-5.2).** The 12° agreement gate is **measured, not chosen**.
It was initially set to 6°, which refused ordinary upright photographs: a
subject standing square still reads a 4–7° shoulder/hip spread frame to frame,
because the two lines are regressed independently.

### FR-6 Review and rejection

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-6.1 | Select all / none / invert | Scoped to what the view filter is currently showing | **Done** |
| FR-6.2 | Reject manually | Marks photos out of the export set | **Done** |
| FR-6.3 | Review before anything is acted on | Rejected frames shown as thumbnails with a per-photo tick to restore | **Done** |
| FR-6.4 | Rejection is never deletion | The original file is never moved, renamed or removed | **Done** (INV-1) |
| FR-6.5 | Reject everything a cull did not pick | One click, reversible | **Done** |
| FR-6.6 | Make the cull's decision visible | Filter the grid to *Keepers* / *AI picks* / *Duplicates* / *Rejected* / *Blurry* / *Motion blur* / *Out of focus* | **Done** |

**Design note (FR-6.1).** Selection is scoped to visible frames deliberately:
selecting frames the user cannot see would be a trap, particularly when the
next action is "reject".

### FR-7 Colour and grading

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-7.1 | Selective colour isolation | 8 hues (red, orange, yellow, green, cyan, blue, purple, magenta); isolated hue keeps saturation, everything else goes monochrome | **Done** |
| FR-7.2 | Cinematic grades | 6 matrix grades with lift/gamma curves: teal-orange, warm film, cool matte, bleach bypass, vintage fade, noir | **Done** |
| FR-7.3 | One-click auto-fix | White balance + CLAHE equalisation | **Done** |
| FR-7.4 | Manual adjustment | Brightness, contrast, saturation | **Done** |
| FR-7.5 | Output to the edit folder | `*_graded.jpg` | **Done** (INV-1) |

**Module:** `src/ai_tools/color_engine.py`. All 6 filters and all 8 hues verified.

### FR-8 Background removal

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-8.1 | Produce a subject cutout | U2Net via `rembg`, CPU-viable | **Done** |
| FR-8.2 | Export transparent PNG | `*_nobg.png`, RGBA at source size, non-uniform alpha | **Done** |
| FR-8.3 | Composite a replacement background | — | **Not built** — cutout exists; compositing does not |
| FR-8.4 | Expose the mask as editable | — | **Not built** — mask is internal |

### FR-9 Interoperability (XMP sidecars)

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-9.1 | Write star ratings and colour labels | `.xmp` beside each image; ratings 0–5 and −1 (reject) round-trip; all five Lightroom labels | **Done** |
| FR-9.2 | **Merge, never clobber** | An existing sidecar's `crs:` develop settings, `dc:title` and keywords must survive a write | **Done** |
| FR-9.3 | Survive a corrupt sidecar | Degrade rather than raise | **Done** |
| FR-9.4 | Never modify the image file | Sidecar only | **Done** (INV-1) |

**Module:** `src/ai_tools/xmp_sidecar.py`. Target applications: Lightroom
Classic, Capture One, Photo Mechanic, Bridge.

**Design note.** FR-9.2 is the requirement that makes the feature usable at
all. A sidecar writer that overwrites destroys another tool's work, which would
make the hand-off to Lightroom — the entire point of the feature — actively
harmful.

### FR-10 Web gallery publishing

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-10.1 | Generate a self-contained static gallery | Responsive masonry HTML + images, uploadable to any host | **Done** |
| FR-10.2 | Publish selectively | Per-thumbnail "Publish to Web" checkbox | **Done** |
| FR-10.3 | Category tabs | Per-image labels rendered as tabs | **Done** |
| FR-10.4 | Custom gallery title | Per batch | **Done** |
| FR-10.5 | Read only from the edit folder | Originals are not sources for publication | **Done** (INV-1) |

### FR-11 Social export

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-11.1 | Panoramic scrolling reel | 3–7 images stitched; 1080×1920 viewport panning over 10 s; MP4. Rejects <3 or >7 | **Done** |
| FR-11.2 | Instagram feed format | Exactly 1080×1350 over a blurred, darkened copy of the image | **Done** |
| FR-11.3 | Instagram story format | Exactly 1080×1920, same treatment | **Done** |
| FR-11.4 | Direct upload to Instagram | — | **Not built** — exports only |

### FR-12 Video engine

Ships its own FFmpeg binary via `imageio-ffmpeg`; **no system FFmpeg install
required**.

| ID | Area | Capabilities | Status |
| --- | --- | --- | --- |
| FR-12.1 | Timing | Trim, reverse, freeze-frame, constant and ramped speed, time-lapse condensation | **Done** |
| FR-12.2 | Spatial | Aspect conversion (16:9 / 9:16 / 1:1 / 4:5 / 4:3, crop or letterbox), explicit crop, subject-aware auto-crop, point-feature stabilisation, static-region redaction | **Done** |
| FR-12.3 | Structure | Split, multi-clip concat, crossfade/wipe transitions, J-cut and L-cut audio offsets | **Done** |
| FR-12.4 | Audio | Track mixing, EBU R128 loudness normalisation, sidechain auto-ducking, FFT noise reduction, volume | **Done** |
| FR-12.5 | Colour | Brightness/contrast/saturation/gamma, temperature white balance, 3D LUT (`.cube`) | **Done** |
| FR-12.6 | Export | 4 presets: Web MP4 (H.264, faststart), MOV (high-quality H.264), WebM (VP9/Opus), Instagram Reel (1080×1920) | **Done** |
| FR-12.7 | Ingestion | MP4, MOV, AVI, MKV, WebM, FLV — any container FFmpeg decodes | **Done** |
| FR-12.8 | Chaining | Each operation writes a new file into `<name>-edit/` and becomes the next operation's input; **Reset to Original** restores | **Done** (INV-1) |
| FR-12.9 | Pipelines | `VideoJob` / `JobManager` multi-step async pipelines, thread pool sized to leave CPU headroom | **Done** |
| FR-12.10 | Local REST API | Optional FastAPI server: `GET /probe`, `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `POST /concat`, `POST /transition` | **Done** |

**Verified:** every timing, spatial, structure, audio and colour operation; all
4 export presets; `.cube` LUT application; multi-step `JobManager` pipelines
**including the failure path**; all 6 API routes plus 400/404 handling.

### FR-13 Hosted browser version

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-13.1 | Run the same engines as the desktop app | Backend imports `src/` unchanged — no divergent pipeline | **Done** |
| FR-13.2 | Session-scoped storage | Temp files, TTL-swept | **Done** |
| FR-13.3 | API surface | `/api/health`, `/api/capabilities`, `/api/session`, `/api/upload`, `/api/file/{id}`, `/api/preview/{id}`, `/api/photo/edit`, `/api/photo/instagram`, `/api/video/edit`, `/api/video/reel`, `/api/job/{id}` | **Done** |
| FR-13.4 | Deployment | FastAPI + Docker → Render; React + Vite + Tailwind → Vercel | **Partial** — see [12.4](#124-the-hosted-build-is-not-deployed) |

**Privacy note.** This deployment uploads user images to a server by necessity.
It is a *different* privacy posture from the desktop app and must never be
described as local-first.

### FR-14 Installation, update and uninstall

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-14.1 | One file to install | A single `.bat` that fetches the app itself — nothing to clone | **Done** |
| FR-14.2 | Same file updates | Re-running installs over the old version, leaving photos and `-edit` folders alone | **Done** |
| FR-14.3 | Never go stale | GitHub Action republishes on every push to `main`; the file fetches `main` at run time | **Done** |
| FR-14.4 | Private Python environment | Installs to `%LOCALAPPDATA%\Granite Studios`, own venv, cannot disturb other Python | **Done** |
| FR-14.5 | Verify the install actually works | Finishes by starting the app headlessly rather than assuming success | **Done** |
| FR-14.6 | Diagnosable failures | Full transcript plus separate pip log; on failure prints the path, opens it, holds the window open | **Done** |
| FR-14.7 | Plain-language diagnosis of common failures | App already running; no Python 3.10+ (links to download rather than failing cryptically) | **Done** |
| FR-14.8 | Register for uninstall | Appears in Settings → Apps; uninstall never touches photos or `-edit` folders | **Done** |
| FR-14.9 | Survive version skew | Bootstrap feature-detects installer parameters via AST inspection rather than assuming a signature | **Done** |

**Design note (FR-14.9 / FR-14.6).** These exist because of real failures. An
older `install.ps1` without a `-LogPath` parameter received one from a newer
bootstrap and failed hard. Separately, a **successful** install was aborted by
a harmless Qt font warning on stderr: under PowerShell 5.1 with
`$ErrorActionPreference = 'Stop'`, capturing native stderr via `2>&1` turns it
into a terminating error. The installer's CI now explicitly asserts that no
native stderr is captured that way.

---

## 5. Non-functional requirements

### NFR-1 Performance

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-1.1 | Cull 1,000 RAW files in under 5 minutes | **Not met** — see below |
| NFR-1.2 | Preview scrolling smooth on integrated graphics | **Done** — LRU cache, embedded RAW thumbnails, off-thread decode |
| NFR-1.3 | Analysis independent of source resolution | **Done** — all analysers downscale first |

**NFR-1.1 is not met, and this is stated rather than finessed.** Measured:

| Workload | Per frame | Extrapolated to 1,000 |
| --- | --- | --- |
| Blink detection, 512 px JPEG | ~19 ms | ~0.3 min |
| Blink detection, 36 MP JPEG | ~341 ms | **~5.7 min** |

The cost is dominated by **decoding the file**, not by the detector. Culling
without eye checking is far quicker. Critically, **these numbers are from
JPEGs — the RAW case has never been benchmarked and RAW decoding is slower
still.** Treat the headline requirement as unmet and the RAW figure as
unmeasured.

Analysis working sizes (resolution independence, NFR-1.3): blur 1024 px,
pose 1280 px, orientation 1400 px, cull previews 1200 px, face 1600 px.

### NFR-2 Resource budgeting

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-2.1 | Batch operations must never starve the OS | **Done** — export budget = half *currently available* RAM, capped at 8 GB, floor 0.5 GB, recomputed at export time rather than cached |
| NFR-2.2 | Preview cache must be bounded | **Done** — thread-safe LRU, 256 images |
| NFR-2.3 | Video pipelines must leave CPU headroom | **Done** — thread pool sized from detected cores |

### NFR-3 Hardware adaptation

Detects and uses, in order of preference: Nvidia CUDA → Apple Silicon
CoreML/MPS → Intel OpenVINO → CPU. **CPU fallback must always work.** Hardware
summary logged at startup for remote diagnosis.

- **Minimum:** 8 GB RAM, dual-core (i5-8350U with integrated graphics is the
  development target).
- **Recommended:** 32 GB RAM, 6+ cores, dedicated GPU.
- **Storage:** ~2 GB plus the user's library.

### NFR-4 Reliability and failure behaviour

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-4.1 | A missing model disables its feature, never raises | **Done** (INV-3) |
| NFR-4.2 | A corrupt sidecar degrades, never raises | **Done** |
| NFR-4.3 | UI never freezes during long work | **Done** — background workers; concurrent-task guard |
| NFR-4.4 | Unreadable image reports which file and why | **Done** — per-path error carried on the report object |
| NFR-4.5 | Levelling must not stall a batch on a model download | **Done** — `default_pose_analyzer()` never fetches; consent is asked up front |

### NFR-5 Privacy and security

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-5.1 | No user data leaves the machine (desktop) | **Done** (INV-2) |
| NFR-5.2 | Models content-addressed and pinned | **Done** (INV-5) |
| NFR-5.3 | Model download requires consent when user-visible | **Done** — pose models (~17 MB) prompt with size and what is and is not sent |
| NFR-5.4 | No credential handling anywhere | **Done** — the app has no accounts, no login, no payment path |
| NFR-5.5 | Uninstall never touches user data | **Done** |

### NFR-6 Supportability

Install transcript at `%TEMP%\GraniteStudios-install.log`; full dependency
detail at `%TEMP%\GraniteStudios-pip.log`. Hardware diagnostics to stdout at
startup. On install failure the log path is printed, opened, and the window
held open so the error cannot scroll away.

### NFR-7 Interface constraints

Dark theme; splash screen covering heavy imports (`rawpy`, `cv2`,
`onnxruntime`); custom AppUserModelID so the taskbar shows the app icon rather
than the generic Python logo; four-item navigation (Photos, Reel Maker, Web
Export, Video Editor) per INV-6.

---

## 6. Architecture

### 6.1 Layers

```
┌─────────────────────────────────────────────────────────────┐
│ Presentation                                                │
│   src/ui/main_window.py        PyQt6, QStackedWidget nav     │
│   web/frontend/                React + Vite + Tailwind       │
├─────────────────────────────────────────────────────────────┤
│ Service                                                     │
│   src/video/api.py             optional local REST          │
│   web/backend/app.py           hosted REST (same engines)    │
├─────────────────────────────────────────────────────────────┤
│ Domain                                                      │
│   src/ai_tools/                cull, blur, orientation,      │
│                                pose, face, colour, bg, xmp   │
│   src/video/                   11 FFmpeg/OpenCV modules      │
│   src/social/, src/web/        reel, IG, gallery             │
├─────────────────────────────────────────────────────────────┤
│ Platform                                                    │
│   src/engine/                  loader, hardware, resources,  │
│                                shortcut                      │
│   src/ai_tools/model_store.py  pinned model acquisition      │
└─────────────────────────────────────────────────────────────┘
```

**The domain layer has no UI dependency.** This is what allows the hosted
browser build to import `src/` unchanged (FR-13.1) and the video REST API to
drive the same code the desktop buttons do.

### 6.2 Module responsibilities

| Module | Responsibility | Lines |
| --- | --- | --- |
| `ui/main_window.py` | All UI, handlers, thread orchestration | 2,660 |
| `ai_tools/pose_analysis.py` | BlazePose two-stage pipeline, posture | 729 |
| `ai_tools/culling.py` | Sharpness, bursts, hashing, target cull | 438 |
| `ai_tools/orientation.py` | Coarse turn, scene lines, levelling | 405 |
| `engine/universal_loader.py` | Format support, EXIF, `-edit` folders | 402 |
| `ai_tools/face_analysis.py` | YuNet detection, eye openness | 400 |
| `web/publisher.py` | Static gallery generation | 282 |
| `video/spatial.py` | Aspect, crop, auto-crop, stabilisation | 279 |
| `ai_tools/blur_analysis.py` | Blur classification and rescue gate | 270 |
| `ai_tools/color_engine.py` | Selective colour, grades, auto-fix | 250 |
| `ai_tools/xmp_sidecar.py` | Sidecar read/merge/write | 226 |
| `ai_tools/model_store.py` | Pinned download, verification, cache | 180 |

### 6.3 The `-edit` folder contract

```
Vacation2024/                 # originals — never modified or moved
├── DSC_0001.NEF
├── DSC_0002.NEF
└── Vacation2024-edit/        # created on load
    ├── DSC_0001_graded.jpg      # grading
    ├── DSC_0001_level.jpg       # auto-level
    ├── DSC_0001_sharpened.jpg   # blur rescue
    ├── DSC_0002_nobg.png        # background removal
    └── DSC_0001.jpg             # RAW keeper flatten
```

Video follows the same rule per file: `clip.mp4` → `clip-edit/`, each operation
writing a new file rather than overwriting — which is what makes operations
chain and **Reset to Original** possible.

**Exception.** The three *publishing* actions — panoramic reel, Instagram
export, Generate Gallery — ask where to save instead, because those outputs
usually belong elsewhere. The gallery still *reads* exclusively from the edit
folder.

### 6.4 Model dependencies

| Model | Purpose | Size | Bundled? | Failure mode |
| --- | --- | --- | --- | --- |
| YuNet (`face_detection_yunet_2023mar.onnx`) | Face detection, blink, coarse orientation | 232,589 B | No — fetched once | Culling degrades to sharpness-only |
| BlazePose person detector | Pose stage 1 | 11,990,159 B | No — consent-gated | Body engine unavailable; lines still work |
| BlazePose landmarks | Pose stage 2 (33 points) | 5,557,238 B | No — consent-gated | As above |
| U2Net (via `rembg`) | Background removal | 175,997,641 B | Pre-fetched by installer | Feature unavailable |

All OpenCV-zoo models are pinned by SHA-256 and byte size from the upstream
git-lfs pointer, fetched via the lfs batch API with a media-host fallback.

**Note.** `_person_anchors()` *generates* the detector's 2,254 SSD anchors
(three feature maps: 28×28×2, 14×14×2, 7×7×6) rather than embedding the
upstream 2,200-line literal. Verified element-wise identical to that table.

### 6.5 Threading

- **Thumbnail/metadata decode** — dedicated cancellable worker; UI never blocks
  on folder load.
- **Long tasks** (cull, blur scan, grading, levelling, export) — `TaskWorker`
  background thread, with an explicit guard and dialog if a second is started.
- **Video pipelines** — `JobManager` thread pool sized from detected cores.
- **Preview cache** — thread-safe LRU bounding RAM during grid scrolling.

---

## 7. Quality and verification

### 7.1 Current approach

**There is no automated test suite. This is the single largest quality gap in
the project.**

Verification today is performed by exercising the real engines against
generated and reference media, and by driving the actual PyQt6 window
headlessly with `QT_QPA_PLATFORM=offscreen` so that every button's real handler
runs — not a mock of it. Results are recorded in `README.md`.

This catches more than a naive unit suite would (it has caught real
integration defects: a version-skew installer crash, the blank-padding
misreading, an inverted sign convention) but it is **not repeatable by anyone
else, not run in CI, and not a regression barrier.**

### 7.2 Coverage

| Area | Verified |
| --- | --- |
| `hardware.py` | RAM profiling; export budget = half available, capped at 8 GB |
| `universal_loader.py` | Folder scan; RAW/HEIC/JPEG/PNG; preview + array decode; EXIF; `-edit` creation |
| `culling.py` | Sharpness ordering; perceptual-hash duplicates; burst grouping |
| `cull_to_target` | Quota spread across bursts; sharpest-first; clamps above image count; rejects negative target |
| `blur_analysis.py` | 3 sharp images not flagged; defocus at 2 radii each; **15/15 motion with direction within 15°**; strictness changes verdicts; rescue gate refuses unrecoverable frames |
| `orientation.py` | All 90/180/270 recovered (also with the face engine disabled); tilts round-trip under 1°; blank padding ignored; crop leaves no black corners |
| `pose_analysis.py` | Landmarks land on the body; **tilt tracks roll 1:1 across ±24°**; 0.53°/1.33° residual full-body; 2.90°/9.54° shoulders-only and therefore never applied; half-out-of-frame body refused; deterministic; degrades to lines-only |
| `model_store.py` | Pinned digests reject substituted/truncated downloads; failed cache refetched; missing model disables feature |
| `face_analysis.py` | Checksum-pinned download; 2 real faces, **0 false positives across 6 non-face images**; stable 0.4×–2× scale; grayscale/RGBA/empty/`None` inputs; landmarks mapped back after downscale; degrades when absent |
| blink-aware culling | Verified on bursts where sharpness and open eyes disagree, both directions |
| `color_engine.py` | All 6 grades; all 8 hues; auto-fix; adjustments |
| `xmp_sidecar.py` | Ratings 0–5 and −1 round-trip; all 5 labels; clamping; **merge preserves foreign `crs:`, `dc:title`, keywords**; corrupt sidecar degrades; batch write over a real cull |
| `background_removal.py` | RGBA at source size with non-uniform alpha |
| `ig_sync.py` / `reel_maker.py` | Exact 1080×1350 and 1080×1920; reel rejects <3 or >7 images |
| `publisher.py` | Category tabs; custom title; responsive masonry; images emitted |
| `create_shortcut.py` | `.lnk` argument quoted; targets `pythonw.exe`; launches from a path containing a space |
| `video/` | Every timing/spatial/structure/audio/colour op; all 4 presets; `.cube` LUT; `JobManager` pipelines **including the failure path** |
| `video/api.py` | All 6 routes plus 400/404 handling |
| `ui/main_window.py` | Construction; folder load; video load; all 10 video ops chained through real handlers; Select Top N; Skip blinks; Write XMP Sidecars; auto-level; blur scan |

### 7.3 Required improvement

**QA-1 — Introduce a `pytest` suite.** Priority order: (1) INV-1 regression
tests asserting source-directory hashes are unchanged after every destructive-
looking operation; (2) the measured accuracy figures above, as assertions with
tolerances, so a model or OpenCV upgrade cannot silently regress them; (3)
the degradation paths in INV-3. External contributions welcome.

---

## 8. Known limitations

Each entry records the measurement that established it, so none of these is
retried blind.

### 8.1 Head roll cannot be levelled from the face model

Two approaches were built, measured, and removed:

- **Eye-line levelling.** Rotating a portrait through **24°** moved the
  measured eye line by about **5°**. YuNet regresses landmarks toward an
  upright configuration regardless of head tilt.
- **Confidence sweep.** Searching for the rotation of peak detection
  confidence failed for the same root cause: the detector is deliberately
  tolerant of in-plane rotation, so confidence stayed **flat across the entire
  ±20° sweep**.

The pose model behaves identically when swept — **confidence 1.00 at all four
quarter turns**. That is exactly why FR-4.2 reads landmark *geometry* rather
than hunting a confidence peak. A landmark moves with the body; a score does
not.

### 8.2 Blink detection is a heuristic

YuNet returns five landmarks, two of which are eye *centres* with no contour,
so the standard Eye Aspect Ratio (six points per eye) **cannot be computed**.
Instead the vertical extent of the dark region in an eye patch is measured —
an open eye shows the iris as a tall blob, a closed lid only a thin lash line.

- `EYE_OPEN_THRESHOLD = 0.42` was placed using **two real faces** plus
  synthetically closed versions. Open eyes measured 0.52–0.78, closed 0.15–0.23
  — a wide gap, but **two faces is not a validated dataset.**
- Sunglasses, heavy shadow, strong side-lighting, motion blur across the eye,
  and faces turned far from camera are **untested**.
- A raw `min_eye_openness` score is exposed on every frame so borderline cases
  can be judged directly rather than trusting the boolean.

Treat it as a strong first pass worth reviewing, not an oracle.

### 8.3 Shoulders-only levelling is not applied

2.90° mean / **9.54° worst** residual, against 0.53° / 1.33° with hips visible.
A ten-degree error is worse than the tilt most people would have left alone.
Reported, never auto-applied.

### 8.4 No generative fill

Rotation leaves wedge-shaped corners. The two honest options are to crop them
away or leave them black — both offered. Filling them requires an inpainting
model that is not bundled.

### 8.5 RAW culling performance is unmeasured

See [NFR-1.1](#nfr-1-performance). All timing figures come from JPEGs.

### 8.6 No automated test suite

See [7.3](#73-required-improvement).

### 8.7 Single-subject validation on pose

Pose accuracy figures come from a small number of reference images, principally
one full-body subject. The behaviour is consistent and deterministic, but the
sample is not a dataset.

---

## 9. Target state

Forward requirements. Each names its **actual blocker**, so none reads as a
promise the code can nearly keep.

### 9.1 Culling

| Requirement | Blocker |
| --- | --- |
| Identify the same person across a shoot | Needs recognition *embeddings* (e.g. SFace) on top of existing detection. Detection ≠ recognition. |
| Loupe / Survey / spray-can review UI | A substantial UI build. Review today is the grid plus the cull report. |
| EXIF-driven smart grouping (camera + lens) | Small — EXIF is already parsed. |
| Keyboard shortcuts for rapid culling | Small, high value for the primary persona. |

### 9.2 Editing and retouching

| Requirement | Blocker |
| --- | --- |
| Skin smoothing, blemish and wrinkle reduction | **Most tractable next item.** Face box and pose both exist; classical frequency-separation retouching targeted at the face region needs no new model. |
| Editable subject masks | The mask exists internally; needs UI and an edit model. |
| Background replacement | Cutout exists; compositing does not. |
| Teeth whitening, glare removal, stray-hair removal, clothing de-wrinkling | Each needs its own trained segmentation model. |
| Generative corner fill | Needs an inpainting model (e.g. LaMa). |
| AI profiles trained from a photographer's Lightroom catalogue | **The single largest item in the spec.** This is a training pipeline and a model architecture, not a feature flag. |
| Style marketplace | Server infrastructure plus curated content. |
| Batch watermarking | Small. |

### 9.3 Client delivery

| Requirement | Blocker |
| --- | --- |
| Client proofing, favourites, face search, print ordering, 4K downloads | All require a hosted backend, user accounts and payments — a different product shape from a local-first desktop app, and in tension with INV-2. Should be treated as a deliberate strategic decision, not a backlog item. |

### 9.4 Platform

| Requirement | Blocker |
| --- | --- |
| Deploy the hosted browser build | See [12.4](#124-the-hosted-build-is-not-deployed). |
| Code signing | Certificate cost; currently triggers SmartScreen. |
| Direct Instagram upload | Meta API access and app review. |

---

## 10. Release and distribution

### 10.1 Windows (primary)

One file: `Install-GraniteStudios.bat`, served from the latest GitHub release.
It downloads the application itself, so there is nothing to clone.

**Flow:** locate Python 3.10+ (launcher → registry → common roots, with a
quote-free probe that survives the Microsoft Store stub) → copy to
`%LOCALAPPDATA%\Granite Studios` → build a private venv → install dependencies
with a separate pip log → pre-download the background-removal model → create
Desktop and Start Menu shortcuts → register in Settings → Apps → **start the
app headlessly to confirm the install actually works**.

**Update:** re-run the same file. It is republished by GitHub Actions on every
push to `main` and fetches `main` at run time, so it cannot go stale.

**Why install rather than run in place:** a checkout often lives in a synced
folder (Google Drive, OneDrive, Dropbox). Those paths vanish when sync pauses,
are slow to read, and break `pip`.

### 10.2 macOS

PyInstaller `.app` → `.dmg` with drag-to-Applications, built by CI, distributed
via GitHub Releases. Gatekeeper bypass required on first launch (right-click →
Open) as the app is unsigned.

### 10.3 Platform matrix

| Platform | Status |
| --- | --- |
| Windows 10/11 | **Primary** — one-file installer, self-updating |
| macOS | **Supported** — `.dmg`, unsigned |
| Linux | **Source only** — no packaging |
| Browser | **Built, not deployed** — see [12.4](#124-the-hosted-build-is-not-deployed) |

---

## 11. Success metrics

> **Proposed and unvalidated.** No metrics are currently collected. The
> application has no telemetry, which is a deliberate consequence of INV-2 —
> any future measurement programme must be opt-in and local, or it contradicts
> the product's central claim.

| Metric | Proposed target | Rationale |
| --- | --- | --- |
| Time to cull 1,000 RAW | < 5 min | Inherited from the reference spec. **Currently unmet** (NFR-1.1). |
| Cull agreement with the photographer | ≥ 80% of AI picks retained | Would require the observational study in [2.4](#24-target-users). |
| Levelling corrections accepted | ≥ 90% kept | Measurable once the review drawer records restores. |
| Install success rate, first attempt | ≥ 95% | Inferable from install logs if users volunteer them. |
| Originals lost or modified | **Zero, permanently** | INV-1. This is a correctness invariant, not a target to trade against. |

---

## 12. Risks and open questions

### 12.1 Upstream model availability

Three of four models are fetched from `opencv/opencv_zoo` at run time. If that
repository moves, re-exports, or changes its lfs hosting, affected features
stop working for new installs. Pinned digests mean a *changed* file is rejected
rather than silently substituted — which is correct, but means the feature
breaks rather than degrading quietly.

**Mitigation to consider:** a secondary mirror, or bundling YuNet (232 KB,
small enough to ship).

### 12.2 OpenCV 5 API churn

This has already caused two defects:

- `cv2.CascadeClassifier` is **absent entirely** from the OpenCV 5 headless
  build, which is why face detection uses YuNet.
- `cv2.HoughLinesP` returns `(N, 4)` in OpenCV 5 where OpenCV 4 returned
  `(N, 1, 4)`, causing an `IndexError`. Both shapes are now handled.

A pinned OpenCV version would reduce exposure; the project currently tracks
whatever `opencv-python-headless` resolves to.

### 12.3 No code signing

Windows SmartScreen and macOS Gatekeeper both warn on first run. For a tool
asking photographers to trust it with client work, this is a credibility cost,
not merely an inconvenience.

### 12.4 The hosted build is not deployed

`web/` is built and the backend imports `src/` unchanged, but the Vercel
project does not currently exist — `granite-studios-web.vercel.app` returns
404, so the tool card on the portfolio site 404s. Additionally, the
`granite-studios` rewrite in the portfolio's `vercel.json` likely needs the
same prefix-preserving fix applied in commit `ae9e123` for IndicationScope.
**Open question: deploy, or remove the portfolio link until it is deployed?**

### 12.5 Single maintainer

One author, no test suite, no second reviewer. The verification discipline is
real but lives in one person's practice rather than in CI.

### 12.6 Open product questions

1. Should client delivery (9.3) be pursued at all, given it contradicts INV-2?
2. Is the primary persona correct? Nothing validates it ([2.4](#24-target-users)).
3. Should the app pin OpenCV, trading security updates for stability?
4. Is bundling models (larger download, no runtime dependency) preferable to
   fetching them?

---

## 13. Appendix

### 13.1 Key constants

| Constant | Value | Module | Meaning |
| --- | --- | --- | --- |
| `EYE_OPEN_THRESHOLD` | 0.42 | `face_analysis` | Below this an eye reads as closed |
| `MOTION_DIRECTIONALITY` | 0.42 | `blur_analysis` | Above this a soft frame is a smear, not defocus |
| `STRICTNESS.balanced` | blurry < 95, soft < 230, rescue floor 30 | `blur_analysis` | Default blur tier |
| `STRICTNESS.aggressive` | blurry < 180, soft < 420, rescue floor 45 | `blur_analysis` | Reclaim storage |
| `STRICTNESS.conservative` | blurry < 42, soft < 110, rescue floor 18 | `blur_analysis` | Only clearly unusable |
| `MIN_CORRECTION_DEG` | 0.75 | `orientation` | Below this, not worth re-encoding |
| `MAX_FINE_CORRECTION_DEG` | 20.0 | `orientation` | Above this, presumed misread |
| `MAX_ASYMMETRY_DEG` | 12.0 | `pose_analysis` | Shoulder/hip agreement gate |
| `MIN_APPLY_CONFIDENCE` | 0.55 | `pose_analysis` | Floor to auto-level from the body |
| `MIN_VISIBILITY` | 0.55 | `pose_analysis` | Landmark counted as seen |
| `MAX_BODY_TILT_DEG` | 30.0 | `pose_analysis` | Above this, presumed misdetection |
| `time_threshold` | 2.0 s | `culling` | Burst grouping window |
| `hash_threshold` | 12 | `culling` | Hamming distance for near-duplicates |
| `top_count` | 2 | `culling` | Auto-starred per burst |
| export budget | ½ available RAM, max 8 GB, min 0.5 GB | `hardware` | Batch ceiling |
| preview cache | 256 images | `ui` | LRU bound |

### 13.2 Glossary

| Term | Meaning |
| --- | --- |
| **Burst** | Frames captured within `time_threshold` of each other |
| **Cull** | Reduce a shoot to keepers |
| **Reject** | Exclude from the edit folder. **Never a delete** (INV-1) |
| **Edit folder** | `<Folder>-edit/`, created on load; all outputs land here |
| **Coarse / fine** | Quarter-turn orientation vs sub-degree levelling |
| **Body tilt** | Camera roll inferred from shoulder and hip lines |
| **Spine lean** | Subject's own lean. Reported, never corrected |
| **Rescue** | Unsharp-mask offered only where detail remains |
| **Sidecar** | `.xmp` file carrying ratings/labels beside an image |

### 13.3 Dependencies

PyQt6 · rawpy · Pillow / pillow-heif · opencv-python-headless · numpy ·
moviepy · onnxruntime · rembg · imageio-ffmpeg · fastapi / uvicorn / pydantic ·
psutil · pyshortcuts

### 13.4 Related documents

- [`README.md`](README.md) — user guide, install, feature reference
- [`README_INSTALL.md`](README_INSTALL.md) — SmartScreen and Gatekeeper notes
- [`web/README.md`](web/README.md) — browser build architecture and deployment

---

*Granite Studios — your RAW, edited locally. Never in the cloud.*
