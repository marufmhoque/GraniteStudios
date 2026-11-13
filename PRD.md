# Granite Studios — Product Requirements Document

| | |
| --- | --- |
| **Version** | 2.0 |
| **Date** | 13 November 2026 |
| **Status** | Living document — tracks the `v2` branch until it merges to `main` |
| **Applies to** | commit `db1e4cf` and later |
| **Previous version** | [1.0](https://github.com/marufmhoque/GraniteStudios/blob/12aa977/PRD.md), commit `12aa977` |
| **Owner** | Maruf Hoque |
| **Repository** | https://github.com/marufmhoque/GraniteStudios |

---

## How to read this document

`README.md` is the **user guide**: how to install the app and what the
controls do. This document is the **specification**: who the product is for,
what it must never do, what each capability has to satisfy to count as
working, how well it actually works, and what it should become.

### The evidence rule

A product requirements document normally carries market sizing, user research
and competitive analysis. **None of that exists for this project, and this
document does not invent it.**

Three conventions make that explicit, and they hold throughout:

- **Measured** — a number that traces to a test in `tests/`, a recorded run
  in `tests/bench_cull.py`, or a module docstring. Cited with its conditions
  and sample size.
- **Assumption** — a belief about users, market or value that has *not* been
  validated. Flagged inline. Sections [2](#2-users-and-positioning) and
  [11](#11-success-metrics) are assumption-heavy by nature and say so.
- **Unmeasured** — a thing we believe works but have not tested against real
  data. Named rather than quietly counted as working.

Where a conventional PRD would cite data we do not have, this one says so.
Several features exist in their current shape *because* a measurement
contradicted the obvious design; those measurements are recorded next to the
requirement they shaped.

### Status vocabulary

| Term | Meaning |
| --- | --- |
| **Done** | Implemented and verified by tests. Evidence cited. |
| **Partial** | Implemented in part, or implemented but unmeasured. The gap is stated. |
| **Not built** | Not implemented. The actual blocker is named. |

### What changed in 2.0

Version 2.0 turns a cull-and-grade utility into a suite with five editing
workspaces. In summary:

| Area | 1.0 | 2.0 |
| --- | --- | --- |
| Interface | One window, four pages | Six workspaces over one shared session ([FR-15](#fr-15-workspaces-and-keyboard)) |
| Editing model | Each action wrote a new JPEG | Edits are recipes; nothing rendered until export ([INV-7](#inv-7--edits-are-settings-until-export)) |
| Cull speed (1,000 RAW) | Unmeasured on RAW; ~5.7 min extrapolated from JPEG | **42–104 s** on nine cameras ([NFR-1.1](#nfr-1-performance)) |
| Blink detection | Eye-patch darkness heuristic | Eye aspect ratio from a 478-point mesh ([FR-2](#fr-2-culling)) |
| Develop | Six fixed grades | Float parametric develop engine ([FR-7](#fr-7-develop-engine)) |
| Retouching | None | Layers, masks, heal, clone, skin, teeth, eyes, inpainting ([FR-16](#fr-16-layers-masks-and-retouching)) |
| Graphics | None | Vector text, shapes, watermarks; SVG and PDF ([FR-17](#fr-17-design-and-vector-graphics)) |
| Tests | None | **493** pytest tests ([7](#7-quality-and-verification)) |

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
the misfires, pick one frame from each burst, discard the blinks, straighten
what was shot handheld. Then the keepers need developing, some need
retouching, and many need a watermark before delivery. Today that spans three
or four applications, at least one of which uploads the photographs and most
of which are rented rather than owned.

*(Assumption — that photographers value local processing and ownership is a
design belief, not a validated finding. See [2.4](#24-target-users).)*

### 1.2 Vision

**Granite Studios is a local-first photo suite that culls, develops,
retouches and designs without the photographs ever leaving the machine.**

The commitments that follow are enforced in code — see
[3. Product invariants](#3-product-invariants):

- Nothing is uploaded. The only outbound traffic in the desktop application
  is downloading AI models, each verified against a pinned digest, each
  (beyond the small face models) asked for before it is fetched.
- Original files are never modified, moved or deleted.
- Edits are settings, not pixels, until the user exports.
- It runs on modest hardware. The reference machine is an **i5-8350U** laptop
  with integrated graphics; every timing in this document was taken on it.

### 1.3 Scope

| In scope | Out of scope |
| --- | --- |
| Ingestion of camera RAW, phone formats, HEIC and standard images | A catalogue database across folders |
| Culling, blur triage, levelling, grouping by person | Cloud sync, accounts, multi-device state |
| Parametric develop, layers, masks, retouching | Pixel painting and compositing of multiple photographs |
| Vector text, shapes and watermarks | General illustration (artboards, symbols, typesetting long text) |
| Export presets, social formats, static web gallery | Hosted client proofing, print ordering, payments |
| A complete FFmpeg-backed video editor | Non-linear multi-track timeline editing |
| Self-updating Windows installation | App-store distribution; code signing |

---

## 2. Users and positioning

> **This entire section is assumption, not research.** No user interviews,
> surveys or market analysis have been conducted.

### 2.1 Primary persona (assumed)

**The high-volume event photographer.** Weddings, graduations, corporate
events; 2,000–5,000 RAW frames per job. The pain is the hours of culling
before editing starts, then repetitive retouching and watermarking before
delivery. Served by the Cull workspace's speed and keyboard, the keeper
score, grouping by person, Sync, and batch export with watermarks.

### 2.2 Secondary persona (assumed)

**The content creator.** Stills and short video for social. Needs 4:5 and
9:16 formats, quick grading and captions. Served by Develop looks, Design
text and the Instagram, reel and video tools.

### 2.3 Tertiary persona (assumed)

**The hobbyist on modest hardware.** A laptop with integrated graphics and
8–16 GB of RAM. Served by the decision never to require a GPU, the proxy
preview renders, strip-wise export and memory-capped worker counts.

### 2.4 Target users

These personas are **unvalidated**; the product is used by its author. The
most valuable research would be watching one high-volume photographer cull
and deliver a real job. The keeper-score weights, the strictness tiers and
the target-count cull all encode guesses that have never been checked against
anyone.

### 2.5 Positioning

Design intent, not competitive research. No benchmarking against these
products has been performed.

| Product | Relationship |
| --- | --- |
| Aftershoot | Same culling problem, opposite architecture: local and owned rather than cloud and subscribed. Granite Studios has no trained style profiles. |
| Lightroom Classic | Overlaps now on develop. Granite Studios still hands off via merging XMP sidecars, and its recipes are its own: the same numbers do not produce Lightroom's rendering. |
| Photoshop | Overlaps on layers, masks, healing and portrait retouching for single photographs. No multi-image compositing, no pixel painting. |
| Illustrator | Overlaps only on watermarks and captions over a photograph. |
| Photo Mechanic | Overlaps on fast keyboard culling and sidecars. Granite Studios adds automated judgement. |

### 2.6 Non-goals

- **Not a DAM.** The filesystem is the source of truth; state lives in each
  folder's edit subfolder.
- **Not a cloud service.** The hosted browser build exists for evaluation.
- **Not a Lightroom clone.** The develop engine is its own rendering.
- **Not a destructive tool.** No code path deletes a user's photograph.

---

## 3. Product invariants

Rules the software must never break. They outrank feature requests. Each
names its enforcement and its test.

### INV-1 — Originals are never modified, moved or deleted

"Reject" means **left out of every export**. It does not mean delete.

- **Enforcement:** the source folder is only ever opened for reading. All
  writes go to `<Folder>-edit/` or a destination the user chooses. Rejected
  frames are dropped by `Workspace._drop_rejected()` /
  `LibrarySession.exportable()` before every export; `develop_pipeline.export()`
  and `export_presets.export_batch()` refuse a destination equal to the source.
- **Verification:** `tests/test_invariants.py` hashes every original before
  and after every action that reads the library — cull, cull to target, reject
  and restore, blur triage and sharpen, grading, levelling, XMP sidecars,
  background removal, Instagram, reel, gallery, file export, video — and fails
  on any change. Two tests check that the hashing itself catches a modified
  and a removed file. Separate tests check that a rejected frame is not
  graded, levelled, exported to Instagram, published, or included in a reel.

### INV-2 — Images never leave the machine

The desktop application sends **no user data** anywhere.

- **Enforcement:** `src/ai_tools/model_store.py` is the only network client
  in the photo pipeline, and it only downloads.
- **Face data:** identity embeddings for grouping by person are biometric
  data. They are computed only from the open folder, stored only in its edit
  subfolder, never compared against anything outside it, and deleted by
  **Forget Faces**.
- **Note:** the hosted browser build ([FR-13](#fr-13-hosted-browser-version))
  uploads to its own backend by necessity. It is a different deployment with a
  different privacy posture and must always be described as such.

### INV-3 — Degrade, never guess

A missing, corrupt or unfetchable model disables the feature that needs it. A
damaged settings file opens the photo unedited. Neither raises.

- **Enforcement:** `ensure_model()` returns `None`; every model wrapper has
  `available`; masks return an empty mask with a `note`; layers that cannot
  render are skipped with a reason; recipes and session state load leniently.
- **Verification:** `test_model_store.py`, the `*_without_the_model_*` tests in
  `test_face_mesh_ear.py`, `test_person_clustering.py` and `test_retouch.py`,
  and the damaged-file tests in `test_parametric_recipe.py`,
  `test_layer_stack.py` and `test_workspaces.py`.

### INV-4 — Never claim unmeasured accuracy

Where a measurement shows a signal is not good enough, the feature reports it
and declines to act, rather than acting with a caveat in the documentation.

- Shoulders-only body levelling: 2.90° mean / 9.54° worst residual. Reported,
  never applied (`MIN_APPLY_CONFIDENCE = 0.55`).
- The 468-point face mesh: its eye aspect ratio did not separate open from
  closed eyes, so it was replaced, not tuned around ([FR-2](#fr-2-culling)).
- The mesh's trained "blink" coefficient: worse than the plain ratio, so not
  used for blinks.
- Hair and sky masks: offered, labelled approximate in the interface.

### INV-5 — Downloaded models are content-addressed

Every model is pinned by SHA-256 and byte size. A download that does not match
is rejected; a cached file that fails verification is deleted and refetched.
Models from outside OpenCV's zoo are fetched from exactly one named URL and
checked identically.

- **Enforcement:** `_matches()` / `verify()` in `model_store.py`.
- **Verification:** `test_model_store.py` (substituted download rejected,
  tampered cache removed, all-or-nothing multi-model fetch).

### INV-6 — One working set, every workspace

*(Rewritten in 2.0. The 1.0 rule was "one pane, not three".)*

Selection, ratings, colour labels, picks and rejections belong to the open
folder, not to a workspace. Moving between workspaces never loses them, and
the editing workspaces show the folder as a filmstrip so the working set stays
in view.

- **Enforcement:** `src/engine/session.py` (`LibrarySession`), shared by every
  workspace through `Workspace` properties; `Filmstrip` in Develop, Retouch and
  Design.
- **Verification:** `test_six_workspaces_share_one_selection`,
  `test_edits_and_marks_survive_reopening_the_folder`.

### INV-7 — Edits are settings until export

*(New in 2.0.)* Moving a slider, painting a mask, healing a spot or placing a
watermark changes the photograph's **recipe**. No image is written until the
user exports, and the source is only ever read.

- **Enforcement:** `parametric_recipe.save()` is the only writer of edit
  state and only writes `<Folder>-edit/<filename>.granite.json`. Rendering
  (`develop_pipeline.render`) is a pure function of pixels and recipe.
  Heal and clone are stored as positions, not pixels.
- **Verification:** `test_editing_and_previewing_write_nothing` (every kind of
  edit, plus previews, leaves the whole folder tree byte-identical),
  `test_a_slider_changes_the_preview_and_nothing_is_written_until_saved`, and
  the grade and level tests in `test_invariants.py`, which assert that a
  sidecar is written and no image is.
- **Exceptions, stated:** background removal writes `*_nobg.png` (it is an
  export); the video editor still chains files ([8.6](#86-video-editing-is-file-chained)).

---

## 4. Functional requirements

### FR-1 Ingestion

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-1.1 | Open camera RAW with no conversion | `.nef .cr2 .cr3 .arw .dng .raf .orf .rw2 .pef .3fr` — preview, metadata and develop | **Done** |
| FR-1.2 | Fast previews from RAW | Embedded JPEG via LibRaw; a scan of the file for JPEG streams when LibRaw cannot open it; half-size development when no usable preview exists. The path taken is recorded on every `Preview` | **Done** |
| FR-1.3 | Read capture time without developing | From the embedded preview's EXIF, else the RAW container's own tags, else file time | **Done** |
| FR-1.4 | Phone stills | HEIC at its own bit depth (10-bit round-trips); Ultra HDR gain maps read and applied per the published formula; Motion Photo clips located (container directory, legacy offset, or structural search) and extracted; Live Photo stills paired with clips by name | **Partial** — tested on files built to the published layouts; **no real phone file has been tested** ([8.4](#84-phone-formats-are-unverified-on-real-files)) |
| FR-1.5 | ProRAW | `ProfileGainTableMap` detected and reported | **Partial** — the table is not applied ([8.4](#84-phone-formats-are-unverified-on-real-files)) |
| FR-1.6 | Video files | MP4, MOV, MKV, WebM, M4V, MXF, AVI; codec named from the container's own tag (ProRes 422 / HQ / 4444, DNxHD, H.264, HEVC, VP9, AV1) | **Done** — ProRes, DNxHD, H.264 and VP9 encoded and recognised in tests; the HEVC and AV1 names are untested |
| FR-1.7 | Rich metadata | Camera, lens, ISO, aperture, shutter, focal length (and 35 mm equivalent), body and lens serial, capture time and UTC offset, ICC profile name | **Done** |
| FR-1.8 | Create the edit subfolder on load | `Vacation2024/Vacation2024-edit/`; idempotent | **Done** |

**Modules:** `engine/fast_preview.py`, `engine/universal_loader.py`,
`engine/phone_media.py`, `video/ffmpeg_utils.py`.

**Measured** (eleven CC0 samples from raw.pixls.us, `tests/fetch_media.py`):
locating the embedded preview takes **1–10 ms**; asserted under 25 ms per
format in `test_locating_the_embedded_preview_is_fast`. Decoding it is the
real cost (30–120 ms at full size). Two of eleven files carry no usable
preview (iPhone XS DNG: 852×640; Hasselblad H3D: 320×240 bitmap) and are
developed at half size instead.

**Design note (FR-1.2).** Reduced-size JPEG decoding is a low-pass filter. On
five cameras it cut the sharpness score of a *sharp* frame to **0.4–0.7×** of
the full-decode score while barely moving a blurred one, which would have
made every blur threshold flag good photographs. Sharpness therefore uses a
full **greyscale** decode (same score to 0.07%, half the time, a third of the
memory) and faces use a reduced colour decode of the same bytes.

### FR-2 Culling

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-2.1 | Decode each frame once | Sharpness, hash, capture time and every face measured from one located preview, on a thread pool | **Done** |
| FR-2.2 | Score focus where it matters | Laplacian variance inside the eye region of each main face (mesh) or the face box (fallback), at a common size; whole-frame when there are no faces | **Done** |
| FR-2.3 | Blink detection by eye aspect ratio | `EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖)` per eye from a 478-point mesh; a face is closed when its **more open** eye is below `EAR_CLOSED_BELOW = 0.135` | **Done** — see measurements |
| FR-2.4 | Smile and open-mouth detection | Mean of the mesh expression network's `mouthSmileLeft/Right`; open mouth from `jawOpen ≥ 0.30` | **Done** |
| FR-2.5 | Keeper score | `0.40·face sharpness + 0.35·eyes + 0.15·smile + 0.10·composition`, each term on 0–1; sharpness scaled to the batch's 95th percentile; a term that could not be measured is left out and the rest re-weighted | **Done** |
| FR-2.6 | Composition term | A stated heuristic: penalises a face cut by the frame edge, touching an edge, or without headroom | **Done** — heuristic by design |
| FR-2.7 | Bursts, duplicates, target cull | Capture-time bursts (2 s); dHash duplicates (12 bits); quota spread round-robin across bursts | **Done** |
| FR-2.8 | Fall back without the mesh | Version 1 eye-darkness heuristic; each frame records `eye_method` | **Done** |
| FR-2.9 | Group by person | SFace 128-d embeddings of main faces; density clustering on cosine distance (`eps = 0.45`, `min_samples = 2`); faces seen once left unassigned; user-named; stored in the edit folder; **Forget Faces** deletes it | **Done** — accuracy on a real event unmeasured ([8.3](#83-face-recognition-is-checked-for-consistency-not-accuracy)) |
| FR-2.10 | Survey | 2–6 selected frames side by side, zoom and pan locked | **Done** |
| FR-2.11 | Face loupe | Every face in the selection at full size | **Done** |
| FR-2.12 | Keyboard culling | Arrows; 1–5 stars, 0 clears; 6–9 red/yellow/green/blue (again clears); P pick; X reject; U clear; Enter to Develop; S survey; F loupe | **Done** |
| FR-2.13 | Marks persist | Ratings, labels, picks, rejects, categories, web marks in `session.granite.json`; a damaged file opens unmarked | **Done** |

**Modules:** `ai_tools/culling.py`, `ai_tools/face_mesh.py`,
`ai_tools/person_clustering.py`, `ui/workspaces/cull.py`.

**Measured — eye aspect ratio** (four reference faces from OpenCV's sample
data, upscaled 3×, `tests/test_face_mesh_ear.py`):

| Mesh | Open eyes | Same faces, eyes painted shut | Usable |
| --- | --- | --- | --- |
| 468-point (MediaPipe `face_landmark`) | 0.22–0.32 | 0.17–0.26 | **No** — ranges overlap |
| **478-point (MediaPipe face landmarker)** | **0.155–0.282** | **0.071–0.142** | Yes, with one limit below |
| Trained "eyeBlink" coefficient | 0.09–0.58 | 0.42–0.72 | No — worse than the ratio |

Three of four painted-shut faces fall below the threshold; all four open faces
stay above it. The fourth — a ~30 px face looking down — reads 0.155 open and
0.142 shut, both beside the threshold, and is pinned as a known limit by
`test_known_limit_a_small_downward_gaze_is_ambiguous`.

**Measured — smile:** 0.91 on a broad smile, 0.06 on a neutral face, 0.21 on
an open mouth mid-action (the older expression classifier had called that
last one "happy" at 0.91 and was replaced). **Head roll:** tracks a known
rotation within 2° across ±25° (asserted); 38.7° and 34.9° measured at ±40°.
**Cost:** 12.9 ms per face.

### FR-3 Blur triage

Unchanged in behaviour from 1.0; now reads greyscale exact decodes and runs on
a thread pool.

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-3.1 | Distinguish motion blur from missed focus | Directional gradient energy; consulted only once sharpness is low | **Done** |
| FR-3.2 | Report the smear direction | | **Done** |
| FR-3.3 | Strictness tiers | `aggressive` / `balanced` / `conservative` | **Done** |
| FR-3.4 | Blurry frames are rejected, not deleted | | **Done** (INV-1) |
| FR-3.5 | Offer sharpening only where detail remains | Adds sharpening to the frame's **recipe** (2.0) rather than writing `*_sharpened.jpg` | **Done** |

**Measured:** 15/15 synthetic motion cases detected with direction within
15° (1.0). Motion vs. defocus and the exact-decode calibration chain are
asserted in `test_analysis.py` and `test_fast_preview.py`.

### FR-4 Orientation and levelling

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-4.1 | Recover 90/180/270 rotations | Faces, then body | **Done** |
| FR-4.2 | Level from scene lines; body when there are none | Unchanged from 1.0 | **Done** |
| FR-4.3 | Apply as a recipe | `set_rotation()` stores quarter turns plus straightening; nothing written | **Done** (2.0) |
| FR-4.4 | Corners | **Crop** to the largest clean rectangle (inset 2 px for resampling), **Fill** by inpainting at export, or **Leave black** | **Done** (2.0 adds Fill) |
| FR-4.5 | Never read blank padding as structure | | **Done** |

**Measured:** known tilts round-trip under 1° (asserted at −8°, −3°, 4°, 9°);
the recipe path matches the direct rotation within 0.6°. Body levelling 0.6°
mean / 1.7° worst over ±14° (1.0). Head roll from the new mesh is measurable
but deliberately not used ([8.1](#81-head-roll-is-measured-but-not-used-to-level)).

### FR-5 Posture analysis

Unchanged from 1.0: shoulders, spine and hips from BlazePose; camera roll told
from a lean by shoulder/hip agreement within `MAX_ASYMMETRY_DEG = 12.0`;
shoulders alone never applied; **Check Posture** changes nothing.
Measurements in 1.0 §FR-5 still hold.

### FR-6 Review and rejection

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-6.1 | Select all / none / invert | Scoped to visible frames | **Done** |
| FR-6.2 | Reject, restore, review before acting | | **Done** |
| FR-6.3 | Rejection is never deletion | | **Done** (INV-1) |
| FR-6.4 | Filter the grid | Keepers, AI picks, duplicates, rejected, blurry, motion, out of focus; and by person | **Done** |

### FR-7 Develop engine

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-7.1 | Parametric recipe | Versioned JSON in the edit folder named for the whole filename (`IMG_1.CR2` and `IMG_1.JPG` do not collide); neutral recipe removes the file; written atomically | **Done** |
| FR-7.2 | Tolerant loading | Unknown fields ignored, wrong types defaulted, out-of-range values clamped, non-JSON opens neutral with `load_error`, newer format versions load with a warning | **Done** |
| FR-7.3 | A neutral recipe changes nothing | Bit-identical output for 8- and 16-bit sources, also through the full float path | **Done** |
| FR-7.4 | RAW from linear 16-bit | LibRaw to linear sRGB, 16-bit; a measured baseline exposure (0–3 stops, from the frame's median) so an unedited RAW is not dark; embedded preview when LibRaw cannot decode the file, with a note | **Done** |
| FR-7.5 | Tone | Exposure (linear, ±5 EV), highlight roll-off instead of clipping, highlights, shadows, whites, blacks, contrast, point curves (master and per channel, monotone cubic) | **Done** |
| FR-7.6 | Presence | Texture, clarity (both on luminance, midtone-weighted), dehaze (dark-channel prior with guided filtering; negative adds haze) | **Done** |
| FR-7.7 | Colour | Temperature and tint (luminance-preserving gains), vibrance, saturation, 8-band HSL, three-way colour grading with balance and blending | **Done** |
| FR-7.8 | Looks and LUTs | The six 1.0 grades as looks with strength; 3D `.cube` LUTs with domain support and strength; malformed LUTs rejected then ignored | **Done** |
| FR-7.9 | Detail | Sharpening (amount, radius), noise reduction | **Done** |
| FR-7.10 | Lens | lensfun profile: distortion and lateral colour (geometry) and vignetting; manual distortion and vignette sliders | **Done** — profile matched by EXIF make, model and lens |
| FR-7.11 | Geometry | Quarter turns, flips, straighten ±45°, vertical and horizontal keystone, crop by fractions; crop-to-clean or fill corners | **Done** |
| FR-7.12 | Preview equals export | Large-radius effects built once on a 768 px copy and scaled, so a preview and a full export agree (mean difference < 2 levels, asserted) and exports render in 512-row strips with no seams (≤ 1 level, asserted) | **Done** |
| FR-7.13 | Export formats | JPEG, PNG, TIFF (LZW), 16-bit TIFF, 16-bit PNG; size limit; DPI tag | **Done** |

**Module:** `engine/parametric_recipe.py`, `engine/develop_pipeline.py`.

**Measured — preview cost** on the i5-8350U at 1600 px (800 px in brackets):
exposure only **13 ms** (9); exposure, contrast, white balance and curve 16 ms
(12); adding shadows and highlights 311 ms (141); clarity and texture 463 ms
(215); vibrance and HSL 617 ms (254); sharpening 658 ms (264). Per-value
settings are folded into exact 65,536-entry tables, which is why the first
two rows are fast; the lookup path matches the float path to ≤ 1 level
(asserted).

**Measured — looks:** each of the six looks renders within 1 level of the 1.0
grade it replaced (asserted).

**Not claimed:** GPU acceleration of the develop engine. It is vectorised
NumPy and OpenCV on the CPU.

### FR-8 Background removal

| ID | Requirement | Status |
| --- | --- | --- |
| FR-8.1 | Subject cutout to transparent PNG, from the edited frame | **Done** |
| FR-8.2 | Subject mask usable as a layer mask | **Done** ([FR-16](#fr-16-layers-masks-and-retouching)) |
| FR-8.3 | Composite a replacement background | **Not built** — needs multi-image compositing, out of scope |

### FR-9 Interoperability (XMP sidecars)

Unchanged from 1.0: ratings and labels written beside each image, merging with
another tool's develop settings, title and keywords; corrupt sidecars degrade.
Granite recipes are **not** translated into Lightroom develop settings.

### FR-10 Web gallery publishing

As 1.0, and in 2.0 an edited frame is rendered through its recipe first;
`*_graded.jpg` files left by 1.0 are still used.

### FR-11 Social export

As 1.0 (reel, 4:5 feed, 9:16 story), and in 2.0 each reads the edited
frame. Direct upload: **Not built**.

### FR-12 Video engine

Unchanged from 1.0 and moved intact into the Video workspace. See the 1.0
document for the operation list and verification.

### FR-13 Hosted browser version

Unchanged. The backend's imports from `src/` are checked by
`test_backend_smoke.py` on every run so a refactor cannot silently break it.
Still **not deployed** ([12.4](#124-the-hosted-build-is-not-deployed)).

### FR-14 Installation, update and uninstall

As 1.0, plus: the installer removes the pre-2.0 `onnxruntime` package before
installing `onnxruntime-directml` (both occupy the same import folder), and
pre-fetches the face mesh with the face detector.

### FR-15 Workspaces and keyboard

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-15.1 | Task workspaces | Cull, Develop, Retouch, Design, Export, Video in one bar; Ctrl+1–6 | **Done** |
| FR-15.2 | Shared state | INV-6 | **Done** |
| FR-15.3 | Open a frame for editing | Double-click a thumbnail or press Enter | **Done** |
| FR-15.4 | Develop keyboard | ←/→ frames; `\` before/after; J clipping; Ctrl+Z / Ctrl+Y; Ctrl+Shift+C / V copy and paste settings; 0 fit; 1 100% | **Done** |
| FR-15.5 | Histogram with clipping | RGB histogram, shadow and highlight clipping indicators and percentages; clipping overlay (blue crushed, red blown) | **Done** |
| FR-15.6 | Responsive previews | Latest-only background rendering: while one render runs, newer requests replace older ones and overtaken results are dropped | **Done** |
| FR-15.7 | Sync settings | Copy develop settings to every selected frame; crop, layers and overlays stay with their own photo | **Done** |
| FR-15.8 | Ask before large downloads | One dialog for every model beyond the face detector and mesh, naming purpose and size; a "no" fetches nothing | **Done** |

**Module:** `ui/main_window.py`, `ui/workspaces/`, `ui/widgets/`.

**Verified:** `tests/test_workspaces.py` drives every workspace headlessly —
keyboard marks, survey lock, face loupe, develop sliders and undo, sync, aspect
crop and looks, healing and brushed layers, design snapping and SVG export,
an overlay appearing in an export, consent declined, grouping by person end to
end. Every workspace was also rendered with real camera files and inspected.

### FR-16 Layers, masks and retouching

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-16.1 | Layer stack | Kinds: adjustment, retouch, pixel, mask, text, shape. Opacity, visibility, order; stored in the recipe as data, positions as fractions of the frame | **Done** |
| FR-16.2 | Blend modes | Normal, multiply, screen, overlay, soft light, colour dodge, luminosity — W3C compositing formulas | **Done** — 16 hand-computed values asserted |
| FR-16.3 | Masks | None, AI, gradient, radial, luminance range, brush (paint and erase), raster; invert, feather, density | **Done** |
| FR-16.4 | AI masks | Subject (U2Net, or the person model), person (PP-HumanSeg), skin (mesh outline grown by a per-face colour model; geometry only for black-and-white), lips, teeth, eyes, iris (mesh), click-to-select (EfficientSAM) | **Done** |
| FR-16.5 | Hair and sky | Hair: EfficientSAM seeded above each forehead minus the face. Sky: blue or bright-overcast, smooth, connected to the top edge | **Partial** — work on the test images, unmeasured, labelled approximate |
| FR-16.6 | Skin smoothing | Three-band split sized by eye distance; only the middle band is reduced, by up to 85% | **Done** — blotch energy cut > 80%, pore energy kept > 90% (asserted) |
| FR-16.7 | Teeth and eyes | Teeth: mostly desaturation, small lift, never to white. Eyes: iris local contrast, whites slightly cleared | **Done** |
| FR-16.8 | Heal and clone | Stored as positions and radii; filled at render time | **Done** |
| FR-16.9 | Inpainting | Fast marching for holes under 24 px; LaMa for larger, in a context window, only the hole replaced | **Done** — LaMa output checked for colour (blue sky stays blue); visual quality on real prints unjudged |
| FR-16.10 | A failing layer must not lose the picture | Skipped with a reason; the rest of the stack applies | **Done** |

**Modules:** `engine/layer_stack.py`, `ai_tools/segmentation.py`,
`ai_tools/retouch.py`, `ui/workspaces/retouch.py`.

**Measured:** PP-HumanSeg 15 ms; EfficientSAM 2.4 s; LaMa **12.9 s** per 512 px
tile on the CPU (one measurement).

### FR-17 Design and vector graphics

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-17.1 | Items | Text, rectangle (rounded), ellipse, star, bezier path (M/L/Q/C/Z), group; rotation, opacity, fill, stroke, drop shadow | **Done** |
| FR-17.2 | Typography | Any installed font; size, bold, italic, kerning, tracking, leading, alignment, nine anchors | **Done** |
| FR-17.3 | Resolution independence | Positions and sizes in fractions of the canvas; text laid out once at a reference size with hinting off, then scaled | **Done** — layout width exactly 10× at 10× size; coverage equal within 3% (asserted) |
| FR-17.4 | Outputs | Raster at any size; SVG whose viewBox and physical size equal the pixel size at 96 dpi, text kept as `<text>` or written as outlines; one-page PDF at a physical size, photograph underneath | **Done** |
| FR-17.5 | Watermarks | Signature, round seal (name fitted inside the ring), and a camera-details strip filled from `{camera} {lens} {settings} {focal} {iso} {aperture} {shutter} {date} {year}` per photograph, with empty values leaving no stray separators | **Done** |
| FR-17.6 | Snapping | Grid snap; alignment guides to other items' edges and centres and the frame's | **Done** |
| FR-17.7 | Overlays render over the final framing | Drawn after crop and straighten | **Done** |

**Module:** `graphics/vector_canvas.py`, `ui/workspaces/design.py`.

### FR-18 Export

| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-18.1 | Presets | Web JPEG 2048 px; full-size JPEG; 16-bit TIFF; print JPEG at 300 dpi; 16-bit PNG | **Done** |
| FR-18.2 | Render everything | Recipe, layers (with the models and patch files they need), overlays, corner fill | **Done** |
| FR-18.3 | Batch watermark | Any template, filled per frame | **Done** |
| FR-18.4 | Unique names | Two sources sharing a stem (`IMG_1.CR2`, `IMG_1.JPG`) get their extension added | **Done** |
| FR-18.5 | One bad frame does not stop a batch | Failures listed afterwards | **Done** |

**Module:** `engine/export_presets.py`, `ui/workspaces/export.py`.

---

## 5. Non-functional requirements

### NFR-1 Performance

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-1.1 | Cull 1,000 RAW files fast | **Met for the 1.0 target of 5 minutes on all nine cameras tested. The 2.0 target of 90 seconds is met on five of nine.** |
| NFR-1.2 | Grid and filmstrip smooth on integrated graphics | **Done** |
| NFR-1.3 | Simple develop edits feel immediate | **Done** — 13–16 ms per preview at 1600 px |
| NFR-1.4 | Analysis independent of source resolution | **Done** |

**NFR-1.1 measurements** (`tests/bench_cull.py`; i5-8350U, 4 cores / 8
threads, 32 GB RAM, 7 workers; 1,000 frames per camera; eye checking on):

| Camera | Embedded preview | First build | Final |
| --- | --- | --- | --- |
| Sony a6400 (ARW) | 1616×1080 | 100 s | **42 s** |
| Samsung S10 (DNG) | 4016×1892 | 95 s | **51 s** |
| Olympus E-M1 II (ORF) | 3200×2400 | 115 s | **53 s** |
| Canon EOS R6 (CR3) | 3408×2272 | 111 s | **61 s** |
| Panasonic GH5S (RW2) | 1920×1920 | 163 s | **85 s** |
| Fujifilm X-S10 (RAF) | 4416×2944 | 166 s | 91 s |
| Nikon Z50 II (NEF) | 3712×3712 | 224 s | 96 s |
| Canon 5D IV (CR2) | 6720×4480 | 258 s | 100 s |
| Pentax K-1 II (PEF) | 4800×3200 | 151 s | 104 s |

**Conditions that limit these numbers.** Each run repeats one file a thousand
times, so after the first read it is served from the operating system's
cache: this measures decoding and analysis, not reading a thousand different
files from a card. Two files with no usable embedded preview (iPhone XS DNG,
Hasselblad H3D) are developed instead and took 291 s and 616 s per 1,000 in
the first build; they were not re-run. What did not help: single-threaded
OpenCV per worker (< 3%); finding faces at 640 px instead of 800 (~8%, at
the cost of small faces).

### NFR-2 Resource budgeting

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-2.1 | Batch work never starves the OS | **Done** — export budget ½ available RAM (≤ 8 GB); cull workers capped by free memory (0.35 GB each) |
| NFR-2.2 | Bounded memory regardless of frame size | **Done** — strip-wise rendering; previews from 1600 px proxies |
| NFR-2.3 | Bounded preview cache | **Done** — 256-image LRU |

### NFR-3 Hardware adaptation

Every ONNX Runtime session goes through `engine/hardware.py`: **DirectML** first
on Windows (`DmlExecutionProvider`, any DirectX 12 adapter), Core ML on macOS,
then CUDA (only when an NVIDIA card is present) or OpenVINO, **CPU always
last**; a provider that fails to start falls back to the CPU.
`GRANITE_ORT_PROVIDER` overrides the choice.

**Measured** (Intel UHD 620, background-removal model): DirectML output within
3e-7 of the CPU and the final cut-out within 1 level in 255; inference
**461 ms vs 625 ms (1.36×)**; whole background removal 470 ms vs 742 ms. The
**first** DirectML session on a machine took **76 s** (driver shader
compilation); later ones 0.6 s. Switching runtimes changed nothing on the CPU
path (pixel-identical cut-out).

**Scope:** only ONNX Runtime models (today, background removal). Face, pose,
mesh, recognition, segmentation and inpainting run through OpenCV's engine on
the CPU. No discrete GPU was available to measure.

### NFR-4 Reliability and failure behaviour

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-4.1 | Missing model disables its feature | **Done** (INV-3) |
| NFR-4.2 | Damaged recipe, session, people index, LUT or sidecar degrades | **Done** |
| NFR-4.3 | UI never freezes | **Done** — task worker, latest-only preview renderer, thread pools |
| NFR-4.4 | One failing layer or frame does not stop the rest | **Done** |
| NFR-4.5 | A RAW the decoder cannot read is still editable | **Done** — falls back to its embedded preview, with a note (Nikon Z50 II in the sample set) |

### NFR-5 Privacy and security

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-5.1 | No user data leaves the machine (desktop) | **Done** (INV-2) |
| NFR-5.2 | Models pinned | **Done** (INV-5) |
| NFR-5.3 | Consent before large downloads | **Done** — FR-15.8; the face detector (0.2 MB) and mesh (3.8 MB) are fetched by the installer |
| NFR-5.4 | Face data is local and deletable | **Done** — INV-2, **Forget Faces** |
| NFR-5.5 | Asset names cannot escape their folder | **Done** — layer asset names reduced to a bare filename (asserted) |

### NFR-6 Supportability

Install transcript and pip log in `%TEMP%`; hardware summary (including the
chosen provider list) at startup; every preview records how it was decoded;
every mask and skipped layer records why.

### NFR-7 Interface constraints

Dark theme; workspace bar; status bar shared by all workspaces; headless Qt is
pointed at the system font folder (otherwise every glyph renders as a box).

---

## 6. Architecture

### 6.1 Layers

```
┌───────────────────────────────────────────────────────────────────┐
│ Presentation                                                      │
│   ui/main_window.py          workspace bar, consent, task runner   │
│   ui/workspaces/             cull develop retouch design export    │
│                              video                                 │
│   ui/widgets/                canvas, histogram, curves, filmstrip, │
│                              latest-only renderer                  │
│   web/frontend/              React (hosted build)                  │
├───────────────────────────────────────────────────────────────────┤
│ Session                                                           │
│   engine/session.py          the open folder: selection, marks,    │
│                              recipes, people                       │
├───────────────────────────────────────────────────────────────────┤
│ Domain                                                            │
│   engine/parametric_recipe   engine/develop_pipeline               │
│   engine/layer_stack         engine/export_presets                 │
│   graphics/vector_canvas                                          │
│   ai_tools/  culling face_mesh person_clustering segmentation      │
│              retouch blur orientation pose face colour bg xmp      │
│   video/, social/, web/                                            │
├───────────────────────────────────────────────────────────────────┤
│ Platform                                                          │
│   engine/fast_preview  universal_loader  phone_media  hardware     │
│   ai_tools/model_store                                             │
└───────────────────────────────────────────────────────────────────┘
```

**Nothing below the session layer imports Qt widgets.** The vector canvas uses
Qt's painter and font engine, which run without a window. This is what lets
the hosted backend and the tests use the same code as the desktop.

### 6.2 Module responsibilities

| Module | Responsibility | Lines |
| --- | --- | --- |
| `ui/workspaces/cull.py` | Grid, cull, marks, survey, loupe, people | 1,439 |
| `engine/develop_pipeline.py` | Float develop, geometry, lens, LUTs, writing | 1,232 |
| `ai_tools/culling.py` | One-pass scoring, faces, keeper score, bursts | 764 |
| `graphics/vector_canvas.py` | Items, layout, raster/SVG/PDF, templates, snapping | 748 |
| `ai_tools/pose_analysis.py` | BlazePose, posture | 726 |
| `ui/workspaces/develop.py` | Develop panel and preview | 645 |
| `engine/fast_preview.py` | Preview location and decoding | 596 |
| `ui/workspaces/retouch.py` | Layers panel and tools | 594 |
| `engine/layer_stack.py` | Layers, masks, blend modes, heal, clone | 583 |
| `ui/workspaces/design.py` | Vector editing | 566 |
| `ai_tools/segmentation.py` | All masks | 518 |
| `ui/workspaces/export.py` | Presets, publishing | 486 |
| `engine/parametric_recipe.py` | Recipe format and sidecars | 484 |
| `ai_tools/face_mesh.py` | Mesh, EAR, mouth, roll, expressions | 482 |
| `engine/universal_loader.py` | Formats, metadata, edit folders | 463 |
| `ai_tools/face_analysis.py` | YuNet detection, eye heuristic | 453 |
| `engine/phone_media.py` | Gain maps, motion and live photos, deep HEIC | 447 |
| `ai_tools/retouch.py` | Skin, teeth, eyes, inpainting | 337 |
| `ai_tools/person_clustering.py` | Embeddings, DBSCAN, people index | 312 |
| `engine/session.py` | Shared state and persistence | 281 |
| `ui/main_window.py` | Workspace bar, consent, tasks | 319 |
| `engine/hardware.py` | Providers, sessions, RAM | 241 |

### 6.3 The edit folder contract

```
Vacation2024/                          # originals — never modified or moved
├── DSC_0001.NEF
├── DSC_0002.NEF
└── Vacation2024-edit/                 # created on load
    ├── DSC_0001.NEF.granite.json      # recipe (INV-7)
    ├── DSC_0001.NEF.granite.assets/   # pixel-layer patches, if any
    ├── session.granite.json           # ratings, labels, picks, rejects, web marks
    ├── people.granite.json            # who appears where
    ├── people.granite.npz             # face embeddings (deleted by Forget Faces)
    ├── DSC_0001_edit.jpg              # delivery copy for the publishing tools
    └── DSC_0002_nobg.png              # background removal
```

File exports go where the user chooses. Video keeps its per-clip
`clip-edit/` chain.

### 6.4 Model dependencies

| Model | Purpose | Bytes | Source | When fetched |
| --- | --- | --- | --- | --- |
| YuNet | Face detection | 232,589 | opencv_zoo | Installer |
| MediaPipe face landmarker | Mesh, EAR, expressions | 3,758,596 | Google `mediapipe-models` | Installer |
| BlazePose detector + landmarks | Posture, body levelling | 11,990,159 + 5,557,238 | opencv_zoo | On consent |
| SFace | Face recognition | 38,696,353 | opencv_zoo | On consent |
| PP-HumanSeg | Person mask | 6,163,938 | opencv_zoo | On consent |
| EfficientSAM-Ti | Click-to-select, hair | 48,312,857 | opencv_zoo | On consent |
| LaMa | Neural inpainting | 92,591,623 | opencv_zoo | On consent |
| U2Net (via `rembg`) | Background removal, subject mask | 175,997,641 | rembg | Installer |

All are pinned by SHA-256 and size. The face landmarker is a zip archive,
verified whole and unpacked in memory, so no unverified file reaches disk.

### 6.5 Threading

- **Thumbnail decode** — cancellable worker.
- **Long tasks** — one `TaskWorker` at a time, guarded.
- **Batch analysis** — `fast_preview.map_pairs` thread pool, one OpenCV
  network per thread (`ThreadLocalFaceMesh`, per-thread detectors).
- **Previews** — `LatestOnlyRenderer` per workspace: one render at a time,
  newest request wins, stale results dropped.
- **Video** — `JobManager` pool.

---

## 7. Quality and verification

### 7.1 Approach

**493 pytest tests** (492 pass and 1 skips on the development machine, where
the hosted backend's own requirements are not installed). They exercise the
real engines and the real window — headless, with `QT_QPA_PLATFORM=offscreen`
— and mock nothing but file dialogs and, in two tests, the network.

Tests needing a cached model or a sample file are marked `model` / `media` and
skip when it is absent. `python tests/fetch_media.py` fetches eleven CC0 camera
files from raw.pixls.us and four reference faces from OpenCV's sample data,
each verified by SHA-256, into `~/.granitestudios/test-media` — outside the
repository, which may sit on a synced drive.

### 7.2 Coverage

| File | Tests | What it pins |
| --- | --- | --- |
| `test_invariants.py` | 24 | INV-1 for every library action; rejected frames never exported; recipes instead of files; reopening a folder |
| `test_fast_preview.py` | 74 | Preview paths, orientation, capture time, the exact-decode calibration, JPEG scanning, every sample format, extraction < 25 ms |
| `test_develop_pipeline.py` | 65 | Identity, tone, colour, HSL, grading, LUTs, presence, strips, preview = export, 16-bit, geometry, lens, formats, RAW baseline and fallback |
| `test_parametric_recipe.py` | 53 | Round trips, tolerance of bad input, sidecars, INV-7, looks match 1.0 |
| `test_vector_canvas.py` | 49 | Shapes, text layout and scaling, tokens, snapping, SVG and PDF dimensions, templates |
| `test_layer_stack.py` | 47 | Blend-mode values, masks, compositing, pixel layers, through the pipeline |
| `test_face_mesh_ear.py` | 38 | EAR geometry exactly; open/shut on reference faces; head roll; smile vs open mouth |
| `test_retouch.py` | 20 | Frequency separation keeps pores; teeth; eyes; inpainting; corner fill; masks |
| `test_workspaces.py` | 21 | Every workspace driven end to end |
| `test_phone_media.py` | 18 | Live/Motion Photos, gain-map arithmetic, 10-bit HEIC, metadata |
| `test_culling_score.py` | 16 | Keeper score arithmetic, composition, region sharpness, the mesh in a cull |
| `test_hardware.py` | 14 | Provider choice per platform, fallback, agreement with the CPU |
| `test_backend_smoke.py` | 14 | Every engine name the hosted backend imports still exists |
| `test_person_clustering.py` | 13 | DBSCAN exactly; people index; recognition self-consistency |
| `test_analysis.py` | 12 | Blur classes, levelling round trips, blank padding |
| `test_video_ingest.py` | 8 | ProRes, DNxHD, H.264, VP9 recognised |
| `test_model_store.py` | 7 | Pinning, tampering, all-or-nothing |

### 7.3 Gaps

- **No CI run of the suite.** Tests run on the development machine; the GitHub
  workflows build installers but do not run pytest.
- **Accuracy figures rest on small samples** — four faces, eleven RAW files,
  synthetic blinks. See section 8.

---

## 8. Known limitations

### 8.1 Head roll is measured but not used to level

The 478-point mesh tracks head roll (within 2° across ±25°), which the 1.0
detector could not (24° of rotation read as 5°). It is not used for levelling:
a tilted head is usually the subject's pose, and rotating the frame to undo it
would tilt the horizon. Same reasoning as spine lean.

### 8.2 Blink detection is checked on synthetic blinks

The eye-aspect-ratio threshold was set against four faces with eyes painted
shut, because no labelled photographs of real blinks were available. That
shows the measurement responds to closed eyes; it is not validation on real
ones. Four faces is not a dataset: skin tones, ages, glasses, sunglasses and
side lighting are untested. A small face looking down is ambiguous
(0.155 open vs 0.142 shut, threshold 0.135).

### 8.3 Face recognition is checked for consistency, not accuracy

The same face under brightness, blur and 6° rotation changes stays closer to
itself (weakest match above SFace's 0.363 threshold) than to three other
people (strongest cross match 0.291 in greyscale). That proves the pipeline is
wired correctly. How often two guests are merged, or one person split, at a
real event is unmeasured.

### 8.4 Phone formats are unverified on real files

Ultra HDR, Motion Photos, Live Photo pairing and 10-bit HEIC are tested on
files built here to the published layouts, not on files from phones. Two gaps
are known: the binary ISO 21496-1 gain-map metadata is not parsed (files with
only that report no gain map), and Apple's ProRAW `ProfileGainTableMap` is
detected but not applied, so ProRAW renders flatter than in Photos. Live
Photo pairing is by filename only.

### 8.5 Some masks are approximate

Hair (click-to-select seeded above the forehead) and sky (colour, smoothness
and position) are unmeasured. A pale wall at the top of a frame can be taken
for sky.

### 8.6 Video editing is file-chained

Each video operation writes a new clip; there is no parametric video recipe.
The non-destructive model of INV-7 covers photographs only.

### 8.7 Neural inpainting is slow on a CPU

About 13 s per 512 px tile. Its visual quality on real photographs, and
whether a filled corner passes in a print, has not been judged. The fast fill
is always available.

### 8.8 Portrait RAW orientation is unverified

Every sample RAW was shot in landscape (`flip = 0`). The code that turns an
embedded preview upright from the RAW's orientation flag has not seen a real
portrait frame.

### 8.9 Lens profiles depend on EXIF names

lensfun matches by camera and lens name. Several sample files carry no lens
name at all, so they get no profile; the interface says so.

### 8.10 Two sample cameras are not decodable by LibRaw 0.22

The Nikon Z50 II NEF and the Samsung S10 DNG fail LibRaw's full decode. Both
cull normally from their previews and open in Develop from their embedded
JPEG, with a note.

---

## 9. Target state

Each item names its actual blocker.

| Requirement | Blocker |
| --- | --- |
| Validate blink, smile and recognition on real photographs | A labelled set of real event photographs, with consent from the people in them |
| Apply ProRAW gain tables | A real ProRAW file to verify against |
| Parse ISO 21496-1 binary gain-map metadata | A sample file and the specification |
| Run the suite in CI | Model and sample downloads in the workflow, or a model-free subset |
| GPU develop engine | A rewrite of the pipeline for OpenCL or compute shaders, and hardware to measure it |
| Parametric video | A video recipe and a render-on-export pipeline |
| Background replacement | Multi-image compositing, currently out of scope |
| AI profiles trained from a photographer's catalogue | A training pipeline and model architecture — the largest item in the reference spec |
| Client proofing, print ordering | A hosted backend, accounts and payments; in tension with INV-2 |
| Deploy the hosted build | [12.4](#124-the-hosted-build-is-not-deployed) |
| Code signing | Certificate cost |

---

## 10. Release and distribution

### 10.1 Windows (primary)

One file, `Install-GraniteStudios.bat`, from the latest release; it fetches the
app, builds a private venv, installs dependencies (DirectML runtime on
Windows), pre-fetches the background-removal, face-detection and face-mesh
models, creates shortcuts, registers for uninstall and starts the app
headlessly to confirm it works. Re-running it updates. The PyInstaller build
collects the lens database and the 2.0 hidden imports.

**2.0 is on the `v2` branch.** The installer republishes from `main`, so
merging `v2` is what releases it.

### 10.2 macOS

PyInstaller `.app` in a `.dmg`, unsigned. **Untested in 2.0** — the
workspaces, Core ML provider choice and font handling have only been run on
Windows.

### 10.3 Platform matrix

| Platform | Status |
| --- | --- |
| Windows 10/11 | **Primary** — tested |
| macOS | **Builds; 2.0 untested** |
| Linux | Source only |
| Browser | Built, not deployed |

---

## 11. Success metrics

> **Proposed and unvalidated.** The application has no telemetry, by design
> (INV-2). Any measurement programme must be opt-in and local.

| Metric | Proposed target | Status |
| --- | --- | --- |
| Time to cull 1,000 RAW | < 90 s | **Measured: 42–104 s; met on 5 of 9 cameras** |
| Simple develop edit to preview | < 50 ms | **Measured: 13–16 ms** |
| AI picks kept by the photographer | ≥ 80% | Needs the study in [2.4](#24-target-users) |
| Blinks caught on real photographs | ≥ 95% at ≤ 5% false alarms | Needs a labelled set ([8.2](#82-blink-detection-is-checked-on-synthetic-blinks)) |
| Originals lost or modified | **Zero, permanently** | INV-1, tested on every action |

---

## 12. Risks and open questions

### 12.1 Upstream model availability

Seven of nine models come from `opencv/opencv_zoo`, one from Google's
`mediapipe-models` bucket, one via `rembg`. A moved or re-exported file breaks
the feature for new installs (pinning rejects it rather than substituting).
Mitigation to consider: a mirror, or bundling the two small face models.

### 12.2 OpenCV 5 API churn

Three defects so far: no `CascadeClassifier`; `HoughLinesP` shape change; and
in 2.0, `forward()` on several named outputs fails for the face landmarker
graph, so outputs are read one at a time. The project does not pin OpenCV.

### 12.3 No code signing

SmartScreen and Gatekeeper warn on first run.

### 12.4 The hosted build is not deployed

Unchanged from 1.0: the Vercel project does not exist and the portfolio card
404s. **Open question: deploy, or remove the link?**

### 12.5 Single maintainer

The suite now makes regressions visible, but it runs only on one machine and
has one reviewer.

### 12.6 Open product questions

1. Is the keeper-score weighting (0.40/0.35/0.15/0.10) right for real
   photographers? It is the reference spec's, unvalidated.
2. Should head roll ever be offered as an optional correction for portraits?
3. Should client delivery be pursued at all, given INV-2?
4. Should the two small face models ship inside the installer?

---

## 13. Appendix

### 13.1 Key constants

| Constant | Value | Module | Meaning |
| --- | --- | --- | --- |
| `EAR_CLOSED_BELOW` | 0.135 | `face_mesh` | A face's more open eye below this is a blink |
| `EAR_OPEN_ABOVE` | 0.20 | `face_mesh` | Fully open, for the 0–1 openness score |
| `JAW_OPEN_ABOVE` | 0.30 | `face_mesh` | Open mouth |
| `SMILE_ABOVE` | 0.45 | `face_mesh` | Smiling |
| `WEIGHT_*` | 0.40 / 0.35 / 0.15 / 0.10 | `culling` | Keeper score: face sharpness, eyes, smile, composition |
| `MAIN_FACE_RATIO` | 0.5 | `culling` | Faces under half the largest are bystanders |
| `DETECT_MAX_SIDE` | 800 px | `culling` | Face search size |
| `DEFAULT_EPS` / `DEFAULT_MIN_SAMPLES` | 0.45 / 2 | `person_clustering` | Clustering radius (cosine distance) and density |
| `DEFAULT_MIN_SIDE` | 900 px | `fast_preview` | Smallest embedded preview accepted for analysis |
| `CONTEXT_SIDE` | 768 px | `develop_pipeline` | Small copy for large-radius effects |
| `STRIP_ROWS` | 512 | `develop_pipeline` | Export strip height |
| `KNEE` | 0.6 | `develop_pipeline` | Highlight roll-off start (linear) |
| `PREVIEW_SIDE` | 1600 px | workspaces | Preview proxy size |
| `SMALL_HOLE_PX` | 24 | `retouch` | Holes narrower than this use the fast fill |
| `EYE_OPEN_THRESHOLD` | 0.42 | `face_analysis` | Fallback heuristic only |
| `MIN_CORRECTION_DEG` / `MAX_FINE_CORRECTION_DEG` | 0.75 / 20.0 | `orientation` | Levelling bounds |
| `MAX_ASYMMETRY_DEG` / `MIN_APPLY_CONFIDENCE` | 12.0 / 0.55 | `pose_analysis` | Body levelling gates |
| Blur strictness | 95/230/30, 180/420/45, 42/110/18 | `blur_analysis` | Blurry / soft / rescue floor per tier |

### 13.2 Glossary

| Term | Meaning |
| --- | --- |
| **Recipe** | A photograph's edit settings; `<filename>.granite.json` in the edit folder |
| **Session** | The open folder's shared state across workspaces |
| **Reject** | Leave out of every export. **Never a delete** |
| **EAR** | Eye aspect ratio: eye height over width from six mesh points |
| **Keeper score** | The weighted ranking used inside a burst |
| **Main face** | A face at least half the size of the largest in the frame |
| **Layer** | A masked edit over the developed frame, stored as data |
| **Proxy** | The 1600 px copy previews are rendered from |
| **Delivery copy** | `*_edit.jpg`, a rendered edit for the publishing tools |

### 13.3 Dependencies

PyQt6 · rawpy · Pillow / pillow-heif · opencv-python-headless · numpy · scipy
· tifffile · lensfunpy · onnxruntime-directml (Windows) / onnxruntime · rembg ·
moviepy · imageio-ffmpeg · fastapi / uvicorn / pydantic · psutil · pyshortcuts.
Development: pytest.

### 13.4 Related documents

- [`README.md`](README.md) — user guide
- [`README_INSTALL.md`](README_INSTALL.md) — SmartScreen and Gatekeeper notes
- [`web/README.md`](web/README.md) — browser build
- [`tests/bench_cull.py`](tests/bench_cull.py) — the cull benchmark and its recorded results
- [PRD 1.0](https://github.com/marufmhoque/GraniteStudios/blob/12aa977/PRD.md) — measurements for features unchanged since

---

*Granite Studios — your RAW, edited locally. Never in the cloud.*
