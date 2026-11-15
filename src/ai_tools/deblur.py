"""De-blurring: recovering motion smear and missed focus, not sharpening.

Sharpening (``blur_analysis.sharpen``, the Detail panel's slider) raises the
contrast of edges that survived. This module inverts the blur itself. A
blurred photograph is, to a good approximation, the sharp scene convolved
with a *point spread function* (PSF): a short line for camera or subject
motion, a small disc for a missed focus. Given that PSF, deconvolution
estimates the scene that produced the photograph.

Engines
-------
``wiener``      Closed-form division in the frequency domain, regularised
                by a signal-to-noise ratio. One FFT per channel: fast enough
                to follow a slider. Softer, and rings more, than the
                iterative engine.
``parametric``  Richardson-Lucy, the classic iterative maximum-likelihood
                deconvolution, with two additions: Biggs-Andrews vector
                acceleration (fewer iterations to the same result) and
                White's damping, which stops pixels whose residual is
                already at noise level from being "improved" into ringing.
``neural``      NAFNet trained on GoPro motion blur, run in overlapping
                tiles. Blind: it needs no PSF, but it only knows the kinds
                of blur it was trained on.

Units and conventions
---------------------
Images are float32, 0..1, H x W or H x W x C; every channel is processed
the same way. A motion PSF has a *length* in pixels and an *angle* in
degrees counter-clockwise from horizontal, as it looks on screen (the
convention ``blur_analysis`` reports in). A defocus PSF has a radius in
pixels.

Boundaries
----------
FFT deconvolution treats the image as periodic, so the jump between the
left and right (and top and bottom) edges is read as structure and rings
back across the frame. Two defences, used together: the image is extended
by reflection before any FFT, and :func:`edge_taper` blends the extended
edges into a circularly-blurred copy of themselves, so the periodic
image has no seam for the deconvolution to amplify.

What this cannot do
-------------------
Deconvolution recovers detail the blur attenuated, not detail it erased:
where the PSF's spectrum is zero, nothing survives and nothing comes back.
A long smear, a wrong angle, noise, or blur that differs across the frame
(a spinning wheel, a subject moving against a still background) all limit
it, and pushing it shows as ringing and amplified noise. The PSF is a
single global estimate; spatially varying blur is not modelled.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Callable

import cv2
import numpy as np
import scipy.fft as sfft

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

MOTION, DEFOCUS = "motion", "defocus"
MAX_LENGTH = 30.0
MAX_RADIUS = 20.0
DIP_SIGMAS = 5.0            # how far a cepstral dip must stand below the rest
EPS = 1e-6

# White's damping is defined on the residual between the photograph and the
# re-blurred estimate. ``damping`` 0..1 maps to a residual level of
# 0..DAMPING_SPAN in 0..1 intensity: 0.05 treats residuals under 0.5 % as
# noise (a little over one level in 255).
DAMPING_SPAN = 0.1
DAMPING_ORDER = 10          # N in White (1994); higher is a sharper cut-off

DECONV_TILE = 2048          # larger frames are deconvolved in tiles of this size
NEURAL_TILE, NEURAL_OVERLAP = 512, 64

NAFNET_SPEC = ModelSpec(
    filename="deblurring_nafnet_2025may.onnx",
    sha256="07263f416febecce10193dd648e950b22e397cf521eedab1a114ef77b2bc9587",
    size=91736251,
    zoo_dir="deblurring_nafnet",
    label="the AI de-blur model (NAFNet, GoPro motion blur)",
    url="https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx",
)


# --------------------------------------------------------------------- #
# Point spread functions
# --------------------------------------------------------------------- #

def _odd(n: float) -> int:
    n = int(math.ceil(n))
    return n if n % 2 else n + 1


def synthesize_motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """A straight-line motion PSF, anti-aliased, summing to 1.

    ``length`` is the smear's full extent in pixels (the distance the image
    travelled while the shutter was open); ``angle_deg`` is counter-clockwise
    from horizontal as seen on screen (image rows run downwards, so the
    line's vertical component is negated). The line is sampled densely and
    each sample is shared between its four nearest pixels, so a fractional
    length or an oblique angle gives a smooth kernel rather than a jagged
    one.
    """
    length = float(max(length, 1.0))
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), -math.sin(theta)
    size = _odd(length + 2)
    centre = (size - 1) / 2.0
    kernel = np.zeros((size, size), np.float64)
    samples = max(64, int(length * 16))
    for t in np.linspace(-length / 2.0, length / 2.0, samples):
        x, y = centre + t * dx, centre + t * dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for yy, xx, w in ((y0, x0, (1 - fx) * (1 - fy)), (y0, x0 + 1, fx * (1 - fy)),
                          (y0 + 1, x0, (1 - fx) * fy), (y0 + 1, x0 + 1, fx * fy)):
            if 0 <= yy < size and 0 <= xx < size:
                kernel[yy, xx] += w
    return (kernel / kernel.sum()).astype(np.float32)


def synthesize_defocus_kernel(radius: float) -> np.ndarray:
    """A uniform disc ("pillbox") PSF, anti-aliased at its rim, summing to 1.

    Each pixel's weight is the fraction of it the disc covers, measured on
    an 8 x 8 grid inside the pixel. A pillbox is the geometric-optics model
    of a defocused lens; the Airy pattern of diffraction is far smaller than
    any defocus worth correcting, so it is not modelled.
    """
    radius = float(max(radius, 0.5))
    size = _odd(2 * radius + 1)
    centre = (size - 1) / 2.0
    sub = (np.arange(8) + 0.5) / 8.0 - 0.5
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    py = (ys[:, :, None, None] + sub[None, None, :, None]) - centre
    px = (xs[:, :, None, None] + sub[None, None, None, :]) - centre
    kernel = ((px * px + py * py) <= radius * radius).mean(axis=(2, 3))
    return (kernel / kernel.sum()).astype(np.float32)


def kernel_for(blur_type: str, length: float = 5.0, angle: float = 0.0, radius: float = 3.0) -> np.ndarray:
    if blur_type == MOTION:
        return synthesize_motion_kernel(length, angle)
    if blur_type == DEFOCUS:
        return synthesize_defocus_kernel(radius)
    raise ValueError(f"Unknown blur type {blur_type!r}")


# --------------------------------------------------------------------- #
# Frequency-domain helpers
# --------------------------------------------------------------------- #

def _channels(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """H x W x C float32, and whether the input was 2-D."""
    array = np.asarray(image, dtype=np.float32)
    flat = array.ndim == 2
    return (array[:, :, None] if flat else array), flat


def psf_to_otf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The kernel's transfer function at ``shape``, centred on the origin."""
    padded = np.zeros(shape, np.float32)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return sfft.rfft2(padded, workers=-1)


def _convolve_circular(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Periodic convolution of an H x W x C image with an OTF of H x W."""
    spectrum = sfft.rfft2(image, axes=(0, 1), workers=-1)
    return sfft.irfft2(spectrum * otf[:, :, None], s=image.shape[:2], axes=(0, 1), workers=-1).astype(np.float32)


def _extend(image: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Reflect-pad by a margin the kernel needs, up to a fast FFT size."""
    margin = 2 * max(kernel.shape) + 8
    h, w = image.shape[:2]
    th, tw = sfft.next_fast_len(h + 2 * margin, real=True), sfft.next_fast_len(w + 2 * margin, real=True)
    top, left = (th - h) // 2, (tw - w) // 2
    padded = cv2.copyMakeBorder(image, top, th - h - top, left, tw - w - left, cv2.BORDER_REFLECT_101)
    if padded.ndim == 2:
        padded = padded[:, :, None]
    return padded.astype(np.float32, copy=False), top, left


def edge_taper(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blend the frame's borders into a circular blur of itself.

    The weight comes from the PSF's own autocorrelation: for each axis, the
    autocorrelation of the PSF's projection onto that axis is 1 at the edge
    of the frame and falls to 0 one kernel-width in. Near the border the
    result is the image blurred *circularly* by the PSF, which is exactly
    consistent with the periodic model FFT deconvolution assumes; in the
    interior it is the image untouched. This is the method of MATLAB's
    ``edgetaper``.
    """
    array, flat = _channels(image)
    h, w = array.shape[:2]
    weights = []
    for axis, n in ((1, h), (0, w)):
        projection = np.asarray(kernel, np.float64).sum(axis=axis)
        spectrum = np.fft.fft(projection, n)
        auto = np.real(np.fft.ifft(np.abs(spectrum) ** 2))
        weights.append(auto / auto.max())
    alpha = ((1.0 - weights[0])[:, None] * (1.0 - weights[1])[None, :]).astype(np.float32)
    blurred = _convolve_circular(array, psf_to_otf(kernel, (h, w)))
    out = alpha[:, :, None] * array + (1.0 - alpha[:, :, None]) * blurred
    return out[:, :, 0] if flat else out


# --------------------------------------------------------------------- #
# Wiener
# --------------------------------------------------------------------- #

def wiener_deconvolution(image: np.ndarray, kernel: np.ndarray, snr: float = 25.0,
                         taper: bool = True, extend: bool = True) -> np.ndarray:
    """Closed-form Wiener deconvolution: ``conj(H) Y / (|H|^2 + 1/snr)``.

    ``snr`` is the assumed signal-to-noise power ratio; lower values
    regularise harder (softer, less ringing). The frame is extended by
    reflection (``extend``) and its border tapered (``taper``) before the
    FFT. Turning the taper off saves a third of the time and is what the
    live preview does; turning both off deconvolves the frame as it is,
    periodic seams and all, which exists only so the boundary handling can
    be measured.
    """
    array, flat = _channels(image)
    h, w = array.shape[:2]
    if extend:
        work, top, left = _extend(array, kernel)
    else:
        work, top, left = array, 0, 0
    if taper:
        work = edge_taper(work, kernel)
    otf = psf_to_otf(kernel, work.shape[:2])
    gain = np.conj(otf) / (np.abs(otf) ** 2 + 1.0 / max(float(snr), 1e-3))
    spectrum = sfft.rfft2(work, axes=(0, 1), workers=-1) * gain[:, :, None]
    out = sfft.irfft2(spectrum, s=work.shape[:2], axes=(0, 1), workers=-1)[top:top + h, left:left + w]
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out[:, :, 0] if flat else out


# --------------------------------------------------------------------- #
# Richardson-Lucy
# --------------------------------------------------------------------- #

def _damping_weight(observed: np.ndarray, reblurred: np.ndarray, level: float) -> np.ndarray:
    """White's damping weight, 0 where the fit is within noise, 1 well outside it.

    ``u`` is the squared residual in units of ``level``; the weight is
    ``u^(N-1) (N - (N-1) u)`` below 1 and 1 above, a smooth step that keeps
    the update off pixels the estimate already explains.
    """
    u = observed - reblurred
    u *= 1.0 / level
    np.multiply(u, u, out=u)
    np.minimum(u, 1.0, out=u)
    power = u * u                       # u^2
    np.multiply(power, power, out=power)  # u^4
    np.multiply(power, power, out=power)  # u^8 (N - 1 = 9 with the next line)
    power *= u
    u *= -(DAMPING_ORDER - 1)
    u += DAMPING_ORDER
    power *= u
    return power


def richardson_lucy_damped(image: np.ndarray, kernel: np.ndarray, iterations: int = 15,
                           damping: float = 0.05) -> np.ndarray:
    """Accelerated, damped Richardson-Lucy deconvolution, float32 per channel.

    Each iteration re-blurs the estimate, compares it with the photograph,
    and multiplies the estimate by the back-projected ratio. Biggs-Andrews
    acceleration extrapolates along the direction the last two iterations
    moved, so fifteen iterations go about as far as forty plain ones.
    ``damping`` (0..1) sets the residual below which a pixel is treated as
    already fitted (White's damping); 0 is plain Richardson-Lucy. To keep
    divergence in check the estimate is also held inside the photograph's
    own range, a few per cent either side, which is where ringing first
    shows.
    """
    array, flat = _channels(image)
    h, w = array.shape[:2]
    padded, top, left = _extend(np.clip(array, 0.0, 1.0), kernel)
    padded = edge_taper(padded, kernel)
    # Channels first: each FFT then runs over one contiguous plane.
    observed = np.ascontiguousarray(np.maximum(padded, EPS).transpose(2, 0, 1))
    shape = observed.shape[1:]
    otf = psf_to_otf(kernel, shape)
    back = np.conj(otf)
    level = max(float(damping), 0.0) * DAMPING_SPAN
    flat_view = array.reshape(-1, array.shape[2])
    low = np.maximum(flat_view.min(axis=0) - 0.02, EPS).astype(np.float32)[:, None, None]
    high = np.minimum(flat_view.max(axis=0) * 1.04 + 0.01, 1.0).astype(np.float32)[:, None, None]

    def convolve(planes: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        spectrum = sfft.rfft2(planes, axes=(-2, -1), workers=-1)
        spectrum *= transfer
        return sfft.irfft2(spectrum, s=shape, axes=(-2, -1), workers=-1)

    estimate = observed.copy()
    y = estimate
    previous_step = None
    for _ in range(max(int(iterations), 0)):
        ratio = convolve(y, otf)
        np.maximum(ratio, EPS, out=ratio)
        if level > 0:
            weight = _damping_weight(observed, ratio, level)
            np.divide(observed, ratio, out=ratio)
            ratio -= 1.0
            ratio *= weight
            ratio += 1.0
        else:
            np.divide(observed, ratio, out=ratio)
        updated = convolve(ratio, back)
        updated *= y
        np.clip(updated, low, high, out=updated)
        step = updated - y
        alpha = 0.0
        if previous_step is not None:
            denominator = float(np.vdot(previous_step, previous_step))
            if denominator > 0:
                alpha = float(np.clip(np.vdot(step, previous_step) / denominator, 0.0, 0.95))
        if alpha:
            y = updated + alpha * (updated - estimate)
            np.clip(y, low, high, out=y)
        else:
            y = updated
        estimate = updated
        previous_step = step
    out = np.ascontiguousarray(estimate[:, top:top + h, left:left + w].transpose(1, 2, 0), np.float32)
    return out[:, :, 0] if flat else out


# --------------------------------------------------------------------- #
# Large frames
# --------------------------------------------------------------------- #

def deconvolve(image: np.ndarray, kernel: np.ndarray, engine: str = "parametric",
               damping: float = 0.05, iterations: int = 15, snr: float = 25.0,
               tile: int = DECONV_TILE, taper: bool = True) -> np.ndarray:
    """Deconvolve a frame of any size, a tile at a time.

    Each tile is processed with a margin of real neighbouring pixels three
    kernel-widths deep and only its centre is kept, so tiles meet without
    seams and memory stays at a few tiles' worth whatever the frame size.
    """
    if engine == "wiener":
        work = lambda block: wiener_deconvolution(block, kernel, snr, taper=taper)  # noqa: E731
    elif engine == "parametric":
        work = lambda block: richardson_lucy_damped(block, kernel, iterations, damping)  # noqa: E731
    else:
        raise ValueError(f"Unknown engine {engine!r}")
    h, w = image.shape[:2]
    if max(h, w) <= tile:
        return work(image)
    margin = 3 * max(kernel.shape) + 16
    out = np.empty(image.shape, np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            ya, xa = max(0, y0 - margin), max(0, x0 - margin)
            yb, xb = min(h, y1 + margin), min(w, x1 + margin)
            result = work(image[ya:yb, xa:xb])
            out[y0:y1, x0:x1] = result[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return out


def tiled(image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray],
          tile: int = NEURAL_TILE, overlap: int = NEURAL_OVERLAP) -> np.ndarray:
    """Apply ``fn`` in overlapping tiles, cross-fading across each overlap.

    Tiles are ``tile`` square and step ``tile - overlap``; in each overlap
    the two tiles' outputs are blended with linear ramps, so a model that
    behaves slightly differently near a tile's edge leaves no visible seam.
    The last row and column of tiles are shifted back to end at the frame
    edge rather than padded.
    """
    h, w = image.shape[:2]
    if h <= tile and w <= tile:
        return np.asarray(fn(image), np.float32)
    stride = tile - overlap

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        points = list(range(0, n - tile, stride)) + [n - tile]
        return sorted(set(points))

    ramp = np.ones(tile, np.float32)
    ramp[:overlap] = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    ramp[-overlap:] = ramp[:overlap][::-1]
    acc = np.zeros(image.shape[:2] + ((image.shape[2],) if image.ndim == 3 else ()), np.float32)
    weight = np.zeros(image.shape[:2], np.float32)
    for y0 in starts(h):
        for x0 in starts(w):
            th, tw = min(tile, h - y0), min(tile, w - x0)
            block = np.asarray(fn(image[y0:y0 + th, x0:x0 + tw]), np.float32)
            wy = ramp[:th] if th == tile else np.ones(th, np.float32)
            wx = ramp[:tw] if tw == tile else np.ones(tw, np.float32)
            # The frame's own edges are not overlaps: weight them fully.
            if y0 == 0:
                wy = wy.copy(); wy[:overlap] = 1.0  # noqa: E702
            if y0 + th == h:
                wy = wy.copy(); wy[-overlap:] = 1.0  # noqa: E702
            if x0 == 0:
                wx = wx.copy(); wx[:overlap] = 1.0  # noqa: E702
            if x0 + tw == w:
                wx = wx.copy(); wx[-overlap:] = 1.0  # noqa: E702
            mask = wy[:, None] * wx[None, :]
            acc[y0:y0 + th, x0:x0 + tw] += block * (mask[:, :, None] if block.ndim == 3 else mask)
            weight[y0:y0 + th, x0:x0 + tw] += mask
    weight = np.maximum(weight, EPS)
    return acc / (weight[:, :, None] if acc.ndim == 3 else weight)


# --------------------------------------------------------------------- #
# Estimating the blur
# --------------------------------------------------------------------- #

def _gray(image: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    """Grey 0..1, no larger than ``side``, and the factor it was scaled by."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)
    array = array.astype(np.float32)
    if array.max() > 1.5:
        array /= 255.0
    h, w = array.shape
    factor = 1.0
    if max(h, w) > side:
        factor = side / max(h, w)
        array = cv2.resize(array, (max(1, round(w * factor)), max(1, round(h * factor))),
                           interpolation=cv2.INTER_AREA)
    return array, factor


def _log_spectrum(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    centred = (gray - gray.mean()) * window
    return np.log(np.abs(np.fft.fft2(centred)) + 1e-3)


def estimate_motion_length(image: np.ndarray, angle_deg: float, side: int = 1024,
                           max_length: float = MAX_LENGTH) -> float | None:
    """The length of a linear motion smear, from the image's cepstrum.

    A uniform smear of length L multiplies the spectrum by a sinc whose
    zeros repeat every N / L along the direction of motion; in the
    cepstrum (the inverse transform of the log spectrum) that periodicity
    becomes a strong negative peak at quefrency L. The frame is turned so the
    motion runs along rows, the log spectra of the rows are averaged, and
    the deepest dip between 3 px and ``max_length`` is taken. Returns the
    length in pixels of ``image`` (it is measured on a copy at most ``side``
    across), or None when no dip stands clear.

    On synthetic tests it is exact from about 8 px; shorter smears, and
    frames with strongly repeating texture, can give a wrong reading, so it
    is only consulted once ``blur_analysis`` has called the frame
    motion-blurred.
    """
    gray, factor = _gray(image, side)
    h, w = gray.shape
    turn = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(angle_deg), 1.0)
    level = cv2.warpAffine(gray, turn, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Keep the middle, which every rotation fills with real pixels.
    level = level[h // 4: h - h // 4, w // 4: w - w // 4]
    rows = level - level.mean(axis=1, keepdims=True)
    window = np.hanning(rows.shape[1]).astype(np.float32)
    spectrum = np.log(np.abs(np.fft.rfft(rows * window, axis=1)) + 1e-3).mean(axis=0)
    cepstrum = np.fft.irfft(spectrum - spectrum.mean())
    return _dip(cepstrum, 3, int(max_length * factor) + 4, lambda q: q / factor, max_length)


def _dip(profile: np.ndarray, lo: int, hi: int, to_size, limit: float) -> float | None:
    """The deepest dip in ``profile[lo:hi]``, if it stands clear of the rest.

    "Clear" is judged against the spread of the profile's tail, measured by
    the median absolute deviation so that the dip itself and the large
    values near zero quefrency do not inflate it.
    """
    hi = min(hi, len(profile) // 2 - 1)
    if hi <= lo:
        return None
    segment = profile[lo:hi + 1]
    index = int(np.argmin(segment))
    tail = profile[lo:min(len(profile) // 2, hi * 3)]
    spread = 1.4826 * float(np.median(np.abs(tail - np.median(tail)))) or 1e-6
    if segment[index] > np.median(tail) - DIP_SIGMAS * spread:
        return None
    # Sub-sample position of the dip, from the parabola through it and its
    # neighbours: the copy is measured reduced, so whole steps are coarse.
    q = lo + index
    offset = 0.0
    if 0 < q < len(profile) - 1:
        left, mid, right = profile[q - 1], profile[q], profile[q + 1]
        curvature = left - 2.0 * mid + right
        if curvature > 1e-12:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(np.clip(to_size(q + offset), 1.0, limit))


def estimate_defocus_radius(image: np.ndarray, side: int = 1024, max_radius: float = MAX_RADIUS) -> float | None:
    """The radius of a defocus disc, from the radial profile of the cepstrum.

    A pillbox's spectrum has rings of zeros; their spacing puts a negative
    ring in the cepstrum at 2R - 1 pixels (calibrated on synthetic discs of
    radius 3 to 12, see the tests). Below a radius of about 3 the reading is
    unreliable. Returns pixels of ``image``, or None.
    """
    gray, factor = _gray(image, side)
    cepstrum = np.fft.fftshift(np.real(np.fft.ifft2(_log_spectrum(gray))))
    h, w = cepstrum.shape
    ys, xs = np.indices(cepstrum.shape)
    r = np.hypot(ys - h // 2, xs - w // 2).astype(np.int32).ravel()
    profile = np.bincount(r, cepstrum.ravel()) / np.maximum(np.bincount(r), 1)
    hi = int(2 * max_radius * factor) + 2
    return _dip(profile, 4, hi, lambda q: (q + 1.0) / 2.0 / factor, max_radius)


# --------------------------------------------------------------------- #
# Neural
# --------------------------------------------------------------------- #

class TiledNeuralDeblur:
    """NAFNet (GoPro) de-blurring, run in overlapping 512 px tiles.

    The model works on display-encoded RGB, 0..1, and needs no PSF. Tiling
    keeps memory at one tile's activations whatever the frame size, so a
    36-megapixel frame runs on integrated graphics or the CPU. It is slow:
    about 11 s per 512 px tile on a laptop CPU (i5-8350U), 136 s for a
    1600 px preview; OpenCV's OpenCL target is no faster. Measured on a
    photograph it gains 0.8 to 2 dB on 9 to 25 px smears and loses a little
    on a 5 px one; flat synthetic graphics are outside its training and it
    distorts them. The model is a 92 MB download,
    fetched only with consent; without it :attr:`available` is False and
    nothing pretends to de-blur.
    """

    def __init__(self, download: bool = False) -> None:
        self._net = None
        self._lock = threading.Lock()
        path = model_store.ensure_model(NAFNET_SPEC, download=download)
        if path is not None:
            try:
                self._net = cv2.dnn.readNet(str(path))
            except Exception as exc:
                logger.warning("The de-blur model failed to load: %s", exc)

    @staticmethod
    def is_cached() -> bool:
        return model_store.is_cached(NAFNET_SPEC)

    @property
    def available(self) -> bool:
        return self._net is not None

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        # The network halves the frame four times; pad to a multiple of 16.
        ph, pw = (-h) % 16, (-w) % 16
        padded = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else rgb
        blob = cv2.dnn.blobFromImage(np.ascontiguousarray(padded, np.float32), 1.0, swapRB=False, crop=False)
        with self._lock:
            self._net.setInput(blob)
            out = self._net.forward()
        result = np.transpose(np.asarray(out)[0], (1, 2, 0))[:h, :w]
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def restore(self, rgb: np.ndarray, progress: Callable[[], None] | None = None) -> np.ndarray:
        """De-blur display-encoded RGB (float 0..1 or uint8)."""
        if not self.available:
            raise RuntimeError("the AI de-blur model is not available")
        array = np.asarray(rgb)
        scale = 255.0 if array.dtype == np.uint8 else 1.0
        work = array[:, :, :3].astype(np.float32) / scale

        def run(block: np.ndarray) -> np.ndarray:
            result = self._infer(block)
            if progress:
                progress()
            return result

        out = tiled(work, run)
        return np.round(out * 255.0).astype(np.uint8) if scale == 255.0 else out
