"""Frequency-domain evidence stream.

Generative upsampling leaves a measurable fingerprint in the spectrum:

* excess high-frequency energy (the network cannot learn a true 1/f fall-off),
* periodic peaks - the classic GAN "checkerboard" grid,
* angular concentration of energy at grid angles,
* anomalous DCT block statistics at the 8x8 JPEG grid.

These survive neither naive brightness changes nor mild colour grading, which is
why they are fused with the spatial and temporal streams instead of trusted alone.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

EPS = 1e-9


def _hann_window(size: int, alpha: float = 0.5) -> np.ndarray:
    """Power-of-two friendly window (scipy-free Tukey/Hann hybrid)."""
    n = np.arange(size, dtype=np.float32)
    w = np.ones(size, dtype=np.float32)
    edge = max(2, int(size * alpha / 2))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    w[:edge] = ramp
    w[-edge:] = ramp[::-1]
    return np.outer(w, w)


def _radial_profile(power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    size = power.shape[0]
    cy = cx = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = r.astype(np.int32)
    max_r = int(r_int.max()) + 1
    total = np.bincount(r_int.ravel(), weights=power.ravel(), minlength=max_r)
    count = np.bincount(r_int.ravel(), minlength=max_r).astype(np.float32)
    count[count == 0] = 1
    return total / count, r


def analyze_frequency(crop_bgr: np.ndarray) -> Dict[str, object]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    size = gray.shape[0]
    windowed = (gray - gray.mean()) * _hann_window(size)
    spec = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(spec) ** 2
    logmag = np.log10(power + EPS)

    profile, radius = _radial_profile(power)
    nyq = size / 2.0

    # --- 1. high-frequency energy ratio -------------------------------------
    total_energy = float(power.sum()) + EPS
    mask_all_but_dc = radius > 2.0
    hf_mask = radius > 0.35 * nyq
    hf_ratio = float(power[hf_mask].sum() / (power[mask_all_but_dc].sum() + EPS))

    # --- 2. spectral slope (log power vs log radius) ------------------------
    lo, hi = 4, int(nyq * 0.55)
    r_axis = np.arange(lo, hi, dtype=np.float32)
    p_axis = profile[lo:hi]
    good = p_axis > 0
    if good.sum() > 8:
        slope, _ = np.polyfit(np.log10(r_axis[good]), np.log10(p_axis[good]), 1)
        spectral_slope = float(slope)
    else:
        spectral_slope = 0.0

    # --- 3. periodic peak prominence (GAN grid) -----------------------------
    axis_band = 3
    cy = cx = size // 2
    interior = np.ones_like(logmag, dtype=bool)
    interior[cy - axis_band:cy + axis_band, :] = False
    interior[:, cx - axis_band:cx + axis_band] = False
    interior[:cy - 12, :] = False
    interior[cy + 12:, :] = False
    interior[:, :cx - 12] = False
    interior[:, cx + 12:] = False
    spec_pixels = logmag[interior]
    if spec_pixels.size:
        peak = float(np.percentile(spec_pixels, 99.98))
        bulk = float(np.percentile(spec_pixels, 75))
        spread = float(np.percentile(spec_pixels, 90) - np.percentile(spec_pixels, 10)) + EPS
        peak_prominence = float(np.clip((peak - bulk) / spread, 0.0, 6.0))
    else:
        peak_prominence = 0.0

    # --- 4. angular concentration in the high band --------------------------
    high_band = power[radius > 0.5 * nyq]
    if high_band.size:
        angles = np.arctan2(*np.mgrid[0:size, 0:size])[radius > 0.5 * nyq]
        nb = 72
        idx = ((angles + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        bins = np.bincount(idx, weights=high_band, minlength=nb) + EPS
        angular_peakiness = float(np.clip(bins.max() / bins.mean() - 1.0, 0.0, 8.0))
    else:
        angular_peakiness = 0.0

    # --- 5. DCT block statistics -------------------------------------------
    dct_energy = []
    block = 8
    gh, gw = size // block, size // block
    for by in range(gh):
        for bx in range(gw):
            patch = gray[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            d = cv2.dct(patch.astype(np.float32))
            ac = np.abs(d).copy()
            ac[0, 0] = 0.0
            denom = ac.sum() + EPS
            iu, iv = np.mgrid[0:block, 0:block]
            high = ac[(iu + iv) >= 6].sum()
            dct_energy.append(high / denom)
    dct_energy = np.asarray(dct_energy, dtype=np.float32)
    dct_blockiness = float(dct_energy.mean())
    dct_dispersion = float(np.std(dct_energy))

    features = {
        "hf_ratio": hf_ratio,
        "peak_prominence": peak_prominence,
        "angular_peakiness": angular_peakiness,
        "spectral_slope": spectral_slope,
        "dct_blockiness": dct_blockiness,
    }
    stats = {
        "hf_ratio": hf_ratio,
        "spectral_slope": spectral_slope,
        "dct_dispersion": dct_dispersion,
        "radial_profile": [round(float(v), 8) for v in profile[: min(64, len(profile))]],
    }

    # spatial map of high-frequency energy: reconstruct the high-pass band
    high_pass_spec = spec * (radius > 0.22 * nyq)
    energy_map = np.abs(np.fft.ifft2(np.fft.ifftshift(high_pass_spec)))
    energy_map = cv2.GaussianBlur(energy_map.astype(np.float32), (0, 0), 2.0)
    energy_map /= (energy_map.max() + EPS)
    return {"features": features, "stats": stats, "map": energy_map.astype(np.float32)}
