"""Spatial evidence stream.

Detects what a face-swap leaves in the image plane:

* a blending seam around the pasted face boundary (gradient ridge),
* an inconsistent sensor-noise residual between the swapped region and the rest,
* colour/chroma mismatch between the swapped face and its surroundings,
* sharpness discontinuity (swaps are resampled twice and end up softer).

All features are computed on aligned 256x256 face crops.  The stream also
returns a per-pixel "artifact energy" map that the explainability layer renders
as a heat-map - so a finding is never a bare number.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

EPS = 1e-9


def _radius_grid(size: int) -> np.ndarray:
    """Normalised radius (0 at centre, 1.0 at the shortest edge)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2.0)


def _robust_sd(values: np.ndarray) -> float:
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def _highpass(gray: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return gray - blur


def _noise_map(gray: np.ndarray, block: int = 8) -> np.ndarray:
    """Block-wise standard deviation of the high-pass residual, upsampled."""
    res = _highpass(gray)
    h, w = res.shape
    gh, gw = h // block, w // block
    trimmed = res[:gh * block, :gw * block]
    blocks = trimmed.reshape(gh, block, gw, block).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    std = blocks.std(axis=2)
    return cv2.resize(std.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def analyze_spatial(crop_bgr: np.ndarray) -> Dict[str, object]:
    """Return spatial features, a scalar artifact map and supporting stats."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    size = gray.shape[0]
    radius = _radius_grid(size)

    # --- 1. blending seam: radial gradient profile ---------------------------
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    bins = np.linspace(0.0, 1.0, 26)
    prof = np.zeros(len(bins) - 1, dtype=np.float32)
    for i in range(len(bins) - 1):
        m = (radius >= bins[i]) & (radius < bins[i + 1])
        prof[i] = float(grad[m].mean()) if m.any() else 0.0
    base = float(np.median(prof)) + EPS
    # the swap boundary sits near r ~ 0.70 because crops are expanded ~22%
    seam_band = prof[(bins[:-1] >= 0.60) & (bins[:-1] < 0.82)]
    seam_contrast = float((seam_band.max() - np.median(prof)) / base) if seam_band.size else 0.0
    seam_contrast = float(np.clip(seam_contrast, 0.0, 5.0))

    # --- 2. noise residual consistency --------------------------------------
    nmap = _noise_map(gray, block=8)
    inner = nmap[radius < 0.55]
    outer = nmap[(radius > 0.80) & (radius < 0.98)]
    n_in = float(inner.mean()) if inner.size else 0.0
    n_out = float(outer.mean()) if outer.size else 0.0
    noise_inconsistency = float(abs(n_out - n_in) / (n_out + n_in + EPS))
    noise_cv = float(_robust_sd(nmap.ravel()) / (nmap.mean() + EPS))
    # scale-free texture-uniformity ratio: a pasted face has unnaturally uniform
    # noise statistics inside it relative to the surrounding real scene
    inner_noise = nmap[radius < 0.55]
    outer_noise = nmap[(radius > 0.80) & (radius < 0.98)]
    cv_in_raw = float(inner_noise.std() / (inner_noise.mean() + EPS)) if inner_noise.size else 1.0
    cv_out_raw = float(outer_noise.std() / (outer_noise.mean() + EPS)) if outer_noise.size else 1.0
    noise_uniformity = float(cv_in_raw / (cv_out_raw + EPS))

    # --- 3. colour / chroma mismatch ----------------------------------------
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m_inner = (radius < 0.55)
    m_outer = (radius > 0.80) & (radius < 0.98)
    ab_in = lab[:, :, 1:3][m_inner].mean(axis=0)
    ab_out = lab[:, :, 1:3][m_outer].mean(axis=0)
    colour_mismatch = float(np.linalg.norm(ab_in - ab_out) / 128.0)
    chroma_inner = float(lab[:, :, 1:3][m_inner].std())

    # --- 4. sharpness discontinuity -----------------------------------------
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lvar_in = float(lap[m_inner].var())
    lvar_out = float(lap[m_outer].var())
    sharpness_mismatch = float(abs(np.log((lvar_in + EPS) / (lvar_out + EPS))))
    sharpness_mismatch = float(min(sharpness_mismatch, 4.0))

    features = {
        "seam_contrast": seam_contrast,
        "noise_uniformity": noise_uniformity,
        "noise_inconsistency": noise_inconsistency,
        "colour_mismatch": colour_mismatch,
        "sharpness_mismatch": sharpness_mismatch,
    }
    stats = {
        "noise_center": n_in,
        "noise_border": n_out,
        "noise_block_cv": noise_cv,
        "chroma_center": chroma_inner,
        "laplacian_var_center": lvar_in,
        "laplacian_var_border": lvar_out,
        "radial_gradient_profile": [round(float(v), 5) for v in prof],
    }

    # artifact map: normalised noise residual + gradient ridge
    n_norm = nmap / (nmap.max() + EPS)
    g_norm = grad / (np.percentile(grad, 99) + EPS)
    artifact_map = np.clip(0.65 * n_norm + 0.35 * g_norm, 0.0, 1.0).astype(np.float32)
    return {"features": features, "stats": stats, "map": artifact_map}
