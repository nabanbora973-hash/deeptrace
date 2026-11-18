"""Temporal evidence stream.

A face-swap is generated frame by frame, so it is temporally *incoherent* in
ways a real recording is not:

* flicker - the per-frame artifact score wobbles instead of holding steady,
* identity jitter - the face descriptor jumps between consecutive frames,
* missing blood-pulse - remote photoplethysmography (rPPG) can recover a heart
  rate from real skin colour changes; generated skin has no pulse.

The stream also reports how much temporal support the input actually gives
(short clips cannot support pulse evidence), and that support down-weights the
stream during fusion.  Absence of evidence is never treated as evidence.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np

EPS = 1e-9
PULSE_BAND = (0.7, 3.5)   # Hz -> 42-210 bpm
NOISE_BAND = (0.4, 8.0)


def _robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _descriptor(crop_bgr: np.ndarray, size: int = 16) -> np.ndarray:
    """Low-frequency appearance descriptor used for identity-jitter measurement."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    v = small.ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / (n + EPS)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _pulse_from_signal(signal: np.ndarray, timestamps: Sequence[float]) -> Dict[str, float]:
    """Estimate rPPG pulse strength / rate from a per-frame colour-ROI signal."""
    t = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 24 or t.size != n:
        return {"pulse_strength": 0.0, "pulse_bpm": 0.0, "pulse_snr": 0.0, "support": 0.25}

    # uniform resampling (videos are usually uniform, but be safe)
    fps = 1.0 / max(np.median(np.diff(t)), 1e-3)
    t_u = np.arange(0, t[-1], 1.0 / fps)
    if t_u.size < 24:
        return {"pulse_strength": 0.0, "pulse_bpm": 0.0, "pulse_snr": 0.0, "support": 0.35}
    x_u = np.interp(t_u, t, x)

    # detrend with a moving average (removes illumination drift)
    win = max(3, int(fps * 1.2) | 1)
    kernel = np.ones(win) / win
    trend = np.convolve(np.pad(x_u, win // 2, mode="edge"), kernel, mode="valid")[: x_u.size]
    resid = x_u - trend
    if resid.std() < 1e-9:
        return {"pulse_strength": 0.0, "pulse_bpm": 0.0, "pulse_snr": 0.0, "support": 0.4}

    resid = (resid - resid.mean()) * np.hanning(resid.size)
    freqs = np.fft.rfftfreq(resid.size, d=1.0 / fps)
    psd = np.abs(np.fft.rfft(resid)) ** 2

    band = (freqs >= PULSE_BAND[0]) & (freqs <= PULSE_BAND[1])
    noise = (freqs >= NOISE_BAND[0]) & (freqs <= NOISE_BAND[1]) & ~band
    if not band.any():
        return {"pulse_strength": 0.0, "pulse_bpm": 0.0, "pulse_snr": 0.0, "support": 0.4}

    peak_idx = np.argmax(psd[band])
    peak_power = float(psd[band][peak_idx])
    total_power = float(psd[band].sum()) + EPS
    noise_floor = float(np.median(psd[noise])) + EPS if noise.any() else EPS
    snr = peak_power / noise_floor

    # concentration of band power in the dominant peak = pulse likelihood
    strength = float(np.clip((peak_power / total_power) * min(snr / 8.0, 1.5), 0.0, 1.0))
    bpm = float(freqs[band][peak_idx] * 60.0)
    support = float(np.clip(resid.size / 96.0, 0.3, 1.0))
    return {"pulse_strength": strength, "pulse_bpm": bpm, "pulse_snr": float(snr), "support": support}


def analyze_temporal(
    crops: List[np.ndarray],
    seam_series: Sequence[float],
    timestamps: Sequence[float],
    fps: float,
) -> Dict[str, object]:
    n = len(crops)
    if n == 0:
        raise ValueError("temporal stream needs at least one crop")

    # --- 1. flicker: instability of the per-frame artifact score -------------
    seam = np.asarray(seam_series, dtype=np.float64)
    if seam.size >= 3:
        flicker = float(_robust_sd(seam) / (np.median(np.abs(seam)) + 0.02))
    else:
        flicker = 0.0
    flicker = float(np.clip(flicker, 0.0, 3.0))

    # --- 2. identity jitter --------------------------------------------------
    descs = [_descriptor(c) for c in crops]
    if len(descs) >= 3:
        sims = np.array([_ncc(descs[i], descs[i + 1]) for i in range(len(descs) - 1)])
        mean_sim = float(sims.mean())
        jitter = float(np.clip((1.0 - mean_sim) * 0.6 + _robust_sd(sims) * 3.0, 0.0, 3.0))
    else:
        mean_sim, jitter = 1.0, 0.0

    # --- 3. rPPG pulse ------------------------------------------------------
    size = crops[0].shape[0]
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2.0)
    roi = r < 0.55
    green = np.array([c[:, :, 1].astype(np.float32)[roi].mean() for c in crops], dtype=np.float64)
    pulse = _pulse_from_signal(green, timestamps)
    # A BPM estimate without supporting power is noise, not a finding.
    if pulse["pulse_strength"] < 0.05:
        pulse["pulse_bpm"] = 0.0

    # --- 4. temporal variance map (per-pixel instability) -------------------
    if n >= 3:
        stack = np.stack(
            [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32) for c in crops], axis=0
        )
        tvar = stack.std(axis=0)
        tvar = cv2.GaussianBlur(tvar, (0, 0), 2.0)
        tvar = tvar / (np.percentile(tvar, 99) + EPS)
        temporal_map = np.clip(tvar, 0.0, 1.0).astype(np.float32)
    else:
        temporal_map = np.zeros((size, size), dtype=np.float32)

    # temporal support: how much a clip can actually prove
    support = float(np.clip(min(n / 24.0, fps / 12.0 if fps else 0.5), 0.15, 1.0)) * pulse["support"] ** 0.5
    support = float(np.clip(support, 0.1, 1.0))

    features = {
        "flicker": flicker,
        "identity_jitter": jitter,
        "pulse_strength": float(pulse["pulse_strength"]),
        "pulse_bpm": float(pulse["pulse_bpm"]),
    }
    stats = {
        "frames_used": n,
        "mean_descriptor_similarity": mean_sim,
        "pulse_snr": pulse["pulse_snr"],
        "temporal_support": support,
        "seam_series": [round(float(v), 5) for v in seam.tolist()],
    }
    return {"features": features, "stats": stats, "map": temporal_map, "support": support}
