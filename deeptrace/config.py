"""Configuration and calibration model for DeepTrace.

The detector fuses three independent forensic evidence streams:

    spatial   - blending seams, noise-residual inconsistency, texture/colour mismatch
    frequency - GAN spectral fingerprints, high-frequency energy, DCT blockiness
    temporal  - frame flicker, identity jitter, missing rPPG blood-pulse

Every number used to turn features into a verdict lives in a versioned
calibration document (``data/calibration.json``).  Nothing is hidden in code,
so a reviewer can audit - and refit - the decision boundary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent

# Paths are overridable so the app can run on hosts with a read-only image and a
# single writable volume (Render disks, Cloud Run /tmp, Hugging Face Spaces).
DATA_DIR = Path(os.environ.get("DEEPTRACE_DATA_DIR", PROJECT_ROOT / "data"))
WEB_DIR = PROJECT_ROOT / "web"
SAMPLE_DIR = Path(os.environ.get("DEEPTRACE_SAMPLE_DIR", PROJECT_ROOT / "samples"))
WORK_DIR = Path(os.environ.get("DEEPTRACE_WORK_DIR", DATA_DIR / "runs"))
CALIBRATION_PATH = Path(os.environ.get("DEEPTRACE_CALIBRATION", DATA_DIR / "calibration.json"))

APP_NAME = "DeepTrace"
APP_VERSION = "1.2.1"
APP_TAGLINE = ("Multi-signal face-swap forensics - three streams of evidence per face, "
               "one explainable verdict per clip.")

# ---------------------------------------------------------------------------
# Analysis defaults (all overridable per request)
# ---------------------------------------------------------------------------
MAX_FRAMES = int(os.environ.get("DEEPTRACE_MAX_FRAMES", 48))   # frames sampled per analysis
MIN_FRAMES_TEMPORAL = 12   # below this, the temporal stream is down-weighted
CROP_SIZE = 256            # aligned face crop resolution used by all streams

# Multi-face analysis: every detected face is tracked and scored separately.
MAX_FACES = int(os.environ.get("DEEPTRACE_MAX_FACES", 3))          # faces analysed per clip
MIN_TRACK_FRAMES = int(os.environ.get("DEEPTRACE_MIN_TRACK_FRAMES", 4))  # hits needed to score a face
TRACK_MAX_GAP = int(os.environ.get("DEEPTRACE_TRACK_MAX_GAP", 3))  # frames a face may vanish and return
MAX_UPLOAD_BYTES = int(float(os.environ.get("DEEPTRACE_MAX_UPLOAD_MB", 60)) * 1024 * 1024)

# Verdict thresholds on the calibrated fake-probability.
FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35

STREAM_NAMES = ("spatial", "frequency", "temporal")

DEFAULT_CALIBRATION: Dict[str, Any] = {
    "version": "1.0-prior",
    "kind": "expert-prior",
    "notes": (
        "Initial hand-set priors. Refit on labelled data with "
        "train/fit_calibration.py; the JSON schema is identical, so the "
        "runtime needs no code change."
    ),
    "thresholds": {"fake": FAKE_THRESHOLD, "real": REAL_THRESHOLD},
    "fusion": {"bias": -0.20, "weights": {"spatial": 1.30, "frequency": 1.05, "temporal": 0.95}},
    "streams": {
        "spatial": {
            "bias": -0.15,
            "features": {
                # name: (prior mean, prior sd, coefficient: + pushes toward FAKE)
                "seam_contrast": {"mean": 0.07, "sd": 0.11, "coef": 1.15},
                "noise_inconsistency": {"mean": 0.12, "sd": 0.20, "coef": 0.90},
                "colour_mismatch": {"mean": 0.05, "sd": 0.09, "coef": 0.70},
                "sharpness_mismatch": {"mean": 0.12, "sd": 0.28, "coef": 0.55},
            },
        },
        "frequency": {
            "bias": -0.15,
            "features": {
                "hf_ratio": {"mean": 0.050, "sd": 0.045, "coef": 1.05},
                "peak_prominence": {"mean": 0.60, "sd": 0.85, "coef": 0.85},
                "angular_peakiness": {"mean": 0.32, "sd": 0.22, "coef": 0.60},
                "spectral_slope": {"mean": -2.60, "sd": 0.90, "coef": -0.50},
                "dct_blockiness": {"mean": 0.10, "sd": 0.12, "coef": 0.50},
            },
        },
        "temporal": {
            "bias": -0.15,
            "features": {
                "flicker": {"mean": 0.12, "sd": 0.15, "coef": 1.00},
                "identity_jitter": {"mean": 0.09, "sd": 0.13, "coef": 0.95},
                "pulse_strength": {"mean": 0.35, "sd": 0.25, "coef": -0.90},
                "pulse_bpm": {"mean": 0.0, "sd": 1.0, "coef": 0.0},
            },
        },
    },
    "evidence_labels": {
        "seam_contrast": "Blending seam at the face boundary",
        "noise_inconsistency": "Sensor-noise profile breaks across the face region",
        "colour_mismatch": "Colour/chroma mismatch between swapped face and background",
        "sharpness_mismatch": "Sharpness discontinuity between face and surroundings",
        "hf_ratio": "Excess high-frequency energy (generator upsampling)",
        "peak_prominence": "Periodic spectral peaks (GAN fingerprint)",
        "angular_peakiness": "Checkerboard-style angular spectral artefacts",
        "spectral_slope": "Atypical spectral fall-off",
        "dct_blockiness": "DCT block high-frequency anomalies",
        "flicker": "Frame-to-frame flicker in the face region",
        "identity_jitter": "Identity jitter / unstable face geometry",
        "pulse_strength": "No recoverable blood-pulse (rPPG) from facial skin",
        "pulse_bpm": "Estimated pulse rate",
    },
}


@dataclass
class Settings:
    """Runtime settings for a single analysis run."""

    max_frames: int = MAX_FRAMES
    crop_size: int = CROP_SIZE
    keep_artifacts: bool = True
    max_faces: int = MAX_FACES
    min_track_frames: int = MIN_TRACK_FRAMES
    track_max_gap: int = TRACK_MAX_GAP
    calibration: Dict[str, Any] = field(default_factory=lambda: load_calibration().raw)

    @property
    def fake_threshold(self) -> float:
        return float(self.calibration.get("thresholds", {}).get("fake", FAKE_THRESHOLD))

    @property
    def real_threshold(self) -> float:
        return float(self.calibration.get("thresholds", {}).get("real", REAL_THRESHOLD))


@dataclass
class Calibration:
    raw: Dict[str, Any]
    path: Path | None = None
    source: str = "default"


def load_calibration(path: str | Path | None = None) -> Calibration:
    """Load a calibration document, falling back to the built-in prior."""
    target = Path(path) if path else CALIBRATION_PATH
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if "streams" in raw and "fusion" in raw:
                return Calibration(raw=raw, path=target, source="file")
        except (OSError, json.JSONDecodeError):
            pass
    return Calibration(raw=json.loads(json.dumps(DEFAULT_CALIBRATION)), path=target, source="default")


def save_calibration(raw: Dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path) if path else CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return target


def ensure_dirs() -> None:
    for d in (DATA_DIR, WORK_DIR, SAMPLE_DIR, WEB_DIR):
        os.makedirs(d, exist_ok=True)
