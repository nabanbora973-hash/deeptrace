"""Video / image ingest and frame sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FrameSet:
    """Frames sampled from one media file, plus provenance metadata."""

    frames: List[np.ndarray]
    indices: List[int]          # original frame index in the source stream
    timestamps: List[float]     # seconds
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float
    kind: str                   # "video" | "image"
    path: Path
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.frames)


class IngestError(RuntimeError):
    """Raised when a media file cannot be decoded."""


def sample_frames(path: str | Path, max_frames: int = 48, min_frames: int = 8) -> FrameSet:
    """Read a video/image and return an evenly spaced frame sample.

    Sampling is deterministic: frame indices are spread across the clip so the
    temporal stream sees the whole timeline rather than the opening seconds.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXT:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IngestError(f"could not decode image: {path.name}")
        h, w = img.shape[:2]
        return FrameSet(
            frames=[img], indices=[0], timestamps=[0.0], fps=1.0, total_frames=1,
            width=w, height=h, duration=0.0, kind="image", path=path,
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestError(f"could not open video: {path.name}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0.01:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: List[np.ndarray] = []
    indices: List[int] = []

    if total > 0:
        n = int(min(max_frames, total))
        targets = np.unique(np.linspace(0, total - 1, n).astype(int))
        for idx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
                indices.append(int(idx))
    if len(frames) < min_frames:
        # Fallback: sequential decode until we have enough frames.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames, indices = [], []
        step = 1
        pos = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if pos % step == 0:
                frames.append(frame)
                indices.append(pos)
            pos += 1
    cap.release()

    if not frames:
        raise IngestError(f"no decodable frames in {path.name}")
    if width == 0 or height == 0:
        height, width = frames[0].shape[:2]

    duration = (total / fps) if total else (len(frames) / fps)
    return FrameSet(
        frames=frames,
        indices=indices,
        timestamps=[i / fps for i in indices],
        fps=fps,
        total_frames=total or len(frames),
        width=width,
        height=height,
        duration=duration,
        kind="video",
        path=path,
        meta={"sampled": float(len(frames))},
    )
