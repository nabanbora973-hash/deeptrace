"""Multi-face detection bookkeeping and identity tracking.

A single "dominant face" assumption breaks the moment two people appear in the
frame: a swapped face on the right can sit next to a genuine face on the left,
and averaging them hides both.  This module keeps the frames' faces *separate*:

* :class:`Detection` - one face box found in one sampled frame,
* :class:`Track`     - one face followed across the sampled timeline,
* :class:`FaceTracker` - greedy but deterministic association between them.

Association is deliberately simple and auditable (no learned embedding):

    cost = normalised_centre_distance
         + 0.55 * |log(size ratio)|
         - 0.90 * IoU
         - 0.25 * appearance_similarity

with hard gates on displacement, size ratio and low-frequency appearance
similarity, so two different people cannot be merged into one identity.  Frames
where a face is temporarily missed are bridged (``max_gap``), which is what turns
"detected in 31 of 48 frames" into one continuous track instead of 31 singletons.

Tracks with too few hits are reported, never silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]
EPS = 1e-9

# Gates: a candidate match beyond these is treated as a different face.
MAX_CENTRE_DIST = 1.40      # in units of the mean box side
MAX_SIZE_RATIO = math.log(2.20)
# Appearance is a *tie-breaker*, not a gate: OpenCV detectors routinely return
# a tight box on one frame and a loose box on the next, and the two crops then
# correlate poorly even though they are the same person.  So a low similarity
# only overrules a large geometric displacement.
APPEARANCE_VETO = -0.25     # cosine similarity below this = different-looking face
APPEARANCE_VETO_DIST = 0.60  # ...and the face must also have moved this much


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def box_area(box: Sequence[float]) -> float:
    return float(max(0.0, box[2]) * max(0.0, box[3]))


def box_side(box: Sequence[float]) -> float:
    """Geometric-mean side length - stable when the box is not square."""
    return float(math.sqrt(box_area(box)) + EPS)


def box_centre(box: Sequence[float]) -> Tuple[float, float]:
    return (float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, aw, ah = a[0], a[1], a[2], a[3]
    bx0, by0, bw, bh = b[0], b[1], b[2], b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > EPS else 0.0


def descriptor(crop_bgr: np.ndarray, size: int = 16) -> np.ndarray:
    """Cheap low-frequency appearance descriptor (unit-norm, mean removed)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    v = small.ravel()
    v = v - v.mean()
    return v / (float(np.linalg.norm(v)) + EPS)


# ---------------------------------------------------------------------------
# detections and tracks
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One face box found in one sampled frame."""

    box: Box
    score: float = 1.0
    kind: str = "face"          # "face" | "fallback" (centred window, no detection)


@dataclass
class Track:
    """One face followed across the sampled timeline."""

    track_id: int
    kind: str = "face"
    boxes: Dict[int, Box] = field(default_factory=dict)        # frame position -> box
    scores: Dict[int, float] = field(default_factory=dict)
    descriptors: Dict[int, np.ndarray] = field(default_factory=dict)

    # -- basic bookkeeping ------------------------------------------------
    def add(self, pos: int, det: Detection, desc: Optional[np.ndarray] = None) -> None:
        self.boxes[pos] = det.box
        self.scores[pos] = float(det.score)
        if desc is not None:
            self.descriptors[pos] = desc

    @property
    def positions(self) -> List[int]:
        return sorted(self.boxes)

    @property
    def hits(self) -> int:
        return len(self.boxes)

    @property
    def first(self) -> int:
        return min(self.boxes) if self.boxes else -1

    @property
    def last(self) -> int:
        return max(self.boxes) if self.boxes else -1

    @property
    def span(self) -> int:
        return (self.last - self.first + 1) if self.boxes else 0

    def gaps(self) -> int:
        """Positions inside [first, last] where this face was not matched."""
        return max(0, self.span - self.hits)

    def continuity(self) -> float:
        """Fraction of the track's own span where the face was actually seen."""
        return float(self.hits / self.span) if self.span else 0.0

    def mean_descriptor(self) -> Optional[np.ndarray]:
        if not self.descriptors:
            return None
        v = np.mean(np.stack(list(self.descriptors.values()), axis=0), axis=0)
        return v / (float(np.linalg.norm(v)) + EPS)

    def mean_box(self) -> Box:
        if not self.boxes:
            return (0, 0, 0, 0)
        arr = np.asarray([self.boxes[p] for p in self.positions], dtype=np.float64)
        m = arr.mean(axis=0)
        return (int(round(m[0])), int(round(m[1])), int(round(m[2])), int(round(m[3])))

    def mean_area(self) -> float:
        if not self.boxes:
            return 0.0
        return float(np.mean([box_area(self.boxes[p]) for p in self.positions]))

    def area_share(self, frame_area: float) -> float:
        return float(self.mean_area() / max(frame_area, 1.0))

    def summary(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "kind": self.kind,
            "frames": self.hits,
            "first_position": self.first,
            "last_position": self.last,
            "span": self.span,
            "gaps": self.gaps(),
            "continuity": round(self.continuity(), 3),
            "box": list(self.mean_box()),
            "positions": self.positions,
        }


# ---------------------------------------------------------------------------
# tracker
# ---------------------------------------------------------------------------
class FaceTracker:
    """Greedy, deterministic multi-face tracker over sampled frames."""

    def __init__(self, max_gap: int = 3, min_hits: int = 2, max_cost: float = 0.85,
                 appearance_veto: float = APPEARANCE_VETO,
                 appearance_veto_dist: float = APPEARANCE_VETO_DIST):
        self.max_gap = int(max_gap)
        self.min_hits = int(min_hits)
        self.max_cost = float(max_cost)
        self.appearance_veto = float(appearance_veto)
        self.appearance_veto_dist = float(appearance_veto_dist)
        self._active: List[Track] = []
        self._done: List[Track] = []
        self._next_id = 1
        self.matched_pairs = 0
        self.created = 0

    # -- association ------------------------------------------------------
    def _cost(self, track: Track, det: Detection, desc: Optional[np.ndarray]) -> Optional[float]:
        last_pos = track.last
        last = track.boxes[last_pos]
        cx0, cy0 = box_centre(last)
        cx1, cy1 = box_centre(det.box)
        dist = math.hypot(cx1 - cx0, cy1 - cy0) / (0.5 * (box_side(last) + box_side(det.box)))
        if dist > MAX_CENTRE_DIST:
            return None
        ratio = abs(math.log((box_side(det.box) + EPS) / (box_side(last) + EPS)))
        if ratio > MAX_SIZE_RATIO:
            return None
        cost = dist + 0.55 * ratio - 0.90 * box_iou(last, det.box)
        mean_desc = track.mean_descriptor()
        if mean_desc is not None and desc is not None and track.hits >= 2:
            sim = float(np.dot(mean_desc, desc))
            if sim < self.appearance_veto and dist > self.appearance_veto_dist:
                return None
            cost -= 0.20 * max(sim, 0.0)
        return cost

    def update(self, pos: int, detections: Sequence[Detection],
               descriptors: Optional[Sequence[np.ndarray]] = None) -> List[int]:
        """Assign detections in frame ``pos`` to tracks; returns ids (or -1)."""
        descs = list(descriptors) if descriptors is not None else [None] * len(detections)

        # close tracks that have been missing for too long
        for track in list(self._active):
            if pos - track.last > self.max_gap:
                self._active.remove(track)
                self._done.append(track)

        pairs: List[Tuple[float, int, int]] = []
        for ti, track in enumerate(self._active):
            for di, det in enumerate(detections):
                if det.kind != track.kind:      # never merge a real face with a fallback window
                    continue
                c = self._cost(track, det, descs[di])
                if c is not None and c <= self.max_cost:
                    pairs.append((c, ti, di))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))

        assigned: List[int] = [-1] * len(detections)
        used_tracks = set()
        for cost, ti, di in pairs:
            if ti in used_tracks or assigned[di] != -1:
                continue
            track = self._active[ti]
            track.add(pos, detections[di], descs[di])
            assigned[di] = track.track_id
            used_tracks.add(ti)
            self.matched_pairs += 1

        for di, det in enumerate(detections):
            if assigned[di] != -1:
                continue
            track = Track(track_id=self._next_id, kind=det.kind)
            self._next_id += 1
            self.created += 1
            track.add(pos, det, descs[di])
            self._active.append(track)
            assigned[di] = track.track_id
        return assigned

    # -- results ----------------------------------------------------------
    def finish(self, include_short: bool = True) -> List[Track]:
        tracks = self._done + [t for t in self._active if t not in self._done]
        if not include_short:
            tracks = [t for t in tracks if t.hits >= self.min_hits]
        return sorted(tracks, key=lambda t: (t.first, t.track_id))

    def stats(self) -> Dict[str, object]:
        return {
            "association": "greedy centre-distance + size-ratio + appearance",
            "max_gap": self.max_gap,
            "matched_pairs": self.matched_pairs,
            "identities_created": self.created,
        }
