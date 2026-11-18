"""Face detection, alignment and crop extraction (multi-face aware).

Uses OpenCV's bundled Haar cascades (no model download required), with an
optional YuNet DNN path when ``data/models/face_detection_yunet.onnx`` exists.
Falls back to a centred crop so the pipeline never crashes on hard input.

Unlike a "dominant face" detector, :meth:`FaceLocator.detect_all` returns *every*
face above a size/quality floor, sorted largest-first.  Each detection carries a
``kind`` - ``"face"`` for a real detection and ``"fallback"`` for the centred
window used when nothing is found - so the tracker can never merge a fallback
window into a real identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .tracking import Detection, box_area

YUNET_PATH = PROJECT_ROOT / "data" / "models" / "face_detection_yunet.onnx"

Box = Tuple[int, int, int, int]


@dataclass
class FaceCrop:
    image: np.ndarray        # aligned BGR crop, crop_size x crop_size
    box: Tuple[int, int, int, int]  # x, y, w, h in the source frame (crop region)
    detected: bool
    method: str
    score: float = 1.0
    kind: str = "face"              # "face" | "fallback"
    face_box: Tuple[int, int, int, int] | None = None  # detector box in canonical coords
    position: int = 0               # sampled-frame position this crop came from
    scale: float = 1.0              # canonicalisation factor applied to this frame


def _suppress_nested(boxes: List[Box]) -> List[int]:
    """Drop small boxes that sit largely inside a bigger face box.

    Haar occasionally fires again on a patch of an already-detected face (a cheek
    or a nostril region).  Intersection-over-own-area is the right test here: a
    genuine second person never occupies 20% of another person's face box, while a
    duplicate detection almost always does.
    """
    order = sorted(range(len(boxes)), key=lambda i: -box_area(boxes[i]))
    keep: List[int] = []
    for i in order:
        xi, yi, wi, hi = boxes[i]
        own = max(wi * hi, 1)
        nested = False
        for j in keep:
            xj, yj, wj, hj = boxes[j]
            ix0, iy0 = max(xi, xj), max(yi, yj)
            ix1, iy1 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter / own >= 0.20:
                nested = True
                break
        if not nested:
            keep.append(i)
    return keep


def _nms(boxes: List[Box], scores: List[float], iou_thr: float = 0.35) -> List[int]:
    """Plain greedy NMS over (x, y, w, h) boxes - keeps overlapping duplicates out."""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep: List[int] = []
    for i in order:
        xi, yi, wi, hi = boxes[i]
        ok = True
        for j in keep:
            xj, yj, wj, hj = boxes[j]
            ix0, iy0 = max(xi, xj), max(yi, yj)
            ix1, iy1 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = wi * hi + wj * hj - inter
            if union <= 0:
                continue
            # plain IoU, plus containment: a small "face" sitting inside a larger
            # one is a duplicate detection, not a second person
            containment = inter / max(min(wi * hi, wj * hj), 1)
            if inter / union > iou_thr or containment > 0.70:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


class FaceLocator:
    """Locate and crop faces in a frame (one or many)."""

    def __init__(self, crop_size: int = 256, margin: float = 0.22,
                 target_face_px: int = 192, max_faces: int = 3):
        self.crop_size = crop_size
        self.margin = margin
        # Canonical face size, in source pixels, before the final resize to
        # crop_size. Without this, a 4K source and a 480p source produce crops
        # with different effective resolution, and every noise/sharpness feature
        # shifts with the input resolution instead of with the manipulation.
        self.target_face_px = target_face_px
        self.max_faces = max(1, int(max_faces))
        cascade_dir = Path(cv2.data.haarcascades)
        self._frontal = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
        self._profile = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml"))
        self._eyes = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))
        self._yunet = None
        if YUNET_PATH.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000
                )
            except cv2.error:
                self._yunet = None

    # -- detection -------------------------------------------------------
    def detect_all(self, frame: np.ndarray) -> Tuple[List[Detection], str]:
        """All face boxes above the size floor, largest first."""
        H, W = frame.shape[:2]
        min_side = max(40, int(0.06 * min(H, W)))

        if self._yunet is not None:
            try:
                self._yunet.setInputSize((W, H))
                _, faces = self._yunet.detect(frame)
                if faces is not None and len(faces):
                    cand = [(int(f[0]), int(f[1]), int(f[2]), int(f[3]),
                             float(f[14]) if len(f) > 14 else 1.0) for f in faces]
                    cand = [c for c in cand if min(c[2], c[3]) >= min_side]
                    if cand:
                        idx = _nms([c[:4] for c in cand], [c[4] for c in cand])
                        kept = [cand[i][:4] for i in idx]
                        sanitised = [kept[i] for i in _suppress_nested(kept)]
                        dets = [Detection(box=b, score=float(max(
                            c[4] for c in cand if c[:4] == b)), kind="face")
                                for b in sanitised][: self.max_faces]
                        if dets:
                            return dets, "yunet"
            except cv2.error:
                pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        scale = 1.0
        if max(H, W) > 900:  # speed: detect on a downscale, then map back
            scale = 900.0 / max(H, W)
            small = cv2.resize(gray, (int(W * scale), int(H * scale)))
        else:
            small = gray

        for cascade, name in ((self._frontal, "haar-frontal"), (self._profile, "haar-profile")):
            rects = cascade.detectMultiScale(small, scaleFactor=1.12, minNeighbors=5,
                                             minSize=(40, 40))
            if len(rects):
                inv = 1.0 / scale
                boxes = [(int(r[0] * inv), int(r[1] * inv), int(r[2] * inv), int(r[3] * inv))
                         for r in rects]
                boxes = [b for b in boxes if min(b[2], b[3]) >= min_side]
                if not boxes:
                    continue
                idx = _nms(boxes, [float(box_area(b)) for b in boxes])
                boxes = [boxes[i] for i in idx]
                boxes = [boxes[i] for i in _suppress_nested(boxes)][: self.max_faces]
                return [Detection(box=b, score=1.0, kind="face") for b in boxes], name
        return [], "none"

    def detect(self, frame: np.ndarray) -> Tuple[Box | None, str]:
        """The single largest face (backwards-compatible helper)."""
        dets, method = self.detect_all(frame)
        if dets:
            return dets[0].box, method
        return None, "none"

    # -- cropping --------------------------------------------------------
    def _canonicalise(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
        """Resize the frame so the face box has a fixed side length."""
        x, y, w, h = box
        side = max(w, h)
        if side <= 0:
            return frame, box, 1.0
        s = self.target_face_px / float(side)
        if 0.95 <= s <= 1.05:
            return frame, box, 1.0
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        H, W = frame.shape[:2]
        new_w, new_h = max(8, int(round(W * s))), max(8, int(round(H * s)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        nb = (int(round(x * s)), int(round(y * s)), max(8, int(round(w * s))), max(8, int(round(h * s))))
        return resized, nb, s

    def _crop_from_box(self, frame: np.ndarray, box: Box) -> Tuple[np.ndarray, Box]:
        """Margin-padded, squared-up crop around one face box."""
        H, W = frame.shape[:2]
        x, y, w, h = box
        mx, my = int(w * self.margin), int(h * self.margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        # square it up so aspect ratio never distorts the spectra
        cw, ch = x1 - x0, y1 - y0
        side = max(cw, ch)
        cx0 = x0 - (side - cw) // 2
        cy0 = y0 - (side - ch) // 2
        x0, y0 = max(0, cx0), max(0, cy0)
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        box = (x0, y0, x1 - x0, y1 - y0)
        if box[2] <= 0 or box[3] <= 0:
            box = (0, 0, W, H)
        x, y, w, h = box
        patch = frame[y:y + h, x:x + w]
        if patch.size == 0:
            patch = frame
        interp = cv2.INTER_AREA if patch.shape[0] > self.crop_size else cv2.INTER_CUBIC
        crop = cv2.resize(patch, (self.crop_size, self.crop_size), interpolation=interp)
        return crop, box

    def crop_faces(self, frame: np.ndarray, position: int = 0,
                   with_fallback: bool = True) -> Tuple[List[FaceCrop], str]:
        """Crop *every* detected face in one frame, plus the detection method.

        One canonicalisation is applied per frame (driven by the largest face) so
        all crops of that frame share the same scale - otherwise a small face and
        a large face would be resampled differently and their spectra would not be
        comparable.
        """
        dets, method = self.detect_all(frame)
        if not dets:
            if not with_fallback:
                return [], "none"
            H, W = frame.shape[:2]
            side = int(min(W, H) * 0.62)
            box = ((W - side) // 2, int((H - side) * 0.30), side, side)
            crop, box = self._crop_from_box(frame, box)
            return [FaceCrop(image=crop, box=box, detected=False, method="none",
                             score=0.0, kind="fallback", face_box=box,
                             position=position, scale=1.0)], "none"

        largest = max(dets, key=lambda d: box_area(d.box)).box
        frame_c, canonical_box, s = self._canonicalise(frame, largest)
        out: List[FaceCrop] = []
        for det in dets:
            if s != 1.0:
                box = (int(round(det.box[0] * s)), int(round(det.box[1] * s)),
                       max(8, int(round(det.box[2] * s))), max(8, int(round(det.box[3] * s))))
            else:
                box = det.box
            crop, padded = self._crop_from_box(frame_c, box)
            out.append(FaceCrop(image=crop, box=box, detected=True, method=method,
                                score=float(det.score), kind="face", face_box=box,
                                position=position, scale=s))
        return out, method

    def crop(self, frame: np.ndarray) -> FaceCrop:
        """Crop the largest face (falls back to a centred window)."""
        crops, _ = self.crop_faces(frame)
        return crops[0]

    def crop_all(self, frames: List[np.ndarray]) -> List[FaceCrop]:
        return [self.crop(f) for f in frames]
