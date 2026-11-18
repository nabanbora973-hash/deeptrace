"""Explainability layer: artifact heat-maps, per-frame timeline, spectrum view.

A verdict without evidence is unusable in newsrooms, courts and compliance
desks, so every analysis renders three artefacts next to the score:

* ``overlay.png``  - where in the face the fused evidence concentrates,
* ``timeline.png`` - how the per-frame evidence behaves over the clip,
* ``spectrum.png`` - the frequency view that drove the spectral findings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
FONT_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

BG = (13, 20, 30)
INK = (236, 241, 247)
MUTED = (138, 156, 178)
GRID = (40, 53, 70)
ACCENT = (64, 224, 208)
WARN = (255, 138, 108)

EPS = 1e-9


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _turbo(mask: np.ndarray) -> np.ndarray:
    """Colourise a 0..1 float mask with the turbo colour-map (BGR out)."""
    m = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


def heatmap_overlay(crop_bgr: np.ndarray, artifact_map: np.ndarray, out_path: Path,
                    caption: str = "Fused artifact energy") -> Path:
    """Blend the artifact heat-map over the analysed face crop."""
    base = crop_bgr.copy()
    if base.shape[0] != artifact_map.shape[0]:
        artifact_map = cv2.resize(artifact_map, (base.shape[1], base.shape[0]))
    heat = _turbo(cv2.GaussianBlur(artifact_map.astype(np.float32), (0, 0), 3.0))
    overlay = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
    panel = np.hstack([base, overlay])

    pad, label_h = 16, 34
    canvas = np.full((panel.shape[0] + label_h + pad * 2, panel.shape[1], 3), BG[::-1], np.uint8)
    canvas[label_h:label_h + panel.shape[0], :, :] = panel
    canvas = cv2.copyMakeBorder(canvas, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=BG[::-1])
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.text((pad, 10), f"{caption}  |  left: analysed crop   right: evidence heat-map",
           font=_font(16), fill=INK)
    img.save(out_path, format="PNG")
    return out_path


def spectrum_image(crop_bgr: np.ndarray, out_path: Path) -> Path:
    """Render the centred log-magnitude spectrum used by the frequency stream."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    spec = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    logmag = np.log10(np.abs(spec) ** 2 + EPS)
    lo, hi = np.percentile(logmag, 5), np.percentile(logmag, 99.5)
    norm = np.clip((logmag - lo) / (hi - lo + EPS), 0, 1)
    img = Image.fromarray(_turbo(norm)[:, :, ::-1])
    d = ImageDraw.Draw(img)
    d.text((10, 8), "Centred log-power spectrum (periodic peaks = generator grid)",
           font=_font(14), fill=(10, 14, 20))
    img.save(out_path, format="PNG")
    return out_path


def timeline_image(series: Sequence[float], timestamps: Sequence[float], out_path: Path,
                   threshold: float | None = None, title: str = "Per-frame artifact score") -> Path:
    """Line chart of the per-frame evidence score across the clip."""
    W, H = 900, 260
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 56, W - 24, 44, H - 42

    d.text((left, 14), title, font=_font(16, True), fill=INK)

    vals = np.asarray(list(series) if series else [0.0], dtype=np.float32)
    ts = np.asarray(list(timestamps) if timestamps else [0.0], dtype=np.float32)
    if ts.size != vals.size:
        ts = np.linspace(0, 1, vals.size)
    vmax = max(float(vals.max()), 0.1) * 1.15
    vmin = 0.0

    def px(i: int) -> float:
        return left + (right - left) * (i / max(vals.size - 1, 1))

    def py(v: float) -> float:
        return bottom - (bottom - top) * ((v - vmin) / (vmax - vmin + EPS))

    for i in range(5):
        y = top + (bottom - top) * i / 4
        d.line([(left, y), (right, y)], fill=GRID)
        d.text((8, y - 8), f"{vmax * (1 - i / 4):.2f}", font=_font(12), fill=MUTED)

    if threshold is not None:
        y = py(threshold)
        for x in range(left, right, 12):
            d.line([(x, y), (x + 6, y)], fill=WARN, width=2)
        d.text((right - 96, y - 18), "flag level", font=_font(12), fill=WARN)

    pts = [(px(i), py(float(v))) for i, v in enumerate(vals)]
    if len(pts) > 1:
        d.line(pts, fill=ACCENT, width=3)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=ACCENT)

    d.line([(left, bottom), (right, bottom)], fill=MUTED)
    d.text((left, bottom + 10), f"0.0 s", font=_font(12), fill=MUTED)
    d.text((right - 60, bottom + 10), f"{float(ts[-1]):.1f} s", font=_font(12), fill=MUTED)
    d.text((left + 4, bottom + 26), f"{vals.size} sampled frames", font=_font(11), fill=MUTED)

    img.save(out_path, format="PNG")
    return out_path


def evidence_bar_image(streams: Dict[str, Dict[str, object]], out_path: Path) -> Path:
    """Small bar chart of the three stream scores (0 = authentic, 1 = synthetic)."""
    W, H = 900, 150
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    left, right, bar_h = 150, W - 140, 22
    y = 30
    titles = {"spatial": "Spatial", "frequency": "Frequency", "temporal": "Temporal"}
    for name, stream in streams.items():
        score = float(stream.get("score", 0.0))
        rel = float(stream.get("reliability", 1.0))
        d.text((16, y), titles.get(name, name), font=_font(14, True), fill=INK)
        d.rectangle([left, y, right, y + bar_h], outline=GRID, fill=(20, 29, 41))
        w = int((right - left) * max(0.0, min(score, 1.0)))
        colour = WARN if score >= 0.65 else (ACCENT if score <= 0.35 else (232, 200, 106))
        d.rectangle([left, y, left + w, y + bar_h], fill=colour)
        d.text((right + 12, y + 3), f"{score:.2f}", font=_font(14, True), fill=INK)
        d.text((right + 58, y + 5), f"w={rel:.2f}", font=_font(11), fill=MUTED)
        y += bar_h + 16
    d.text((16, H - 20), "stream score: 0 = authentic-looking, 1 = synthetic-looking   (w = reliability weight)",
           font=_font(11), fill=MUTED)
    img.save(out_path, format="PNG")
    return out_path


# ---------------------------------------------------------------------------
# multi-face artefacts
# ---------------------------------------------------------------------------
_VERDICT_RGB = {"FAKE": (217, 74, 74), "REAL": (38, 166, 124), "INCONCLUSIVE": (203, 156, 44)}


def face_panel_image(frame_bgr: np.ndarray, entries: Sequence[Dict[str, object]],
                     out_path: Path, title: str = "Faces tracked in this clip") -> Path:
    """Annotated full frame + one card per tracked face.

    ``entries`` items: ``track_id, box, verdict, p_fake, crop, frames, total,
    analysed``.  Boxes are in source-frame coordinates.
    """
    W = 1000
    fh, fw = frame_bgr.shape[:2]
    top_h = int(round(W * fh / max(fw, 1)))
    frame_img = Image.fromarray(cv2.cvtColor(
        cv2.resize(frame_bgr, (W, top_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

    head_h, card_h, pad = 46, 168, 14
    canvas_h = head_h + top_h + pad + card_h + pad
    img = Image.new("RGB", (W, canvas_h), BG)
    img.paste(frame_img, (0, head_h))
    d = ImageDraw.Draw(img)
    d.text((pad, 14), title, font=_font(17, True), fill=INK)
    if len(entries) > 1:
        sub = "each face is scored and reported separately"
        x_sub = pad + d.textlength(title, font=_font(17, True)) + 18
        if x_sub + d.textlength(sub, font=_font(12)) < W - pad:
            d.text((x_sub, 19), sub, font=_font(12), fill=MUTED)

    for entry in entries:
        box = entry.get("box") or (0, 0, 0, 0)
        sx = W / max(fw, 1)
        x0, y0 = box[0] * sx, box[1] * sx + head_h
        x1, y1 = (box[0] + box[2]) * sx, (box[1] + box[3]) * sx + head_h
        colour = _VERDICT_RGB.get(str(entry.get("verdict")), (150, 160, 172))
        width = 3 if entry.get("analysed") else 1
        d.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        label = f"#{entry.get('track_id')} {entry.get('verdict', 'not scored')}"
        if entry.get("p_fake") is not None:
            label += f"  p={float(entry['p_fake']):.2f}"
        tw = d.textlength(label, font=_font(13, True))
        ly = max(head_h + 2, y0 - 20)
        d.rectangle([x0, ly, x0 + tw + 12, ly + 19], fill=colour)
        d.text((x0 + 6, ly + 3), label, font=_font(13, True), fill=(10, 14, 20))

    card_w = max(190, (W - pad * (len(entries) + 1)) // max(len(entries), 1))
    cy = head_h + top_h + pad
    for i, entry in enumerate(entries):
        cx = pad + i * (card_w + pad)
        if cx + card_w > W:
            break
        d.rectangle([cx, cy, cx + card_w, cy + card_h], outline=GRID, fill=(17, 24, 35))
        crop = entry.get("crop")
        if isinstance(crop, np.ndarray):
            thumb = Image.fromarray(cv2.cvtColor(
                cv2.resize(crop, (108, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            img.paste(thumb, (cx + 10, cy + 12))
        colour = _VERDICT_RGB.get(str(entry.get("verdict")), (150, 160, 172))
        tx = cx + 128
        d.text((tx, cy + 14), f"face #{entry.get('track_id')}", font=_font(14, True), fill=INK)
        d.text((tx, cy + 36), str(entry.get("verdict", "not scored")), font=_font(15, True), fill=colour)
        sub = []
        if entry.get("p_fake") is not None:
            sub.append(f"p(fake) {float(entry['p_fake']):.3f}")
        sub.append(f"{entry.get('frames')}/{entry.get('total')} frames")
        for j, line in enumerate(sub):
            d.text((tx, cy + 62 + j * 18), line, font=_font(12), fill=MUTED)
        if entry.get("role"):
            d.text((tx, cy + 104), str(entry["role"]), font=_font(11), fill=MUTED)
        if entry.get("note"):
            d.text((tx, cy + 122), str(entry["note"])[:34], font=_font(11), fill=MUTED)

    img.save(out_path, format="PNG")
    return out_path


def track_map_image(tracks: Sequence[Dict[str, object]], n_positions: int,
                    timestamps: Sequence[float], out_path: Path,
                    title: str = "Face identity across the sampled timeline") -> Path:
    """Lane chart: one row per tracked face, marks where it was present."""
    W, row_h, top, bottom, left = 900, 40, 52, 34, 248
    rows = max(len(tracks), 1)
    H = top + rows * row_h + bottom
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((14, 14), title, font=_font(16, True), fill=INK)

    right = W - 20
    span = max(n_positions - 1, 1)

    def px(pos: int) -> float:
        return left + (right - left) * (pos / span)

    for i in range(5):
        x = left + (right - left) * i / 4
        d.rectangle([x, top - 6, x + 1, top + rows * row_h + 4], fill=GRID)
    for i, track in enumerate(tracks):
        y = top + i * row_h + 8
        bar_h = 18
        colour = _VERDICT_RGB.get(str(track.get("verdict")), (150, 160, 172))
        present = [int(p) for p in (track.get("positions") or [])]
        d.rectangle([left, y, right, y + bar_h], fill=(20, 29, 41))
        if present:
            for pos in present:
                x = px(pos)
                d.rectangle([x - 2, y, x + 2, y + bar_h], fill=colour)
            d.rectangle([px(present[0]) - 2, y, px(present[-1]) + 2, y + bar_h], outline=colour)
        label = f"#{track.get('track_id')} · {len(present)} steps"
        d.text((14, y + 2), label, font=_font(13, True), fill=INK)
        detail = f"{track.get('verdict', 'not scored')}"
        if track.get("p_fake") is not None:
            detail += f" {float(track['p_fake']):.2f}"
        # right-align the verdict so it never runs into the lane marks
        dx = left - 12 - d.textlength(detail, font=_font(12, True))
        d.text((max(14, dx), y + 3), detail, font=_font(12, True), fill=colour)

    d.text((left, top + rows * row_h + 10),
           f"0.0 s", font=_font(11), fill=MUTED)
    if timestamps:
        d.text((right - 54, top + rows * row_h + 10), f"{float(timestamps[-1]):.1f} s",
               font=_font(11), fill=MUTED)
    img.save(out_path, format="PNG")
    return out_path
