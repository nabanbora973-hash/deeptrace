"""Analysis engine: ingest -> multi-face tracking -> per-face 3-stream scoring -> fusion.

The engine is intentionally framework-free so it can be used from the REST API,
the CLI, notebooks or a batch evaluation harness:

    from deeptrace.engine import analyze_media
    result = analyze_media("clip.mp4")

Two-level design
----------------
1. **Per face.** Every detected face is tracked across the sampled timeline and
   gets its *own* three streams, its own calibrated fake-probability and its own
   evidence ledger.  A swapped face on the right of the frame can no longer be
   diluted by a genuine face on the left.
2. **Per clip.** The document verdict is the worst-case face: the clip is FAKE if
   any analysed face is FAKE, because one manipulated identity is enough to make
   the recording suspect.  Face disagreement is reported explicitly instead of
   being averaged away.

``MIN_TRACK_FRAMES`` guards against scoring faces seen in one or two frames; such
faces are listed with ``analysed: false`` and a reason, never silently dropped.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .explain import (evidence_bar_image, face_panel_image, heatmap_overlay,
                      spectrum_image, timeline_image, track_map_image)
from .face import FaceCrop, FaceLocator
from .abstain import describe as describe_abstention
from .fusion import apply_reliability, fuse, score_stream, summarise
from .ingest import FrameSet, sample_frames
from .report import build_pdf
from .signals.frequency import analyze_frequency
from .signals.spatial import analyze_spatial
from .signals.temporal import analyze_temporal
from .tracking import Detection, FaceTracker, Track, descriptor


def _median_feature(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = set().union(*[d.keys() for d in dicts]) if dicts else set()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        out[k] = float(np.median(vals)) if vals else 0.0
    return out


def _mean_map(maps: List[np.ndarray]) -> np.ndarray:
    if not maps:
        return np.zeros((config.CROP_SIZE, config.CROP_SIZE), np.float32)
    stack = np.stack([m.astype(np.float32) for m in maps], axis=0)
    return stack.mean(axis=0)


def _source_box(crop: FaceCrop) -> Tuple[int, int, int, int]:
    """Face box in source-frame pixels (undoes the per-frame canonicalisation)."""
    scale = crop.scale or 1.0
    box = crop.face_box or crop.box
    return tuple(int(round(v / scale)) for v in box)  # type: ignore[return-value]


def _sampling_fps(timestamps: Sequence[float], fallback: float) -> float:
    """Effective sampling rate of a track's own timeline."""
    if len(timestamps) >= 2:
        span = float(timestamps[-1] - timestamps[0])
        if span > 1e-6:
            return float((len(timestamps) - 1) / span)
    return float(fallback)


# ---------------------------------------------------------------------------
# 1. detection + tracking pass
# ---------------------------------------------------------------------------
def _scan_tracks(frameset: FrameSet, locator: FaceLocator,
                 settings: config.Settings) -> Dict[str, Any]:
    """Detect all faces in every sampled frame and associate them into tracks."""
    tracker = FaceTracker(max_gap=settings.track_max_gap,
                          min_hits=min(settings.min_track_frames, 2))
    crops_by_track: Dict[int, Dict[int, FaceCrop]] = {}
    per_position: List[List[Tuple[FaceCrop, int]]] = []
    frames_canon: List[np.ndarray] = []
    methods: List[str] = []
    detected_frames = 0
    max_simultaneous = 0
    multi_face_frames = 0

    for pos, frame in enumerate(frameset.frames):
        crops, method = locator.crop_faces(frame, position=pos)
        methods.append(method)
        real = [c for c in crops if c.kind == "face"]
        if real:
            detected_frames += 1
        if len(real) > 1:
            multi_face_frames += 1
        max_simultaneous = max(max_simultaneous, len(real))

        # Association happens in *source-frame* coordinates. Tracking canonical
        # boxes would compare boxes from different rescale factors frame to frame,
        # which shreds one face into several short tracks.
        dets = [Detection(box=_source_box(c), score=c.score, kind=c.kind) for c in crops]
        descs = [descriptor(c.image) for c in crops]
        ids = tracker.update(pos, dets, descs)
        for crop, tid in zip(crops, ids):
            if tid < 0:
                continue
            crops_by_track.setdefault(tid, {})[pos] = crop
        per_position.append(list(zip(crops, ids)))
        frames_canon.append(frame)

    n = max(len(frameset.frames), 1)
    detected_methods = [m for m in methods if m not in ("none",)]
    method = max(set(detected_methods), key=detected_methods.count) if detected_methods else "none"
    return {
        "tracker": tracker,
        "tracks": tracker.finish(),
        "crops_by_track": crops_by_track,
        "per_position": per_position,
        "frames": frames_canon,
        "method": method,
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / n,
        "max_simultaneous": max_simultaneous,
        "multi_face_frames": multi_face_frames,
    }


# ---------------------------------------------------------------------------
# 2. per-face analysis
# ---------------------------------------------------------------------------
def _analyse_track(track: Track, crops: Dict[int, FaceCrop], frameset: FrameSet,
                   settings: config.Settings) -> Dict[str, Any]:
    """Run the three evidence streams for one face, then fuse them."""
    positions = [p for p in track.positions if p in crops]
    if not positions:
        positions = sorted(crops)
    face_crops = [crops[p] for p in positions]
    images = [c.image for c in face_crops]
    timestamps = [frameset.timestamps[p] for p in positions]
    calibration = settings.calibration

    # -- spatial
    spatial_frames = [analyze_spatial(img) for img in images]
    spatial_feats = _median_feature([s["features"] for s in spatial_frames])
    spatial_map = _mean_map([s["map"] for s in spatial_frames])
    seam_series = [float(s["features"]["seam_contrast"]) for s in spatial_frames]

    # -- frequency (every other frame is enough)
    step = 1 if len(images) <= 16 else 2
    freq_frames = [analyze_frequency(img) for img in images[::step]]
    freq_feats = _median_feature([f["features"] for f in freq_frames])
    freq_map = _mean_map([f["map"] for f in freq_frames])

    # -- temporal (this face's own timeline, its own effective sampling rate)
    eff_fps = _sampling_fps(timestamps, frameset.fps)
    temporal = analyze_temporal(images, seam_series, timestamps, eff_fps)

    # -- scoring + reliability
    streams = {
        "spatial": score_stream("spatial", spatial_feats, calibration),
        "frequency": score_stream("frequency", freq_feats, calibration),
        "temporal": score_stream("temporal", temporal["features"], calibration),
    }
    detected = sum(1 for c in face_crops if c.detected)
    det_rate = detected / max(len(face_crops), 1)
    det_reliability = 0.45 + 0.55 * det_rate
    note = None
    if det_rate < 0.6:
        note = (f"Face detected in only {detected}/{len(face_crops)} of this face's frames - "
                "part of the crop set falls back to a centred window.")
    apply_reliability(streams["spatial"], det_reliability, note)
    apply_reliability(streams["frequency"], det_reliability, note)

    t_support = float(temporal["support"])
    t_note = None
    if len(face_crops) < config.MIN_FRAMES_TEMPORAL:
        t_note = (f"Only {len(face_crops)} frames for this face (image or short "
                  "appearance): pulse evidence unavailable, temporal weight reduced.")
    elif t_support < 0.7:
        t_note = f"Temporal support {t_support:.2f} (short appearance / low frame rate)."
    apply_reliability(streams["temporal"], t_support, t_note)

    verdict = fuse(streams, calibration,
                   {"fake": settings.fake_threshold, "real": settings.real_threshold})
    verdict["summary"] = summarise(streams, verdict)
    # why the review band sits where it does - comes from the calibration policy, not code
    verdict["abstention_note"] = describe_abstention(calibration.get("abstention"))

    return {
        "track": track,
        "positions": positions,
        "crops": face_crops,
        "images": images,
        "timestamps": timestamps,
        "seam_series": seam_series,
        "maps": {"spatial": spatial_map, "frequency": freq_map, "temporal": temporal["map"]},
        "streams": streams,
        "temporal": temporal["stats"],
        "verdict": verdict,
        "detection_rate": det_rate,
        "sampling_fps": eff_fps,
        "p_fake": float(verdict["p_fake"]),
    }


def _aggregate(analyses: List[Dict[str, Any]], tracks: List[Track]) -> Dict[str, Any]:
    """Worst-case document verdict over the analysed faces."""
    scored = [a for a in analyses if a is not None]
    if not scored:
        return {"strategy": "none", "verdict": "INCONCLUSIVE", "p_fake": 0.5,
                "track_id": None, "n_faces": len(tracks), "n_analysed": 0}
    worst = max(scored, key=lambda a: (float(a["verdict"]["p_fake"]), a["track"].hits))
    lo = min(float(a["verdict"]["p_fake"]) for a in scored)
    hi = float(worst["verdict"]["p_fake"])
    verdicts = [a["verdict"]["verdict"] for a in scored]
    if len(scored) == 1:
        agreement = "single face"
    elif len(set(verdicts)) == 1:
        agreement = f"all {len(scored)} faces agree ({verdicts[0]})"
    else:
        flagged = sum(1 for v in verdicts if v == "FAKE")
        agreement = f"{flagged} of {len(scored)} faces flagged"
    return {
        "strategy": "worst-case face",
        "verdict": worst["verdict"]["verdict"],
        "p_fake": round(hi, 4),
        "track_id": worst["track"].track_id,
        "n_faces": len(tracks),
        "n_analysed": len(scored),
        "p_fake_min": round(lo, 4),
        "p_fake_max": round(hi, 4),
        "disagreement": round(hi - lo, 4),
        "agreement": agreement,
        "abstention_note": worst["verdict"].get("abstention_note"),
        "per_face": [
            {"track_id": a["track"].track_id,
             "verdict": a["verdict"]["verdict"],
             "p_fake": a["verdict"]["p_fake"]}
            for a in sorted(scored, key=lambda a: a["track"].track_id)
        ],
    }


def analyze_media(
    path: str | Path,
    settings: Optional[config.Settings] = None,
    run_id: Optional[str] = None,
    make_pdf: bool = True,
    render_artifacts: bool = True,
) -> Dict[str, Any]:
    """Run the full multi-face, multi-signal analysis on one video or image."""
    config.ensure_dirs()
    settings = settings or config.Settings()
    calibration = settings.calibration
    run_id = run_id or uuid.uuid4().hex[:12]
    outdir = config.WORK_DIR / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    # ---- 1. ingest ------------------------------------------------------
    t0 = time.perf_counter()
    frameset = sample_frames(path, max_frames=settings.max_frames)
    timings["ingest_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ---- 2. detect + track every face -----------------------------------
    t0 = time.perf_counter()
    locator = FaceLocator(crop_size=settings.crop_size, max_faces=settings.max_faces)
    scan = _scan_tracks(frameset, locator, settings)
    timings["face_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    tracks: List[Track] = scan["tracks"]
    crops_by_track: Dict[int, Dict[int, FaceCrop]] = scan["crops_by_track"]
    faces_notes: List[str] = []

    # pick the faces worth scoring: enough frames to say anything about them
    eligible = [t for t in tracks if t.hits >= settings.min_track_frames]
    if not eligible and tracks:
        eligible = [t for t in tracks if t.hits >= 1]
        faces_notes.append(
            f"No face was visible in >= {settings.min_track_frames} sampled frames; "
            "every tracked face was scored with reduced temporal support."
        )
    eligible.sort(key=lambda t: t.track_id)

    analyses = [_analyse_track(t, crops_by_track.get(t.track_id, {}), frameset, settings)
                for t in eligible]
    by_id = {a["track"].track_id: a for a in analyses}

    # primary face = the one that dominates the frame (used for "which face is this")
    primary_id = None
    if analyses:
        primary_id = max(analyses,
                         key=lambda a: (a["track"].hits, a["track"].mean_area()))["track"].track_id
    aggregate = _aggregate(analyses, tracks)
    verdict_track_id = aggregate.get("track_id") or primary_id
    chosen = by_id.get(verdict_track_id) or (analyses[0] if analyses else None)

    # A "fallback" track is the centred window used when the cascade finds nothing.
    # Scoring it and printing REAL is how a missed face turns into a false clearance,
    # so it is treated exactly like "no face at all": abstain, with the reason attached.
    fallback_only = chosen is not None and getattr(chosen["track"], "kind", "face") != "face"
    no_face = chosen is None or fallback_only or scan["detection_rate"] == 0.0
    if no_face:
        # Nothing tracked. This used to raise (HTTP 500) and, before that, to look like a
        # clean "REAL". Both are wrong: absence of a face is absence of evidence, so we
        # abstain explicitly and tell the caller why.
        verdict = _no_face_verdict(scan, len(tracks), fallback_only=fallback_only)
        streams: Dict[str, Any] = {}
        temporal_section: Dict[str, Any] = {}
        faces_notes.append(
            "No real face was tracked (detection rate "
            f"{scan['detection_rate']:.0%}"
            + ("; only the fallback window was available" if fallback_only else "")
            + "). Front-facing Haar cascades miss profile views, heavy occlusion, low light and "
            "small faces, so this clip is not evidence of authenticity."
        )
    else:
        verdict = dict(chosen["verdict"])
        streams = chosen["streams"]
        temporal_section = chosen["temporal"]
        verdict["scope"] = "document"
        verdict["track_id"] = chosen["track"].track_id
        verdict["face_agreement"] = {
            "n_faces": len(tracks), "n_analysed": len(by_id),
            "detail": aggregate.get("agreement"),
        }
        if len(by_id) > 1:
            verdict["summary"] = (f"{aggregate['agreement']}; the clip verdict follows the "
                                  f"worst-scoring face (#{chosen['track'].track_id}). "
                                  + verdict["summary"])

    # ---- 3. explainability artefacts ------------------------------------
    t0 = time.perf_counter()
    artifacts: Dict[str, str] = {}
    heat = tl = spec = bars = faces_img = tracks_img = None
    if render_artifacts and not no_face:
        seam_series = chosen["seam_series"]
        rep_idx = (int(np.argmin(np.abs(np.array(seam_series) - np.median(seam_series))))
                   if seam_series else 0)
        rep_crop = chosen["images"][min(rep_idx, len(chosen["images"]) - 1)]
        maps = chosen["maps"]
        fused_map = np.clip(
            0.55 * maps["spatial"] + 0.30 * cv2_resize(maps["frequency"])
            + 0.15 * cv2_resize(maps["temporal"]), 0, 1
        )
        heat = heatmap_overlay(rep_crop, fused_map, outdir / "overlay.png",
                               caption=f"Fused evidence (face #{chosen['track'].track_id}, "
                                       f"{verdict['verdict']})")
        tl = timeline_image(seam_series, chosen["timestamps"], outdir / "timeline.png",
                            threshold=0.25,
                            title=f"Per-frame blending-seam energy - face #{chosen['track'].track_id}")
        spec = spectrum_image(rep_crop, outdir / "spectrum.png")
        bars = evidence_bar_image(streams, outdir / "streams.png")

        # per-face crops (small, used by the UI as face-selector thumbnails)
        for a in analyses:
            tid = a["track"].track_id
            thumb = a["images"][len(a["images"]) // 2]
            cv2.imwrite(str(outdir / f"face_{tid}.png"), thumb)
            artifacts[f"face_{tid}"] = f"/runs/{run_id}/face_{tid}.png"

        entries = []
        for a in analyses:
            tid = a["track"].track_id
            role = []
            if tid == primary_id:
                role.append("primary face")
            if tid == verdict_track_id and len(by_id) > 1:
                role.append("verdict source")
            entries.append({
                "track_id": tid,
                "box": a["crops"][0].face_box or a["track"].mean_box(),
                "verdict": a["verdict"]["verdict"],
                "p_fake": a["verdict"]["p_fake"],
                "crop": a["images"][len(a["images"]) // 2],
                "frames": a["track"].hits,
                "total": len(frameset.frames),
                "analysed": True,
                "role": " · ".join(role),
            })
        for t in tracks:
            if t.track_id in by_id:
                continue
            entries.append({
                "track_id": t.track_id,
                "box": t.mean_box(),
                "verdict": "not scored",
                "p_fake": None,
                "crop": (crops_by_track.get(t.track_id, {}).get(t.last).image
                         if t.track_id in crops_by_track else None),
                "frames": t.hits,
                "total": len(frameset.frames),
                "analysed": False,
                "role": "too few frames",
                "note": f"< {settings.min_track_frames} frames",
            })
        entries.sort(key=lambda e: e["track_id"])

        # representative frame = the sampled frame showing the most faces
        per_position = scan["per_position"]
        rep_pos = max(range(len(per_position)),
                      key=lambda i: (len([1 for c, _ in per_position[i] if c.kind == "face"]), -i),
                      default=0)
        rep_frame = frameset.frames[rep_pos]
        for entry in entries:
            box = _source_box_at(entry["track_id"], crops_by_track, rep_pos)
            if box is not None:
                entry["box"] = box
        faces_img = face_panel_image(rep_frame, entries, outdir / "faces.png",
                                     title=f"Faces tracked in this clip ({len(tracks)} found, "
                                           f"{len(by_id)} scored)")
        tracks_img = track_map_image(
            [{"track_id": a["track"].track_id, "positions": a["positions"],
              "verdict": a["verdict"]["verdict"], "p_fake": a["verdict"]["p_fake"]}
             for a in analyses]
            + [{"track_id": t.track_id, "positions": t.positions,
                "verdict": "not scored", "p_fake": None}
               for t in tracks if t.track_id not in by_id],
            len(frameset.frames), frameset.timestamps, outdir / "tracks.png")

        artifacts.update({
            "overlay": f"/runs/{run_id}/overlay.png",
            "timeline": f"/runs/{run_id}/timeline.png",
            "spectrum": f"/runs/{run_id}/spectrum.png",
            "streams": f"/runs/{run_id}/streams.png",
            "faces": f"/runs/{run_id}/faces.png",
            "tracks": f"/runs/{run_id}/tracks.png",
        })
    timings["explain_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ---- 4. assemble result ---------------------------------------------
    result: Dict[str, Any] = {
        "run_id": run_id,
        "engine": {"name": config.APP_NAME, "version": config.APP_VERSION},
        "input": {
            "filename": Path(path).name,
            "kind": frameset.kind,
            "duration_s": round(frameset.duration, 2),
            "fps": round(frameset.fps, 2),
            "resolution": f"{frameset.width}x{frameset.height}",
            "frames_sampled": frameset.count,
            "frames_total": frameset.total_frames,
        },
        "faces": {
            "detected_frames": scan["detected_frames"],
            "detection_rate": round(scan["detection_rate"], 3),
            "method": scan["method"],
            "crop_size": settings.crop_size,
            "count": len(tracks),
            "max_simultaneous": scan["max_simultaneous"],
            "multi_face_frames": scan["multi_face_frames"],
            "analysed": len(by_id),
            "primary_track": primary_id,
            "verdict_track": verdict_track_id,
            "tracking": {
                **scan["tracker"].stats(),
                "frames_sampled": len(frameset.frames),
                "min_track_frames": settings.min_track_frames,
                "max_faces": settings.max_faces,
            },
            "aggregate": aggregate,
            "notes": faces_notes,
            "tracks": [_track_payload(a, primary_id, verdict_track_id,
                                      len(frameset.frames)) for a in analyses]
            + [_unscored_payload(t, crops_by_track, len(frameset.frames), settings)
               for t in tracks if t.track_id not in by_id],
        },
        "verdict": verdict,
        "streams": streams,
        "temporal": temporal_section,
        "timings_ms": timings,
        "artifacts": artifacts,
        "disclaimer": (
            "DeepTrace reports forensic evidence, not proof. Treat the verdict as one "
            "input alongside provenance metadata, source history and human review."
        ),
    }
    result["faces"]["tracks"].sort(key=lambda t: t["track_id"])
    timings["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)

    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["artifacts"]["json"] = f"/runs/{run_id}/result.json"
    if no_face:
        result.setdefault("warnings", []).append(
            "No face was tracked, so no evidence images or PDF were rendered - "
            "an empty heatmap would look like a clean result."
        )

    if make_pdf and render_artifacts and not no_face:
        try:
            pdf_path = outdir / "report.pdf"
            build_pdf(result, pdf_path,
                      images={"overlay": heat, "timeline": tl, "spectrum": spec,
                              "streams": bars, "faces": faces_img, "tracks": tracks_img})
            result["artifacts"]["pdf"] = f"/runs/{run_id}/report.pdf"
        except RuntimeError as exc:  # reportlab missing - keep JSON + images
            result["warnings"] = [str(exc)]
        (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result



def _no_face_verdict(scan: Dict[str, Any], n_tracks: int,
                     fallback_only: bool = False) -> Dict[str, Any]:
    """Explicit abstention for inputs where no face could be scored.

    A clip with no detectable face says nothing about manipulation. Returning REAL here is
    the most dangerous possible false negative, so the honest answer is INCONCLUSIVE with
    the detection diagnostics attached.
    """
    cal_raw = config.load_calibration().raw
    return {
        "verdict": "INCONCLUSIVE",
        "p_fake": 0.5,
        "p_real": 0.5,
        "confidence": 0.0,
        "logit": 0.0,
        "thresholds": cal_raw.get("thresholds", {}),
        "stream_contributions": {},
        "effective_weights": {},
        "leading_stream": None,
        "stream_spread": 0.0,
        "mixed_evidence": False,
        "calibration_version": cal_raw.get("version"),
        "summary": (
            "No real face was tracked, so no stream produced usable evidence. This is an "
            "abstention, not a verdict: the clip is neither cleared nor accused. A missed "
            "face is the most dangerous failure mode, so DeepTrace refuses to call it REAL."
        ),
        "abstention_note": (
            f"face detection rate {scan['detection_rate']:.0%} across the sampled frames"
            + (" (fallback window only)" if fallback_only else "")
            + "; frontal Haar cascades fail on profile views, occlusion, low light and small faces"
        ),
        "scope": "document",
        "track_id": None,
        "face_agreement": {"n_faces": n_tracks, "n_analysed": 0,
                           "detail": "no scored face"},
        "no_face": True,
    }


def _source_box_at(track_id: int, crops_by_track: Dict[int, Dict[int, FaceCrop]],
                   rep_pos: int) -> Optional[Tuple[int, int, int, int]]:
    """Face box of a track, in *source-frame* pixels, nearest the representative frame.

    Crops live in canonicalised coordinates (the frame is resized so the largest
    face is a fixed size), so the stored scale is undone here before the box is
    drawn on the untouched frame.
    """
    crops = crops_by_track.get(track_id) or {}
    if not crops:
        return None
    pos = min(crops, key=lambda p: abs(p - rep_pos))
    crop = crops[pos]
    scale = crop.scale or 1.0
    box = crop.face_box or crop.box
    return tuple(int(round(v / scale)) for v in box)  # type: ignore[return-value]


def _track_payload(a: Dict[str, Any], primary_id: Optional[int],
                   verdict_id: Optional[int], total_frames: int) -> Dict[str, Any]:
    track: Track = a["track"]
    payload = track.summary()
    payload.update({
        "role": "primary" if track.track_id == primary_id else "secondary",
        "verdict_source": track.track_id == verdict_id,
        "analysed": True,
        "verdict": a["verdict"],
        "streams": a["streams"],
        "temporal": a["temporal"],
        "detection_rate": round(a["detection_rate"], 3),
        "sampling_fps": round(a["sampling_fps"], 2),
        "frames_total": total_frames,
    })
    return payload


def _unscored_payload(track: Track, crops_by_track: Dict[int, Dict[int, FaceCrop]],
                      total_frames: int, settings: config.Settings) -> Dict[str, Any]:
    payload = track.summary()
    payload.update({
        "role": "unscored",
        "analysed": False,
        "verdict": None,
        "streams": None,
        "reason": f"visible in {track.hits} sampled frames "
                  f"(minimum {settings.min_track_frames} to score a face)",
        "detection_rate": 1.0 if track.kind == "face" else 0.0,
        "frames_total": total_frames,
    })
    return payload


def cv2_resize(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (config.CROP_SIZE, config.CROP_SIZE))
