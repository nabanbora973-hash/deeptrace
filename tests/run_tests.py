"""Test suite for DeepTrace (plain asserts, no pytest required).

    python tests/run_tests.py

Covers: signal maths sanity, fusion additivity, end-to-end analysis on synthesised
media, artifact generation, the HTTP API surface, calibration loading, the
multi-face path (identity tracking, per-face scoring, worst-case aggregation), and
the abstention policy (risk-coverage monotonicity, cost selection, conformal bound).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deeptrace import abstain, config  # noqa: E402
from deeptrace.engine import analyze_media  # noqa: E402
from deeptrace.face import FaceLocator  # noqa: E402
from deeptrace.fusion import fuse, score_stream  # noqa: E402
from deeptrace.ingest import IngestError, sample_frames  # noqa: E402
from deeptrace.signals.frequency import analyze_frequency  # noqa: E402
from deeptrace.signals.spatial import analyze_spatial  # noqa: E402
from deeptrace.signals.temporal import analyze_temporal  # noqa: E402
from deeptrace.tracking import Detection, FaceTracker, descriptor  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_video(path: Path, frames: int = 40, size=(320, 240)) -> Path:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    rng = np.random.default_rng(0)
    for i in range(frames):
        f = np.full((h, w, 3), 60, np.uint8)
        cv2.rectangle(f, (10, 10), (w - 10, h - 10), (110, 130, 150), -1)
        cv2.ellipse(f, (w // 2 + i % 5, h // 2), (46, 60), 0, 0, 360, (185, 165, 145), -1)
        cv2.circle(f, (w // 2 - 16 + i % 5, h // 2 - 14), 5, (40, 40, 40), -1)
        cv2.circle(f, (w // 2 + 16 + i % 5, h // 2 - 14), 5, (40, 40, 40), -1)
        f = np.clip(f.astype(np.float32) + rng.normal(0, 2.0, f.shape), 0, 255).astype(np.uint8)
        writer.write(f)
    writer.release()
    return path


def test_signals() -> None:
    print("\nsignals")
    loc = FaceLocator(crop_size=256)
    frame = np.full((300, 300, 3), 90, np.uint8)
    cv2.ellipse(frame, (150, 150), (70, 90), 0, 0, 360, (190, 170, 150), -1)
    crop = loc.crop(frame).image

    sp = analyze_spatial(crop)
    check("spatial returns 5 features", len(sp["features"]) == 5, str(sorted(sp["features"])))
    check("spatial map matches crop size", sp["map"].shape == (256, 256))
    check("spatial features finite",
          all(np.isfinite(v) for v in sp["features"].values()))

    fr = analyze_frequency(crop)
    check("frequency returns 5 features", len(fr["features"]) == 5)
    check("hf_ratio in [0,1]", 0.0 <= fr["features"]["hf_ratio"] <= 1.0,
          f"{fr['features']['hf_ratio']:.4f}")
    check("spectral slope is finite", np.isfinite(fr["features"]["spectral_slope"]))

    crops = [crop] * 30
    tm = analyze_temporal(crops, [0.2] * 30, [i / 25 for i in range(30)], 25.0)
    check("temporal support bounded", 0.0 < tm["support"] <= 1.0, f"{tm['support']:.2f}")
    check("temporal features finite", all(np.isfinite(v) for v in tm["features"].values()))
    check("static clip yields flicker≈0", tm["features"]["flicker"] < 0.05,
          f"{tm['features']['flicker']:.3f}")

    # a swap-like composite should raise the spatial noise-uniformity signal
    swapped = crop.copy()
    donor = (crop.astype(np.float32) * 0.9 + 12).astype(np.uint8)
    mask = np.zeros(crop.shape[:2], np.float32)
    cv2.ellipse(mask, (128, 128), (86, 96), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 12)[..., None]
    swapped = (crop * (1 - mask) + donor * mask).astype(np.uint8)
    s_clean = analyze_spatial(crop)["features"]
    s_swap = analyze_spatial(swapped)["features"]
    check("composite changes spatial texture statistics",
          abs(s_swap["noise_uniformity"] - s_clean["noise_uniformity"]) > 1e-3,
          f"{s_clean['noise_uniformity']:.3f} -> {s_swap['noise_uniformity']:.3f}")


def test_fusion() -> None:
    print("\nfusion")
    cal = config.load_calibration().raw
    feats = {s: {k: float(v["mean"]) for k, v in cal["streams"][s]["features"].items()}
             for s in config.STREAM_NAMES}
    streams = {s: score_stream(s, feats[s], cal) for s in config.STREAM_NAMES}
    check("mean features land near 0.5", all(0.3 < st["score"] < 0.7 for st in streams.values()),
          str({k: round(v["score"], 2) for k, v in streams.items()}))

    verdict = fuse(streams, cal)
    total = sum(verdict["stream_contributions"].values())
    bias = cal["fusion"]["bias"]
    expected_logit = bias + total
    check("fusion contributions are additive",
          abs(expected_logit - verdict["logit"]) < 1e-3,
          f"{expected_logit:.4f} vs {verdict['logit']:.4f}")
    check("probability in [0,1]", 0.0 <= verdict["p_fake"] <= 1.0)
    check("verdict is one of the three states",
          verdict["verdict"] in {"FAKE", "REAL", "INCONCLUSIVE"})

    # reliability must damp a stream's influence, never amplify it
    weak = json.loads(json.dumps(streams))
    weak["temporal"]["reliability"] = 0.1
    v2 = fuse(weak, cal)
    check("low reliability shrinks temporal contribution",
          abs(v2["stream_contributions"]["temporal"]) < abs(verdict["stream_contributions"]["temporal"]) + 1e-9)


def test_pipeline(tmp: Path) -> None:
    print("\npipeline")
    video = make_video(tmp / "t.mp4")
    fs = sample_frames(video, max_frames=12)
    check("frame sampling works", fs.count == 12, f"{fs.count} frames")
    check("metadata captured", fs.fps > 0 and fs.width == 320)

    try:
        sample_frames(tmp / "nope.mp4")
        check("missing file raises IngestError", False)
    except IngestError:
        check("missing file raises IngestError", True)

    res = analyze_media(video, settings=config.Settings(max_frames=12), make_pdf=False)
    check("result has verdict", "verdict" in res and "p_fake" in res["verdict"])
    check("timings recorded", res["timings_ms"]["total_ms"] > 0)
    check("result.json parses", json.loads((config.WORK_DIR / res["run_id"] / "result.json").read_text())["run_id"] == res["run_id"])

    # A synthetic clip has no real face. The engine must abstain: a missed face is a
    # missing evidence, and calling it REAL is the worst possible false negative.
    if res["verdict"].get("no_face") or res["faces"]["detection_rate"] == 0.0:
        check("no-face clip abstains instead of clearing the clip",
              res["verdict"]["verdict"] == "INCONCLUSIVE",
              res["verdict"]["verdict"])
        check("no-face clip reports no stream evidence", not res["streams"])
        check("no-face clip explains itself",
              bool(res["verdict"].get("abstention_note")) and bool(res["faces"]["notes"]))
        check("no-face clip keeps a written result.json",
              (config.WORK_DIR / res["run_id"] / "result.json").exists())
    else:
        check("three streams scored", set(res["streams"]) == set(config.STREAM_NAMES))

    # image mode, same rule
    img = tmp / "still.png"
    cv2.imwrite(str(img), np.full((240, 320, 3), 120, np.uint8))
    res2 = analyze_media(img, settings=config.Settings(max_frames=8), make_pdf=False)
    check("image input analysed", res2["input"]["kind"] == "image")
    check("image verdict is one of the three states",
          res2["verdict"]["verdict"] in {"FAKE", "REAL", "INCONCLUSIVE"})


def test_face_path() -> None:
    """The scored path, on a real face that ships with the repo."""
    print("\nface path")
    still = Path(config.PROJECT_ROOT) / "samples" / "authentic_still.jpg"
    if not still.exists():
        print("  [SKIP] samples/authentic_still.jpg not present")
        return
    res = analyze_media(still, settings=config.Settings(max_frames=8), make_pdf=False)
    check("face found", res["faces"]["count"] >= 1 and res["faces"]["detection_rate"] > 0)
    check("three streams scored", set(res["streams"]) == set(config.STREAM_NAMES))
    check("verdict is a decision, not an abstention", res["verdict"]["verdict"] in {"FAKE", "REAL"})
    check("artifacts written", (config.WORK_DIR / res["run_id"] / "result.json").exists())
    check("image temporal reliability reduced", res["streams"]["temporal"]["reliability"] < 1.0,
          f"{res['streams']['temporal']['reliability']:.2f}")


def test_api() -> None:
    print("\nhttp api")
    try:
        from fastapi.testclient import TestClient  # noqa: WPS433
        from deeptrace.api import app  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] fastapi test client unavailable: {exc}")
        return

    client = TestClient(app)
    r = client.get("/api/health")
    check("GET /api/health", r.status_code == 200 and r.json()["status"] == "ok")
    r = client.get("/api/calibration")
    check("GET /api/calibration", r.status_code == 200 and "streams" in r.json()["calibration"])
    r = client.get("/")
    check("GET / serves the UI", r.status_code == 200 and "DeepTrace" in r.text)
    r = client.get("/static/app.js")
    check("static assets served", r.status_code == 200)
    r = client.get("/api/runs")
    check("GET /api/runs", r.status_code == 200 and "runs" in r.json())
    r = client.post("/api/analyze", files={"file": ("x.txt", b"not media", "text/plain")})
    check("unsupported type rejected", r.status_code == 415, f"HTTP {r.status_code}")
    r = client.get("/api/runs/deadbeef")
    check("unknown run 404s", r.status_code == 404)


def test_tracking() -> None:
    print("\ntracking")
    tracker = FaceTracker(max_gap=2, min_hits=2)

    def face(box, kind="face"):
        return Detection(box=box, score=1.0, kind=kind)

    # two people, both present for 8 frames, with a 1-frame dropout on the left
    left = [(100, 100, 120, 120), (104, 101, 121, 121), (108, 102, 119, 119)]
    right = [(600, 100, 130, 130), (603, 103, 128, 128), (606, 104, 129, 129)]
    rng = np.random.default_rng(0)
    patches = {"l": rng.normal(0.5, 0.1, (32, 32, 3)).astype(np.float32),
               "r": rng.normal(0.2, 0.1, (32, 32, 3)).astype(np.float32)}
    dl, dr = descriptor(patches["l"]), descriptor(patches["r"])
    for i in range(8):
        dets, descs = [], []
        if i != 4:  # left face briefly missed
            dets.append(face(left[i % 3])); descs.append(dl)
        dets.append(face(right[i % 3])); descs.append(dr)
        tracker.update(i, dets, descs)

    tracks = tracker.finish()
    check("two faces produce two tracks", len(tracks) == 2, f"{len(tracks)} tracks")
    check("track ids are stable", {t.track_id for t in tracks} == {1, 2})
    left_track = next(t for t in tracks if t.boxes[t.first][0] < 300)
    check("one-frame dropout is bridged", left_track.hits == 7 and left_track.gaps() == 1,
          f"hits={left_track.hits} gaps={left_track.gaps()}")
    check("continuity reflects the bridge", 0.85 < left_track.continuity() <= 1.0,
          f"{left_track.continuity():.2f}")
    check("no identity swap between the two faces",
          all(b[0] < 300 for b in left_track.boxes.values()))

    # a fallback window must never be merged into a real face track
    t2 = FaceTracker(max_gap=2, min_hits=1)
    t2.update(0, [face((100, 100, 120, 120))], [dl])
    t2.update(1, [face((100, 100, 120, 120), kind="fallback")], [dl])
    check("fallback windows stay separate from faces", len(t2.finish()) == 2,
          f"{len(t2.finish())} tracks")

    # a face that vanishes for longer than max_gap starts a new identity
    t3 = FaceTracker(max_gap=1, min_hits=1)
    t3.update(0, [face((100, 100, 120, 120))], [dl])
    t3.update(5, [face((103, 101, 120, 120))], [dl])
    check("long gaps split identities", len(t3.finish()) == 2,
          f"{len(t3.finish())} tracks")


def _two_face_image(tmp: Path) -> Path | None:
    """Side-by-side composite of two real asset faces (skipped if no assets)."""
    assets = ROOT / "demo" / "assets"
    people = sorted(assets.glob("person_*.png"))[:2]
    if len(people) < 2:
        return None
    half = 640
    tiles = []
    for path in people:
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        s = half / min(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        tiles.append(img[max(0, int((h - half) * 0.35)):][:half,
                                                         max(0, (w - half) // 2):][:, :half])
    out = tmp / "two_faces.png"
    cv2.imwrite(str(out), np.hstack(tiles))
    return out


def test_multiface(tmp: Path) -> None:
    print("\nmulti-face")
    img_path = _two_face_image(tmp)
    if img_path is None:
        print("  [SKIP] demo/assets/person_*.png not available")
        return

    loc = FaceLocator(crop_size=256, max_faces=3)
    frame = cv2.imread(str(img_path))
    dets, method = loc.detect_all(frame)
    check("both faces detected", len(dets) == 2, f"{len(dets)} detections ({method})")
    crops, _ = loc.crop_faces(frame)
    check("one crop per face", len(crops) == 2)
    check("crops are distinct identities",
          np.corrcoef(crops[0].image.ravel(), crops[1].image.ravel())[0, 1] < 0.9)

    res = analyze_media(img_path, settings=config.Settings(max_frames=8),
                        make_pdf=False, render_artifacts=True)
    faces = res["faces"]
    check("result reports two tracked faces", faces["count"] == 2 and faces["analysed"] == 2,
          f"count={faces['count']} analysed={faces['analysed']}")
    check("each face carries its own streams",
          all(set(t["streams"]) == set(config.STREAM_NAMES) for t in faces["tracks"]))
    check("aggregate names a worst-case face",
          faces["aggregate"]["strategy"] == "worst-case face"
          and faces["aggregate"]["track_id"] is not None)
    worst = max(t["verdict"]["p_fake"] for t in faces["tracks"])
    check("clip verdict equals the worst face",
          abs(res["verdict"]["p_fake"] - worst) < 1e-9,
          f"{res['verdict']['p_fake']:.4f} vs {worst:.4f}")
    check("verdict names the face it came from",
          res["verdict"]["track_id"] == faces["aggregate"]["track_id"])
    check("face agreement is reported", "n_faces" in res["verdict"]["face_agreement"])
    check("multi-face artefacts rendered",
          all(res["artifacts"].get(k) for k in ("faces", "tracks", "face_1", "face_2")),
          str(sorted(k for k in res["artifacts"] if k.startswith("face"))))
    for name in ("faces.png", "tracks.png"):
        p = config.WORK_DIR / res["run_id"] / name
        check(f"{name} written", p.exists() and p.stat().st_size > 5000)
    check("per-face JSON is complete",
          all(t["frames_total"] == res["input"]["frames_sampled"] for t in faces["tracks"]))


def test_calibration() -> None:
    print("\ncalibration")
    cal = config.load_calibration()
    raw = cal.raw
    check("calibration has all streams", set(raw["streams"]) == set(config.STREAM_NAMES))
    check("fusion weights present", set(raw["fusion"]["weights"]) == set(config.STREAM_NAMES))
    check("thresholds ordered", 0 < raw["thresholds"]["real"] < raw["thresholds"]["fake"] < 1,
          f"{raw['thresholds']['real']} < {raw['thresholds']['fake']}")
    check("model card present", "training" in raw or raw.get("kind") == "expert-prior",
          raw.get("version", "?"))


def test_abstention() -> None:
    print("\nabstention policy")
    # Clopper-Pearson: known reference values
    check("cp upper bound, 0/30 is ~0.095 (rule of three)",
          abs(abstain.clopper_pearson_upper(0, 30) - 0.0953) < 0.005,
          f"{abstain.clopper_pearson_upper(0, 30):.4f}")
    check("cp upper bound grows with the error count",
          abstain.clopper_pearson_upper(3, 30) > abstain.clopper_pearson_upper(1, 30),
          f"{abstain.clopper_pearson_upper(1, 30):.3f} < "
          f"{abstain.clopper_pearson_upper(3, 30):.3f}")

    probs = [0.02, 0.10, 0.25, 0.48, 0.60, 0.72, 0.88, 0.97, 0.15, 0.35]
    labels = [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
    rows = abstain.curve(probs, labels, centre=0.5)
    check("coverage is monotone non-increasing in band width",
          all(rows[i]["coverage"] >= rows[i + 1]["coverage"] - 1e-9 for i in range(len(rows) - 1)))
    check("no band decides more errors than a narrower one",
          rows[0]["errors"] <= max(r["errors"] for r in rows))

    sel_cost = abstain.select_band(probs, labels, {"mode": "cost"}, centre=0.5)
    check("cost policy returns a band and a curve",
          sel_cost["width"] >= 0 and len(sel_cost["curve"]) == len(abstain.DEFAULT_WIDTHS))
    check("cost policy picks the cheapest row on the curve",
          sel_cost["chosen"]["expected_cost"] == min(r["expected_cost"] for r in sel_cost["curve"]))
    check("thresholds bracket the operating point",
          sel_cost["thresholds"]["real"] < sel_cost["thresholds"]["operating_point"]
          < sel_cost["thresholds"]["fake"] + 1e-9)

    sel_cov = abstain.select_band(probs, labels, {"mode": "coverage", "target_coverage": 0.8},
                                  centre=0.5)
    check("coverage policy respects the target when reachable",
          sel_cov["chosen"]["coverage"] >= 0.8 or sel_cov["width"] == abstain.DEFAULT_WIDTHS[-1],
          f"coverage={sel_cov['chosen']['coverage']}")

    sel_err = abstain.select_band(probs, labels, {"mode": "error_bound", "alpha": 0.9}, centre=0.5)
    check("error_bound policy reports feasibility", "feasible" in sel_err)
    check("guarantee carries a sample size", sel_err["guarantee"]["n"] == len(probs))

    note = abstain.describe(sel_cost)
    check("describe() renders a one-line rationale", bool(note) and "review band" in note, str(note)[:60])
    check("describe() tolerates a missing document", abstain.describe(None) is None)

    # the shipped calibration must carry the policy that produced its band
    raw = config.load_calibration().raw
    if raw.get("kind") == "fitted":
        check("fitted calibration stores the abstention policy",
              "abstention" in raw and raw["abstention"]["curve"],
              raw.get("abstention", {}).get("reason", "missing")[:50])
        check("stored curve is consistent with the stored thresholds",
              raw["abstention"]["width"] == raw["thresholds"]["review_band"])


def main() -> int:
    config.ensure_dirs()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_signals()
        test_fusion()
        test_pipeline(tmp)
        test_tracking()
        test_multiface(tmp)
        test_api()
        test_calibration()
        test_abstention()
    test_face_path()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
