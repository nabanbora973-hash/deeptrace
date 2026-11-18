"""FastAPI service: upload -> analyse -> JSON verdict + PDF report.

Endpoints
---------
GET  /                      interactive console (web/index.html)
GET  /api/health            liveness + engine/calibration metadata
GET  /api/calibration       the exact decision document in force
POST /api/analyze           multipart video/image -> full result JSON
GET  /api/runs/{id}         stored result JSON for a previous run
GET  /runs/{id}/{file}      rendered artifacts (overlay, timeline, report.pdf)

Run locally:  uvicorn deeptrace.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .engine import analyze_media
from .ingest import IngestError

config.ensure_dirs()

app = FastAPI(
    title=f"{config.APP_NAME} API",
    version=config.APP_VERSION,
    description=config.APP_TAGLINE,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/runs", StaticFiles(directory=str(config.WORK_DIR)), name="runs")
app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")
app.mount("/samples", StaticFiles(directory=str(config.SAMPLE_DIR)), name="samples")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    cal = config.load_calibration()
    return {
        "status": "ok",
        "engine": {"name": config.APP_NAME, "version": config.APP_VERSION},
        "calibration": {"version": cal.raw.get("version"), "kind": cal.raw.get("kind"), "source": cal.source},
        "streams": list(config.STREAM_NAMES),
        "faces": {
            "max_faces": config.MAX_FACES,
            "min_track_frames": config.MIN_TRACK_FRAMES,
            "track_max_gap": config.TRACK_MAX_GAP,
            "aggregation": "worst-case face",
        },
        "thresholds": cal.raw.get("thresholds"),
        "runs_dir": str(config.WORK_DIR),
    }


@app.get("/api/calibration")
def calibration() -> JSONResponse:
    cal = config.load_calibration()
    return JSONResponse({"source": cal.source, "calibration": cal.raw})


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    max_frames: int = Form(config.MAX_FRAMES),
    max_faces: int = Form(config.MAX_FACES),
    make_pdf: bool = Form(True),
) -> JSONResponse:
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f"unsupported file type: {suffix or 'none'}")

    tmpdir = Path(tempfile.mkdtemp(prefix="deeptrace_"))
    tmp_path = tmpdir / f"upload{suffix}"
    size = 0
    try:
        with tmp_path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="file exceeds 200 MB limit")
                fh.write(chunk)
        settings = config.Settings(max_frames=max(4, min(int(max_frames), 96)),
                                   max_faces=max(1, min(int(max_faces), 8)))
        result = analyze_media(tmp_path, settings=settings, run_id=uuid.uuid4().hex[:12], make_pdf=make_pdf)
        result["input"]["bytes"] = size
        runs_json = config.WORK_DIR / result["run_id"] / "result.json"
        runs_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return JSONResponse(result)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/eval")
def evaluation() -> JSONResponse:
    """Serving the last evaluation run so the UI never hard-codes a metric."""
    cal = config.load_calibration().raw
    out: Dict[str, Any] = {
        "trained": cal.get("version"),
        "kind": cal.get("kind"),
        "samples": None,
        "in_corpus": None,
        "cross_corpus": None,
    }
    held = cal.get("training", {}).get("held_out", {})
    if held.get("fused"):
        out["in_corpus"] = held["fused"].get("auc")
    training = cal.get("training", {}).get("samples")
    if training:
        out["samples"] = f"{training['real']} real / {training['fake']} fake"
    path = config.DATA_DIR / "eval_hard.json"
    if path.exists():
        try:
            out["cross_corpus"] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return JSONResponse(out)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    if not run_id.isalnum():
        raise HTTPException(status_code=400, detail="invalid run id")
    path = config.WORK_DIR / run_id / "result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/runs")
def list_runs(limit: int = 20) -> JSONResponse:
    runs = sorted(config.WORK_DIR.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in runs[: max(1, min(limit, 100))]:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "run_id": r["run_id"],
                "filename": r["input"]["filename"],
                "verdict": r["verdict"]["verdict"],
                "p_fake": r["verdict"]["p_fake"],
                "kind": r["input"]["kind"],
                "faces": (r.get("faces") or {}).get("count", 1),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return JSONResponse({"runs": out})
