"""Command-line interface.

    python -m deeptrace.cli samples/fake_one_shot.mp4
    python -m deeptrace.cli clip.mp4 --json out.json --quiet
    python -m deeptrace.cli batch ./clips --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import config
from .engine import analyze_media

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _print_result(res: dict) -> None:
    v = res["verdict"]
    bar = "=" * 62
    print(bar)
    print(f"  {config.APP_NAME} v{res['engine']['version']}  |  run {res['run_id']}")
    print(bar)
    print(f"  file        : {res['input']['filename']}  ({res['input']['kind']}, "
          f"{res['input']['resolution']}, {res['input']['frames_sampled']} frames sampled)")
    print(f"  verdict     : {v['verdict']}   p(fake)={v['p_fake']:.3f}  confidence={v['confidence']:.3f}")
    print(f"  summary     : {v['summary']}")
    print("-" * 62)
    for name, s in res["streams"].items():
        print(f"  {s['title']:<18} score={s['score']:.3f}  reliability={s.get('reliability', 1):.2f}  "
              f"contribution={v['stream_contributions'].get(name, 0):+.3f}")
    print("-" * 62)
    for key, path in res["artifacts"].items():
        print(f"  {key:<10} {path}")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deeptrace", description=config.APP_TAGLINE)
    ap.add_argument("input", help="video/image file, or a directory with --batch")
    ap.add_argument("--max-frames", type=int, default=config.MAX_FRAMES)
    ap.add_argument("--json", type=str, default=None, help="write the result JSON here")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF report generation")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--batch", action="store_true", help="analyse every media file in the directory")
    ap.add_argument("--csv", type=str, default=None, help="batch mode: write a CSV summary here")
    args = ap.parse_args(argv)

    settings = config.Settings(max_frames=max(4, min(args.max_frames, 96)))
    target = Path(args.input)

    if args.batch:
        if not target.is_dir():
            ap.error("--batch needs a directory")
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in VIDEO_EXT)
        rows = []
        for f in files:
            try:
                res = analyze_media(f, settings=settings, make_pdf=not args.no_pdf)
            except Exception as exc:  # noqa: BLE001 - batch must not die on one file
                print(f"  !! {f.name}: {exc}", file=sys.stderr)
                continue
            v = res["verdict"]
            faces = res.get("faces") or {}
            per_face = " ".join(
                f"#{t['track_id']}:{t['verdict']['verdict']}"
                for t in faces.get("tracks", []) if t.get("analysed") and t.get("verdict")
            )
            rows.append({
                "file": f.name, "verdict": v["verdict"], "p_fake": v["p_fake"],
                "confidence": v["confidence"],
                "faces": faces.get("count", 1), "faces_scored": faces.get("analysed", 1),
                "per_face": per_face or "-",
                "spatial": res["streams"]["spatial"]["score"],
                "frequency": res["streams"]["frequency"]["score"],
                "temporal": res["streams"]["temporal"]["score"],
                "run_id": res["run_id"],
            })
            if not args.quiet:
                extra = f"  [{per_face}]" if faces.get("analysed", 1) > 1 else ""
                print(f"  {f.name:<34} {v['verdict']:<12} p(fake)={v['p_fake']:.3f}{extra}")
        out = Path(args.csv or "batch_results.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {out} ({len(rows)} files)")
        return 0

    try:
        res = analyze_media(target, settings=settings, make_pdf=not args.no_pdf)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not args.quiet:
        _print_result(res)
    else:
        print(json.dumps({"verdict": res["verdict"]["verdict"], "p_fake": res["verdict"]["p_fake"],
                          "run_id": res["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
