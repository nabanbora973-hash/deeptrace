"""Forensic PDF report.

Produces the document an analyst can attach to a case file: verdict, calibrated
confidence, per-stream scores with reliability weights, the individual evidence
items with their logit contributions, the rendered artefacts, and the method
disclaimer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:  # PDF reporting is optional; JSON + images are the core output
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False
    colors = None
    A4 = None
    mm = 1.0

INK = colors.HexColor("#10151d")
MUTED = colors.HexColor("#5b6b7f")
ACCENT = colors.HexColor("#0f766e")
DANGER = colors.HexColor("#b91c1c")
AMBER = colors.HexColor("#a16207")
LINE = colors.HexColor("#d7dee6")


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                                leading=24, textColor=INK, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                             textColor=INK, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK),
        "small": ParagraphStyle("sm", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED),
    }


def _verdict_colour(verdict: str):
    return {"FAKE": DANGER, "REAL": ACCENT}.get(verdict, AMBER)


def _table(data, widths, header=True, style_extra=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f7")),
        ]
    if style_extra:
        cmds += style_extra
    t.setStyle(TableStyle(cmds))
    return t


def build_pdf(result: Dict[str, Any], out_path: Path, images: Dict[str, Path] | None = None) -> Path:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed - run: pip install reportlab")
    st = _styles()
    verdict = result["verdict"]
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"DeepTrace forensic report - {result['run_id']}", author="DeepTrace",
    )
    flow = []
    vc = _verdict_colour(verdict["verdict"])

    flow.append(Paragraph("DeepTrace - Forensic Analysis Report", st["title"]))
    flow.append(Paragraph(
        f"Run <b>{result['run_id']}</b> &nbsp;|&nbsp; "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; "
        f"engine v{result['engine']['version']} &nbsp;|&nbsp; "
        f"calibration {verdict.get('calibration_version', 'n/a')}", st["sub"]))
    flow.append(Spacer(1, 8))

    # verdict banner
    banner = Table([[
        Paragraph(f"<font size=15 color='white'><b>{verdict['verdict']}</b></font>", st["body"]),
        Paragraph(
            f"<font size=8 color='white'>fake-probability <b>{verdict['p_fake']:.3f}</b> &nbsp; "
            f"confidence <b>{verdict['confidence']:.3f}</b> &nbsp; "
            f"thresholds {verdict['thresholds']['real']:.2f}/{verdict['thresholds']['fake']:.2f}</font>",
            st["body"]),
    ]], colWidths=[38 * mm, 118 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), vc),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    flow.append(banner)
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(verdict.get("summary", ""), st["body"]))
    note = verdict.get("abstention_note")
    if note:
        flow.append(Spacer(1, 3))
        flow.append(Paragraph(
            f"<font size=7.5 color='#555555'>Review-band policy: {note}. "
            "Anything inside the band is reported INCONCLUSIVE rather than forced into a "
            "confident answer.</font>", st["body"]))

    # input + faces
    inp = result["input"]
    flow.append(Paragraph("1. Input and pre-processing", st["h2"]))
    rows = [
        ["File", inp["filename"], "Resolution", inp["resolution"]],
        ["Kind", inp["kind"], "Duration", f"{inp['duration_s']} s"],
        ["Frame rate", f"{inp['fps']} fps", "Frames sampled", f"{inp['frames_sampled']} / {inp['frames_total']}"],
        ["Face detected", f"{result['faces']['detected_frames']} frames "
                          f"({result['faces']['detection_rate'] * 100:.0f}%)",
         "Detector", result["faces"]["method"]],
        ["Faces tracked", f"{result['faces'].get('count', 1)} "
                          f"({result['faces'].get('analysed', 1)} scored)",
         "Concurrent faces", f"{result['faces'].get('max_simultaneous', 1)} max"],
    ]
    flow.append(_table(rows, [30 * mm, 55 * mm, 32 * mm, 39 * mm], header=False))

    # per-face table (the face-level answer the clip verdict is built on)
    tracks = result["faces"].get("tracks") or []
    if tracks:
        flow.append(Paragraph("2. Faces analysed", st["h2"]))
        frows = [["Face", "Verdict", "p(fake)", "Frames", "Continuity", "Role"]]
        for t in tracks:
            v = t.get("verdict")
            frows.append([
                f"#{t['track_id']}",
                v["verdict"] if v else "not scored",
                f"{v['p_fake']:.3f}" if v else "-",
                f"{t['frames']} / {t.get('frames_total', t['frames'])}",
                f"{t.get('continuity', 0):.2f}",
                "primary" if t.get("role") == "primary" else
                ("verdict source" if t.get("verdict_source") else "secondary"),
            ])
        flow.append(_table(frows, [16 * mm, 30 * mm, 22 * mm, 28 * mm, 26 * mm, 34 * mm],
                           style_extra=[
                               ("TEXTCOLOR", (1, 1), (1, -1), INK),
                           ]))
        agg = result["faces"].get("aggregate") or {}
        if agg.get("n_analysed", 0) > 1:
            flow.append(Spacer(1, 3))
            flow.append(Paragraph(
                f"Multi-face rule: the clip verdict is the worst-case face "
                f"(face #{agg.get('track_id')}, p={agg.get('p_fake'):.3f}). "
                f"Faces disagree by {agg.get('disagreement', 0):.2f} in fake-probability - "
                f"one manipulated identity is enough to make the recording suspect, so the "
                f"per-face numbers above are reported rather than averaged.", st["small"]))
        else:
            flow.append(Spacer(1, 3))
            flow.append(Paragraph(
                "One face carried the verdict; no disagreement to report.", st["small"]))

    # stream table
    flow.append(Paragraph("3. Evidence streams", st["h2"]))
    srows = [["Stream", "Score", "Reliability", "Fusion contribution", "Top finding"]]
    for name, s in result["streams"].items():
        top = s.get("evidence", [{}])[0] if s.get("evidence") else {}
        srows.append([
            s["title"], f"{s['score']:.3f}", f"{s.get('reliability', 1.0):.2f}",
            f"{verdict['stream_contributions'].get(name, 0.0):+.3f}",
            top.get("label", "-"),
        ])
    flow.append(_table(srows, [34 * mm, 18 * mm, 24 * mm, 32 * mm, 48 * mm]))
    flow.append(Spacer(1, 3))
    flow.append(Paragraph(
        "Fusion: logit(p) = bias + sum(weight x reliability x (2 x score - 1)). "
        "Contributions are additive, so the table above reconstructs the score exactly.", st["small"]))

    # evidence detail
    flow.append(Paragraph("4. Evidence detail", st["h2"]))
    erows = [["Stream", "Signal", "Value", "z", "Logit contribution", "Direction"]]
    for name, s in result["streams"].items():
        for item in s.get("evidence", [])[:4]:
            feat = item["feature"]
            erows.append([
                s["title"], item["label"], f"{s['features'].get(feat, 0):.4f}",
                f"{s['z_scores'].get(feat, 0):+.2f}",
                f"{item['logit_contribution']:+.3f}", item["direction"],
            ])
    flow.append(_table(erows, [27 * mm, 60 * mm, 20 * mm, 16 * mm, 30 * mm, 24 * mm]))

    # temporal
    tstats = result.get("temporal", {})
    flow.append(Paragraph("5. Temporal stream diagnostics", st["h2"]))
    flow.append(_table([
        ["Frames used", str(tstats.get("frames_used", "-")),
         "Temporal support", f"{tstats.get('temporal_support', 0):.2f}"],
        ["Pulse estimate", f"{result['streams']['temporal']['features'].get('pulse_bpm', 0):.0f} bpm "
                           f"(strength {result['streams']['temporal']['features'].get('pulse_strength', 0):.2f})",
         "Identity jitter", f"{result['streams']['temporal']['features'].get('identity_jitter', 0):.3f}"],
    ], [32 * mm, 58 * mm, 32 * mm, 35 * mm], header=False))

    # artefacts
    if images:
        flow.append(Paragraph("6. Rendered evidence", st["h2"]))
        for key, caption in (("overlay", "Fused artifact heat-map"),
                             ("timeline", "Per-frame seam energy"),
                             ("spectrum", "Frequency spectrum"),
                             ("streams", "Stream scores"),
                             ("faces", "Tracked faces with per-face verdicts"),
                             ("tracks", "Face identity across the sampled timeline")):
            p = images.get(key)
            if not p or not Path(p).exists():
                continue
            img = Image(str(p))
            max_w = 168 * mm
            ratio = max_w / img.imageWidth
            img.drawWidth = max_w
            img.drawHeight = img.imageHeight * ratio
            if img.drawHeight > 92 * mm:
                img.drawHeight = 92 * mm
                img.drawWidth = img.imageWidth * (92 * mm / img.imageHeight)
            flow.append(KeepTogether([Paragraph(caption, st["small"]), img, Spacer(1, 6)]))

    flow.append(Spacer(1, 8))
    flow.append(Paragraph("7. Limitations", st["h2"]))
    flow.append(Paragraph(
        "This report describes statistical forensic evidence, not proof of manipulation. "
        "Artifact-based detectors can be defeated by adversarial post-processing (heavy "
        "re-encoding, strong denoising, resizing). Absence of a detected artifact does not "
        "establish authenticity: provenance metadata (e.g. C2PA) and source verification "
        "remain the stronger evidence class. The verdict must be read together with the "
        "per-stream reliability weights, especially for short clips where the temporal "
        "stream is down-weighted. Faces seen in only a handful of frames are listed but "
        "not scored, and a face that leaves and re-enters the frame can be reported as two "
        "identities.", st["body"]))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(result.get("disclaimer", ""), st["small"]))
    flow.append(Paragraph(
        f"Timings (ms): " + ", ".join(f"{k}={v}" for k, v in result.get("timings_ms", {}).items()), st["small"]))

    doc.build(flow)
    return out_path
