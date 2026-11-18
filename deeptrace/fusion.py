"""Evidence fusion and verdict calibration.

Each stream is mapped to a synthetic-likelihood score with a logistic model over
standardised features, then the three stream scores are combined in logit space:

    logit(p) = b + SUM_i  w_i * r_i * (2 * s_i - 1)

where ``r_i`` is the *reliability* of stream i for this particular input (for
example, a 12-frame clip cannot support pulse evidence, so the temporal stream is
down-weighted).  Every term in that sum is an exact, additive contribution, so
the reported explanation is arithmetic - not a post-hoc approximation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

STREAM_TITLES = {
    "spatial": "Spatial domain",
    "frequency": "Frequency domain",
    "temporal": "Temporal domain",
}


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def score_stream(
    name: str, features: Dict[str, float], calibration: Dict[str, Any]
) -> Dict[str, Any]:
    """Map raw stream features to a calibrated synthetic-likelihood score."""
    cfg = calibration["streams"][name]
    labels = calibration.get("evidence_labels", {})
    z_terms: Dict[str, float] = {}
    raw_contrib: Dict[str, float] = {}
    for feat, spec in cfg["features"].items():
        if feat not in features:
            continue
        value = float(features[feat])
        z = (value - float(spec["mean"])) / max(float(spec["sd"]), 1e-6)
        z = float(np.clip(z, -4.0, 4.0))
        term = float(spec["coef"]) * z
        z_terms[feat] = z
        raw_contrib[feat] = term
    logit = float(cfg.get("bias", 0.0)) + sum(raw_contrib.values())
    score = _sigmoid(logit)
    return {
        "stream": name,
        "title": STREAM_TITLES.get(name, name),
        "score": round(score, 4),
        "logit": round(logit, 4),
        "reliability": 1.0,
        "features": {k: round(float(v), 5) for k, v in features.items()},
        "z_scores": {k: round(v, 3) for k, v in z_terms.items()},
        "feature_contributions": {k: round(v, 4) for k, v in raw_contrib.items()},
        "evidence": _feature_evidence(raw_contrib, labels),
    }


def _feature_evidence(contrib: Dict[str, float], labels: Dict[str, str]) -> List[Dict[str, Any]]:
    items = []
    for feat, term in contrib.items():
        items.append(
            {
                "feature": feat,
                "label": labels.get(feat, feat),
                "logit_contribution": round(term, 4),
                "direction": "synthetic" if term > 0 else "authentic",
                "strength": round(min(abs(term) / 1.5, 1.0), 3),
            }
        )
    return sorted(items, key=lambda i: -abs(i["logit_contribution"]))


def fuse(
    streams: Dict[str, Dict[str, Any]],
    calibration: Dict[str, Any],
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Combine stream scores into a calibrated verdict with additivity kept intact."""
    fusion_cfg = calibration["fusion"]
    th = thresholds or calibration.get("thresholds", {"fake": 0.65, "real": 0.35})

    contributions: Dict[str, float] = {}
    effective_weights: Dict[str, float] = {}
    for name, stream in streams.items():
        w = float(fusion_cfg["weights"].get(name, 0.0))
        r = float(stream.get("reliability", 1.0))
        w_eff = w * r
        contributions[name] = w_eff * (2.0 * float(stream["score"]) - 1.0)
        effective_weights[name] = round(w_eff, 4)

    logit = float(fusion_cfg.get("bias", 0.0)) + sum(contributions.values())
    p_fake = _sigmoid(logit)
    p_real = 1.0 - p_fake

    fake_t, real_t = float(th["fake"]), float(th["real"])
    if p_fake >= fake_t:
        verdict = "FAKE"
    elif p_fake <= real_t:
        verdict = "REAL"
    else:
        verdict = "INCONCLUSIVE"

    scores = [float(s["score"]) for s in streams.values()]
    spread = float(max(scores) - min(scores)) if scores else 0.0
    mean_reliability = float(np.mean([s.get("reliability", 1.0) for s in streams.values()])) if streams else 1.0

    # confidence: distance from the decision boundary, penalised by disagreement
    margin = abs(p_fake - 0.5) * 2.0
    confidence = float(np.clip(margin * (1.0 - 0.45 * min(spread, 1.0)) * (0.7 + 0.3 * mean_reliability), 0.0, 1.0))

    ranking = sorted(contributions.items(), key=lambda kv: -kv[1])
    leading = ranking[0][0] if ranking else None

    return {
        "verdict": verdict,
        "p_fake": round(p_fake, 4),
        "p_real": round(p_real, 4),
        "confidence": round(confidence, 4),
        "logit": round(logit, 4),
        "thresholds": {"fake": fake_t, "real": real_t},
        "stream_contributions": {k: round(v, 4) for k, v in contributions.items()},
        "effective_weights": effective_weights,
        "leading_stream": leading,
        "stream_spread": round(spread, 4),
        "mixed_evidence": bool(spread > 0.55),
        "calibration_version": calibration.get("version", "unknown"),
    }


def apply_reliability(stream: Dict[str, Any], reliability: float, reason: str | None = None) -> Dict[str, Any]:
    """Record how trustworthy a stream's score is for this input."""
    reliability = float(np.clip(reliability, 0.0, 1.0))
    stream["reliability"] = round(reliability, 3)
    if reason:
        stream["reliability_note"] = reason
    return stream


def summarise(streams: Dict[str, Dict[str, Any]], verdict: Dict[str, Any]) -> str:
    """One-paragraph, human-readable explanation of the verdict."""
    lead = verdict.get("leading_stream") or "spatial"
    title = STREAM_TITLES.get(lead, lead)
    p = verdict["p_fake"]
    if verdict["verdict"] == "FAKE":
        head = f"Face-swap indicators dominate: the fused fake-probability is {p:.2f}."
    elif verdict["verdict"] == "REAL":
        head = f"No consistent swap signature: the fused fake-probability is {p:.2f}."
    else:
        head = f"Evidence is inconclusive (fake-probability {p:.2f}) - refer for manual review."
    top_bits = []
    for name, stream in streams.items():
        ev = stream.get("evidence", [])
        if ev:
            top = ev[0]
            top_bits.append(f"{STREAM_TITLES.get(name, name)}: {top['label'].lower()} "
                            f"({top['direction']}, {top['logit_contribution']:+.2f})")
    return f"{head} Leading stream: {title}. " + "; ".join(top_bits) + "."
