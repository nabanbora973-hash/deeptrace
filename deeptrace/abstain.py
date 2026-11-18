"""Abstention policy: where the FAKE / INCONCLUSIVE / REAL lines actually sit.

A verdict of ``INCONCLUSIVE`` is only honest if the *width* of the review band is a stated
decision rather than a magic number.  This module turns the band into a policy:

* ``cost``        - minimise expected cost per clip from a cost matrix
                    (false accusation vs missed swap vs cost of a human review);
* ``coverage``    - abstain as little as possible while deciding at least a target share
                    of clips;
* ``error_bound`` - widen the band until the *upper confidence bound* on the error rate
                    among decided clips is at or below ``alpha`` (Clopper-Pearson), then
                    keep the narrowest band that satisfies it.

Everything is computed from out-of-fold scores, so the curve that justifies the choice is
also the curve that can be published.  The selection is stored in ``calibration.json`` next
to the thresholds, which means the runtime never invents a band on its own.

    python train/fit_calibration.py --features data/features.csv --policy cost
    python scripts/risk_coverage.py data/features_hard.csv
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_WIDTHS: Tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(0, 41))

DEFAULT_POLICY: Dict[str, Any] = {
    "mode": "cost",
    "cost_fp": 10.0,      # calling an authentic clip FAKE ("your video is fake")
    "cost_fn": 3.0,       # missing a swap that review would have caught
    "cost_review": 2.0,   # a human has to look at an abstention
    "target_coverage": 0.85,
    "alpha": 0.10,
}


# --------------------------------------------------------------------------- statistics


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p).  Pure-python so the runtime needs no scipy."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(0, min(k, n) + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(total, 1.0)


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper bound on a binomial rate (exact, via bisection on the CDF).

    ``k`` errors out of ``n`` decided clips -> the largest rate ``p`` still consistent with
    seeing ``k`` or fewer errors with probability ``1 - conf``.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    target = 1.0 - conf  # we want P(X <= k | p) == 1 - conf
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binomial_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- curve


def band(centre: float, width: float) -> Tuple[float, float]:
    """Thresholds implied by an operating point and a symmetric half-width."""
    return max(0.0, centre - width), min(1.0, centre + width)


def curve(
    probs: Sequence[float],
    labels: Sequence[int],
    centre: float,
    widths: Iterable[float] = DEFAULT_WIDTHS,
) -> List[Dict[str, Any]]:
    """Risk-coverage table over band widths.

    ``labels``: 1 = fake, 0 = authentic.  ``fpr`` is the expensive error: an authentic clip
    called FAKE.  ``expected_cost`` is per clip, using the policy's cost matrix
    (filled in by :func:`select_band`).
    """
    n = len(probs)
    rows: List[Dict[str, Any]] = []
    for width in widths:
        lo, hi = band(centre, width)
        decided = [(p, y) for p, y in zip(probs, labels) if not (lo < p < hi)]
        abstained = n - len(decided)
        tp = sum(1 for p, y in decided if p >= hi and y == 1)
        fp = sum(1 for p, y in decided if p >= hi and y == 0)
        fn = sum(1 for p, y in decided if p <= lo and y == 1)
        tn = sum(1 for p, y in decided if p <= lo and y == 0)
        n_real = tp + fp + fn + tn
        n_real_pos = sum(1 for _, y in decided if y == 0)
        n_fake_pos = sum(1 for _, y in decided if y == 1)
        errors = fp + fn
        rows.append(
            {
                "width": round(float(width), 3),
                "real_threshold": round(lo, 4),
                "fake_threshold": round(hi, 4),
                "coverage": round(len(decided) / n, 4) if n else 0.0,
                "decided": n_real,
                "abstained": abstained,
                "errors": errors,
                "error_rate": round(errors / len(decided), 4) if decided else None,
                "fpr": round(fp / n_real_pos, 4) if n_real_pos else None,
                "fnr": round(fn / n_fake_pos, 4) if n_fake_pos else None,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
        )
    return rows


# --------------------------------------------------------------------------- selection


def select_band(
    probs: Sequence[float],
    labels: Sequence[int],
    policy: Dict[str, Any] | None = None,
    centre: float | None = None,
    widths: Iterable[float] = DEFAULT_WIDTHS,
) -> Dict[str, Any]:
    """Choose the review band and record why.

    Returns a JSON-serialisable document that is stored under ``abstention`` in the
    calibration file: policy, chosen width, thresholds, the full curve and the guarantee.
    """
    policy = {**DEFAULT_POLICY, **(policy or {})}
    probs = [float(p) for p in probs]
    labels = [int(y) for y in labels]
    if centre is None:
        positives = [p for p, y in zip(probs, labels) if y == 1]
        negatives = [p for p, y in zip(probs, labels) if y == 0]
        centre = 0.5 * ((sum(positives) / len(positives) if positives else 0.5)
                        + (sum(negatives) / len(negatives) if negatives else 0.5))

    rows = curve(probs, labels, centre, widths)
    n = len(probs)
    c_fp, c_fn, c_rev = (float(policy["cost_fp"]), float(policy["cost_fn"]),
                         float(policy["cost_review"]))
    for row in rows:
        per_clip = (c_fp * row["fp"] + c_fn * row["fn"] + c_rev * row["abstained"]) / n if n else 0.0
        row["expected_cost"] = round(per_clip, 4)
        row["cp_upper95"] = (
            round(clopper_pearson_upper(row["errors"], row["decided"]), 4)
            if row["decided"] else None
        )

    mode = str(policy["mode"])
    feasible = True
    if mode == "coverage":
        target = float(policy["target_coverage"])
        candidates = [r for r in rows if r["coverage"] >= target]
        chosen = candidates[0] if candidates else rows[-1]
        reason = f"smallest band that still decides >= {target:.0%} of clips"
    elif mode == "error_bound":
        alpha = float(policy["alpha"])
        candidates = [r for r in rows if r["cp_upper95"] is not None and r["cp_upper95"] <= alpha]
        if candidates:
            chosen = candidates[0]
            reason = (f"narrowest band whose 95% upper bound on error among decided clips "
                      f"is <= {alpha:.0%}")
        else:
            feasible = False
            chosen = min(rows, key=lambda r: (r["cp_upper95"] if r["cp_upper95"] is not None else 1.0,
                                              r["errors"]))
            reason = (f"no band reaches a 95% bound of {alpha:.0%} on this sample; "
                      f"fell back to the tightest bound available")
    else:  # cost
        chosen = min(rows, key=lambda r: (r["expected_cost"], r["width"]))
        reason = ("minimises expected cost per clip: "
                  f"{c_fp:g}x false accusation, {c_fn:g}x missed swap, {c_rev:g}x review")

    return {
        "policy": {**policy, "mode": mode},
        "reason": reason,
        "feasible": feasible,
        "centre": round(float(centre), 4),
        "width": chosen["width"],
        "thresholds": {
            "fake": chosen["fake_threshold"],
            "real": chosen["real_threshold"],
            "operating_point": round(float(centre), 4),
            "review_band": chosen["width"],
        },
        "chosen": chosen,
        "guarantee": {
            "n": n,
            "decided": chosen["decided"],
            "errors": chosen["errors"],
            "error_rate": chosen["error_rate"],
            "cp_upper95": chosen["cp_upper95"],
            "alpha": float(policy["alpha"]),
        },
        "curve": rows,
    }


def describe(abstention: Dict[str, Any] | None) -> str | None:
    """One line for the report/UI: what the band costs and what it buys."""
    if not abstention or "chosen" not in abstention:
        return None
    ch = abstention["chosen"]
    if not ch.get("decided"):
        return None
    parts = [f"review band {ch['real_threshold']:.2f}-{ch['fake_threshold']:.2f}: "
             f"decides {ch['coverage']:.0%} of the calibration set"]
    if ch.get("error_rate") is not None:
        parts.append(f"{ch['error_rate']:.0%} error among decided clips")
    if ch.get("cp_upper95") is not None:
        parts.append(f"95% upper bound {ch['cp_upper95']:.0%}")
    return "; ".join(parts)
