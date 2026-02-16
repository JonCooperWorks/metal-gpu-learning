#!/usr/bin/env python3
"""
Combine Lesson 10 kernel outputs into a single likely-fraud ranking.

Input:
  JSON report produced by:
    cargo run -p medicaid-provider-spending -- ... --output-json <path>

Output:
  - Aggregated JSON with ranked candidates
  - Flat CSV for quick filtering/sorting
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# Kernel 1 is a top-spender baseline and not directly used for fraud ranking.
FRAUD_KERNEL_WEIGHTS: Dict[int, float] = {
    2: 1.00,  # z-score anomaly
    3: 1.10,  # MAD anomaly
    4: 1.30,  # paid-per-claim MAD anomaly
    5: 1.20,  # month-over-month spike
    6: 1.00,  # drift sigma anomaly
    7: 1.15,  # rarity-weighted anomaly
    8: 1.25,  # distance outlier
}

TIER_THRESHOLDS: List[Tuple[float, str]] = [
    (80.0, "critical"),
    (60.0, "high"),
    (40.0, "medium"),
    (0.0, "low"),
]


@dataclass
class Candidate:
    group_key: str
    month: str
    paid: float
    claims: int
    beneficiaries: int
    votes: int = 0
    weighted_sum: float = 0.0
    kernels: List[int] = None  # type: ignore[assignment]
    reasons: List[str] = None  # type: ignore[assignment]
    kernel_scores: Dict[int, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kernels is None:
            self.kernels = []
        if self.reasons is None:
            self.reasons = []
        if self.kernel_scores is None:
            self.kernel_scores = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse lesson 10 kernel outputs into likely fraud ranks.")
    p.add_argument("--input", required=True, help="Path to lesson10 raw report JSON.")
    p.add_argument("--output-json", required=True, help="Path for aggregated fraud JSON.")
    p.add_argument("--output-csv", required=True, help="Path for aggregated fraud CSV.")
    p.add_argument("--top-n", type=int, default=100, help="How many rows to keep in final output.")
    p.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Require this many distinct kernel hits to keep a candidate.",
    )
    return p.parse_args()


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def tier_for_score(score: float) -> str:
    for threshold, tier in TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "low"


def main() -> int:
    args = parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    reports = raw.get("reports", [])

    candidates: Dict[Tuple[str, str], Candidate] = {}
    total_fraud_kernels = len(FRAUD_KERNEL_WEIGHTS)

    for kernel_report in reports:
        kernel_id = int(kernel_report.get("kernel_id", 0))
        weight = FRAUD_KERNEL_WEIGHTS.get(kernel_id)
        if weight is None:
            continue

        top_rows = kernel_report.get("top", [])
        if not top_rows:
            continue

        raw_scores = [float(r.get("score", 0.0)) for r in top_rows]
        norm_scores = normalize(raw_scores)
        n = len(top_rows)

        for rank, row in enumerate(top_rows, start=1):
            group_key = str(row.get("group_key", ""))
            month = str(row.get("month", ""))
            key = (group_key, month)

            score_norm = norm_scores[rank - 1]
            rank_norm = 1.0 if n == 1 else 1.0 - ((rank - 1) / (n - 1))
            signal = 0.6 * score_norm + 0.4 * rank_norm
            contribution = weight * signal

            c = candidates.get(key)
            if c is None:
                c = Candidate(
                    group_key=group_key,
                    month=month,
                    paid=float(row.get("paid", 0.0)),
                    claims=int(row.get("claims", 0)),
                    beneficiaries=int(row.get("beneficiaries", 0)),
                )
                candidates[key] = c

            c.paid = max(c.paid, float(row.get("paid", 0.0)))
            c.claims = max(c.claims, int(row.get("claims", 0)))
            c.beneficiaries = max(c.beneficiaries, int(row.get("beneficiaries", 0)))
            c.votes += 1
            c.weighted_sum += contribution
            c.kernels.append(kernel_id)
            c.reasons.append(str(row.get("reason", "")))
            c.kernel_scores[kernel_id] = float(row.get("score", 0.0))

    filtered = [c for c in candidates.values() if c.votes >= args.min_votes]
    if not filtered:
        out = {
            "source_report": args.input,
            "total_candidates": 0,
            "message": "No candidates met min-votes threshold.",
            "candidates": [],
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "fraud_score",
                    "tier",
                    "votes",
                    "group_key",
                    "month",
                    "paid",
                    "claims",
                    "beneficiaries",
                    "kernels",
                    "kernel_raw_scores",
                ]
            )
        return 0

    calibrated = []
    for c in filtered:
        vote_bonus = 0.7 + 0.3 * (c.votes / total_fraud_kernels)
        calibrated.append(c.weighted_sum * vote_bonus)

    norm = normalize(calibrated)

    ranked = []
    for c, value_norm in sorted(
        zip(filtered, norm),
        key=lambda item: (item[1], item[0].votes, item[0].weighted_sum),
        reverse=True,
    ):
        fraud_score = round(value_norm * 100.0, 2)
        ranked.append(
            {
                "fraud_score": fraud_score,
                "fraud_tier": tier_for_score(fraud_score),
                "votes": c.votes,
                "group_key": c.group_key,
                "month": c.month,
                "paid": round(c.paid, 2),
                "claims": c.claims,
                "beneficiaries": c.beneficiaries,
                "kernels_triggered": sorted(set(c.kernels)),
                "kernel_raw_scores": {str(k): v for k, v in sorted(c.kernel_scores.items())},
                "reasons": c.reasons,
            }
        )

    ranked = ranked[: args.top_n]

    payload = {
        "source_report": args.input,
        "heuristic": {
            "kernel_weights": FRAUD_KERNEL_WEIGHTS,
            "min_votes": args.min_votes,
            "scoring": "kernel_signal=0.6*normalized_score+0.4*normalized_rank; weighted_sum adjusted by vote bonus",
        },
        "total_candidates": len(ranked),
        "candidates": ranked,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "fraud_score",
                "tier",
                "votes",
                "group_key",
                "month",
                "paid",
                "claims",
                "beneficiaries",
                "kernels",
                "kernel_raw_scores",
            ]
        )
        for idx, row in enumerate(ranked, start=1):
            writer.writerow(
                [
                    idx,
                    row["fraud_score"],
                    row["fraud_tier"],
                    row["votes"],
                    row["group_key"],
                    row["month"],
                    row["paid"],
                    row["claims"],
                    row["beneficiaries"],
                    ",".join(str(k) for k in row["kernels_triggered"]),
                    json.dumps(row["kernel_raw_scores"], separators=(",", ":")),
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
