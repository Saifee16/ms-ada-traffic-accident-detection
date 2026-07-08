"""Threshold sweep helper for CollisionDetector outputs.

Usage:
    python scripts/tune_collision_thresholds.py --labels data/labels.csv --detected detected_events.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class GTEvent:
    video: str
    frame_id: int
    track_a: int
    track_b: int
    is_collision: bool


def load_labels(csv_path: str) -> List[GTEvent]:
    events = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            events.append(
                GTEvent(
                    video=row["video_path"],
                    frame_id=int(row["frame_id"]),
                    track_a=int(row["track_a"]),
                    track_b=int(row["track_b"]),
                    is_collision=bool(int(row["is_collision"])),
                )
            )
    return events


def load_detected_events(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("events", data) if isinstance(data, dict) else data


def _compute_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return precision, recall, f1


def sweep_thresholds(detected_events: List[Dict], gt_events: List[GTEvent], frame_tolerance: int = 30) -> Dict:
    gt_positives = [event for event in gt_events if event.is_collision]
    tp, fp = 0, 0
    matched_gt = set()

    for det in detected_events:
        matched = False
        for idx, gt in enumerate(gt_positives):
            if idx in matched_gt:
                continue
            same_video = gt.video == det.get("video", "")
            close_frame = abs(gt.frame_id - int(det["frame_id"])) <= frame_tolerance
            same_pair = {det.get("track_a", -1), det.get("track_b", -1)} == {gt.track_a, gt.track_b}
            if same_video and close_frame and (same_pair or frame_tolerance >= 30):
                tp += 1
                matched_gt.add(idx)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(gt_positives) - tp
    precision, recall, f1 = _compute_f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CollisionDetector threshold candidates")
    parser.add_argument("--labels", required=True, help="Ground-truth labels CSV")
    parser.add_argument("--detected", required=True, help="Detected events JSON")
    parser.add_argument("--frame-tolerance", type=int, default=30)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--output", default="tuning_report.json")
    args = parser.parse_args()

    for required_path in [args.labels, args.detected]:
        if not Path(required_path).exists():
            print(f"ERROR: file not found: {required_path}", file=sys.stderr)
            sys.exit(1)

    gt = load_labels(args.labels)
    detected_events = load_detected_events(args.detected)
    metrics = sweep_thresholds(detected_events, gt, frame_tolerance=args.frame_tolerance)

    param_grid = {
        "min_signals_required": [2, 3, 4],
        "min_rel_delta_v": [0.08, 0.15, 0.25],
        "moving_threshold": [0.01, 0.02, 0.04],
    }
    report = {
        "metrics": metrics,
        "param_grid_size": len(list(itertools.product(*param_grid.values()))),
        "param_grid": param_grid,
        "min_precision_constraint": args.min_precision,
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
