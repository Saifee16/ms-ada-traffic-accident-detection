"""Summarize accident events from the detection CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(csv_path: str = "output/detections.csv") -> None:
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}")
        return

    df = pd.read_csv(path)
    acc = df[df["accident_flag"] == 1].copy()

    print(f"Detection CSV: {path}")
    print(f"Total rows: {len(df)}")
    print(f"Accident rows: {len(acc)}")

    if acc.empty:
        print("No accident events recorded yet.")
        return

    print("\nBy camera")
    print(acc.groupby("camera_id").size().rename("accident_rows").to_string())

    print("\nUnique tracks involved")
    print(f"{acc['track_id'].nunique()} unique track IDs flagged")
    print(f"Track IDs: {sorted(acc['track_id'].unique().tolist())}")

    print("\nIoU at collision")
    iou = acc["IoU_at_collision"]
    print(f"mean={iou.mean():.3f} max={iou.max():.3f} min={iou.min():.3f}")

    print("\nMotion anomaly score")
    motion = acc["motion_anomaly_score"]
    print(f"mean={motion.mean():.3f} max={motion.max():.3f} min={motion.min():.3f}")

    print("\nPlate coverage")
    has_plate = acc["plate_text"].notna() & (acc["plate_text"] != "")
    print(f"{has_plate.sum()} / {len(acc)} rows have a plate reading")

    if "timestamp" in acc.columns:
        print("\nTimeline")
        acc_sorted = acc.sort_values("timestamp")
        print(f"First event: {acc_sorted['timestamp'].iloc[0]}")
        print(f"Last event: {acc_sorted['timestamp'].iloc[-1]}")

    snaps = acc["snapshot_path"].dropna()
    snaps = snaps[snaps != ""]
    clips = acc["clip_path"].dropna()
    clips = clips[clips != ""]
    print("\nEvidence")
    print(f"Snapshots saved: {len(snaps)}")
    print(f"Clips saved: {len(clips)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/detections.csv")
    args = parser.parse_args()
    main(args.csv)
