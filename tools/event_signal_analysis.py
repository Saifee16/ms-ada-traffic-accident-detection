"""Create a per-track accident signal summary from detections CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(csv_path: str = "output/detections.csv", out_path: str = "tools/event_signal_analysis.csv") -> None:
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}")
        return

    df = pd.read_csv(path)
    acc = df[df["accident_flag"] == 1].copy()
    if acc.empty:
        print("No accident rows found in CSV.")
        return

    acc["timestamp"] = pd.to_datetime(acc["timestamp"], utc=True, errors="coerce")
    acc = acc.sort_values("timestamp")

    rows = []
    for (camera, track), group in acc.groupby(["camera_id", "track_id"]):
        iou_peak = group["IoU_at_collision"].max()
        motion_max = group["motion_anomaly_score"].max()
        speed = group["speed_px_per_s"].iloc[0] if "speed_px_per_s" in group.columns else None
        plate_values = group["plate_text"].dropna()
        snap_values = group["snapshot_path"].dropna()

        inferred = []
        if iou_peak >= 0.20:
            inferred.append("iou")
        if motion_max >= 0.10:
            inferred.append("motion_anomaly")

        rows.append(
            {
                "timestamp": group["timestamp"].min().isoformat() if not group["timestamp"].isna().all() else "",
                "camera_id": camera,
                "track_id": track,
                "plate_text": plate_values.iloc[0] if len(plate_values) else "",
                "iou_peak": round(iou_peak, 4),
                "motion_score_max": round(motion_max, 4),
                "speed_px_per_s": round(speed, 2) if speed is not None else "",
                "signals_inferred": ", ".join(inferred) if inferred else "",
                "snapshot_path": snap_values.iloc[0] if len(snap_values) else "",
            }
        )

    out = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Events analysed: {len(rows)} track-level records")
    print(f"Written to {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/detections.csv")
    parser.add_argument("--out", default="tools/event_signal_analysis.csv")
    args = parser.parse_args()
    main(args.csv, args.out)
