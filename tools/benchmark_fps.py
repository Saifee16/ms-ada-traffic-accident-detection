"""Measure video read, resize, and YOLO throughput."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def bench_raw_read(source: str, n: int) -> float:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    start = time.perf_counter()
    count = 0
    while count < n:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    elapsed = time.perf_counter() - start
    cap.release()
    return count / elapsed if elapsed > 0 else 0.0


def bench_resize(source: str, n: int, resize_w: int = 640) -> float:
    cap = cv2.VideoCapture(source)
    start = time.perf_counter()
    count = 0
    while count < n:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width != resize_w:
            scale = resize_w / width
            frame = cv2.resize(frame, (resize_w, int(height * scale)))
        count += 1
    elapsed = time.perf_counter() - start
    cap.release()
    return count / elapsed if elapsed > 0 else 0.0


def bench_yolo(source: str, n: int, resize_w: int = 640) -> float:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [SKIP] ultralytics is not installed")
        return 0.0

    model = YOLO("yolo11n.pt")
    dummy = np.zeros((resize_w, resize_w, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, verbose=False)

    cap = cv2.VideoCapture(source)
    start = time.perf_counter()
    count = 0
    while count < n:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width != resize_w:
            scale = resize_w / width
            frame = cv2.resize(frame, (resize_w, int(height * scale)))
        model(frame, verbose=False, classes=[0, 1, 2, 3, 5, 7])
        count += 1
    elapsed = time.perf_counter() - start
    cap.release()
    return count / elapsed if elapsed > 0 else 0.0


def main(source: str, n: int, resize_w: int, frame_skip: int) -> None:
    path = Path(source)
    if not path.exists():
        print(f"[ERROR] Video not found: {source}")
        return

    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    processed_target = src_fps / max(frame_skip, 1)

    print(f"Source: {source}")
    print(f"Resolution: {src_w}x{src_h} | Source FPS: {src_fps:.1f} | Frames: {total_frames}")
    print(f"Target processed FPS: {processed_target:.1f}")
    print(f"Raw read FPS: {bench_raw_read(source, n):.1f}")
    print(f"Read + resize FPS: {bench_resize(source, n, resize_w):.1f}")

    yolo_fps = bench_yolo(source, n, resize_w)
    if yolo_fps > 0:
        headroom = yolo_fps / processed_target if processed_target else 0.0
        print(f"YOLO throughput FPS: {yolo_fps:.1f}")
        print(f"Headroom: {headroom:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to a local traffic video")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()
    main(args.source, args.frames, args.resize_width, args.frame_skip)
