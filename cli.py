"""cli.py — Command-line interface for Traffic Surveillance System.

Usage examples:
  python cli.py run --source video.mp4 --output output/processed.mp4
  python cli.py run --source video.mp4 --camera-id cam_01 --no-alerts
  python cli.py run --source1 cam1.mp4 --source2 cam2.mp4 --dual
  python cli.py export-onnx --model yolo11n.pt
  python cli.py benchmark --source sample.mp4 --frames 300
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.option("--config", default="configs/default.yaml", show_default=True, help="Path to YAML config.")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


# ──────────────────────────────────────────────────────────────
@cli.command()
@click.option("--source", required=False, help="Path to MP4 video file (single camera).")
@click.option("--source1", default=None, help="Primary camera MP4 (dual-camera mode).")
@click.option("--source2", default=None, help="Secondary camera MP4 (dual-camera mode).")
@click.option("--dual", is_flag=True, default=False, help="Enable dual-camera mode.")
@click.option("--output", default=None, help="Path for processed output MP4.")
@click.option("--camera-id", default="cam_01", show_default=True, help="Camera identifier tag.")
@click.option("--alerts/--no-alerts", default=True, help="Enable/disable alert dispatch.")
@click.option("--display/--no-display", default=False, help="Show live preview window.")
@click.option("--debug-fusion", is_flag=True, default=False, help="Log per-frame fusion gate decisions (verbose).")
@click.pass_context
def run(ctx, source, source1, source2, dual, output, camera_id, alerts, display, debug_fusion):
    """Process video file(s) through the full surveillance pipeline."""
    from utils.config import Config
    from utils.logger import setup_logging

    cfg = Config.load(ctx.obj["config"])
    log_level = "DEBUG" if debug_fusion else (cfg.get("system", "log_level") or "INFO")
    setup_logging(level=log_level, json_output=False)  # never JSON in debug mode

    # Override alerts flag from CLI
    if not alerts:
        cfg._data["alerts"]["enabled"] = False

    from pipelines.video_pipeline import VideoPipeline
    import threading

    if dual and source1 and source2:
        console.print("[bold cyan]Dual-camera mode[/bold cyan]")
        out1 = output or "output/cam1_processed.mp4"
        out2 = str(Path(out1).with_stem(Path(out1).stem + "_cam2"))
        pipes = [
            VideoPipeline(source1, cfg, camera_id="cam_01", output_path=out1, display=display),
            VideoPipeline(source2, cfg, camera_id="cam_02", output_path=out2, display=display),
        ]
        threads = [threading.Thread(target=p.run, daemon=False) for p in pipes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        if not source:
            console.print("[red]Error: --source is required for single-camera mode.[/red]")
            sys.exit(1)
        out = output or "output/processed.mp4"
        pipe = VideoPipeline(source, cfg, camera_id=camera_id, output_path=out, display=display)
        pipe._debug_fusion = debug_fusion
        pipe.run()

    console.print("[bold green]Processing complete.[/bold green]")


# ──────────────────────────────────────────────────────────────
@cli.command("export-onnx")
@click.option("--model", default="yolo11n.pt", show_default=True)
@click.option("--output", default="models/yolo11n.onnx", show_default=True)
@click.pass_context
def export_onnx(ctx, model, output):
    """Export YOLO model to ONNX for future GPU/TensorRT deployment."""
    from detectors.yolo_wrapper import YOLODetector
    det = YOLODetector(model_path=model)
    path = det.export_onnx(output)
    console.print(f"[green]ONNX export saved to: {path}[/green]")


# ──────────────────────────────────────────────────────────────
@cli.command()
@click.option("--source", required=True, help="Video file path for benchmarking.")
@click.option("--frames", default=300, show_default=True, help="Number of frames to process.")
@click.pass_context
def benchmark(ctx, source, frames):
    """Measure inference FPS and latency."""
    import time
    import cv2
    from utils.config import Config
    from detectors.yolo_wrapper import YOLODetector

    cfg = Config.load(ctx.obj["config"])
    det = YOLODetector(
        model_path=cfg.get("detector", "model") or "yolo11n.pt",
        conf_threshold=cfg.get("detector", "conf_threshold") or 0.35,
    )
    cap = cv2.VideoCapture(source)
    latencies = []
    count = 0
    t_total = time.perf_counter()
    while count < frames:
        ret, frame = cap.read()
        if not ret:
            break
        result = det.infer(frame, frame_id=count)
        latencies.append(result.inference_ms)
        count += 1
    cap.release()
    elapsed = time.perf_counter() - t_total

    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Frames processed", str(count))
    table.add_row("Total time (s)", f"{elapsed:.2f}")
    table.add_row("Avg FPS", f"{count/elapsed:.1f}")
    table.add_row("Avg inference (ms)", f"{sum(latencies)/len(latencies):.1f}")
    table.add_row("Min / Max (ms)", f"{min(latencies):.1f} / {max(latencies):.1f}")
    console.print(table)


if __name__ == "__main__":
    cli()