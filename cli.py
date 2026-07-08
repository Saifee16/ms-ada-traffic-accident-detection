"""Command-line interface for the traffic surveillance system."""
from __future__ import annotations

import json
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


@cli.command()
@click.option("--source", required=False, help="Path to MP4 video file.")
@click.option("--source1", default=None, help="Primary camera MP4 for dual-camera mode.")
@click.option("--source2", default=None, help="Secondary camera MP4 for dual-camera mode.")
@click.option("--dual", is_flag=True, default=False, help="Enable dual-camera mode.")
@click.option("--output", default=None, help="Path for processed output MP4.")
@click.option("--camera-id", default="CAM01", show_default=True, help="Camera identifier.")
@click.option("--alerts", "alerts_mode", type=click.Choice(["on", "off", "mock"]), default="on", show_default=True)
@click.option("--no-alerts", is_flag=True, default=False, help="Compatibility alias for --alerts off.")
@click.option("--display/--no-display", default=True, show_default=True, help="Show live preview window.")
@click.option("--debug-events", is_flag=True, default=False, help="Write accident reasoning to debug_events.jsonl.")
@click.option("--debug-fusion", is_flag=True, default=False, help="Compatibility alias for --debug-events.")
@click.option("--max-frames", default=None, type=int, help="Optional smoke-test limit for processed frames.")
@click.pass_context
def run(ctx, source, source1, source2, dual, output, camera_id, alerts_mode, no_alerts, display, debug_events, debug_fusion, max_frames):
    """Process video file(s) through detection, tracking, ALPR, accident detection, and alerts."""
    from pipelines.video_pipeline import VideoPipeline
    from utils.config import Config
    from utils.logger import setup_logging
    import threading

    cfg = Config.load(ctx.obj["config"])
    if no_alerts:
        alerts_mode = "off"
    if alerts_mode == "off":
        cfg._data["alerts"]["enabled"] = False
    elif alerts_mode == "mock":
        cfg._data["alerts"]["enabled"] = True
        cfg._data["alerts"]["mock_mode"] = True
    if debug_events or debug_fusion:
        cfg._data["accident"]["debug_events"] = True
        cfg._data["output"]["save_debug_jsonl"] = True
        cfg._data["accident"]["debug_mode"] = True

    setup_logging(
        level="DEBUG" if (debug_events or debug_fusion) else (cfg.get("system", "log_level") or "INFO"),
        json_output=cfg.get("system", "log_json") or False,
    )

    if display:
        console.print("[cyan]Display enabled. Press q in the video window to stop, or Ctrl+C in this terminal.[/cyan]")
    else:
        console.print("[yellow]Display disabled. Use --display to show the processed video window.[/yellow]")

    if dual:
        if not source1 or not source2:
            console.print("[red]Error: --source1 and --source2 are required for --dual.[/red]")
            sys.exit(1)
        out1 = output or "output/cam1_processed.mp4"
        out2 = str(Path(out1).with_stem(Path(out1).stem + "_cam2"))
        pipes = [
            VideoPipeline(source1, cfg, camera_id="CAM01", output_path=out1, display=display, max_frames=max_frames),
            VideoPipeline(source2, cfg, camera_id="CAM02", output_path=out2, display=display, max_frames=max_frames),
        ]
        threads = [threading.Thread(target=p.run, daemon=False) for p in pipes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        if not source:
            console.print("[red]Error: --source is required.[/red]")
            sys.exit(1)
        out = output or "output/processed.mp4"
        VideoPipeline(source, cfg, camera_id=camera_id, output_path=out, display=display, max_frames=max_frames).run()

    console.print("[bold green]Processing complete.[/bold green]")


@cli.command("validate-config")
@click.pass_context
def validate_config(ctx):
    """Validate required configuration keys and basic numeric thresholds."""
    from utils.config import Config

    cfg = Config.load(ctx.obj["config"])
    errors = cfg.validate()
    if errors:
        for err in errors:
            console.print(f"[red]{err}[/red]")
        sys.exit(1)
    console.print("[green]Config validation passed.[/green]")


@cli.command()
@click.option("--source", required=True, help="Video file path for benchmarking.")
@click.option("--frames", default=300, show_default=True, help="Number of frames to process.")
@click.pass_context
def benchmark(ctx, source, frames):
    """Measure YOLO inference FPS and latency."""
    import time
    import cv2
    from detectors.yolo_wrapper import YOLODetector
    from utils.config import Config

    cfg = Config.load(ctx.obj["config"])
    det = YOLODetector(
        model_path=cfg.get("detector", "model") or "yolo11n.pt",
        conf_threshold=cfg.get("detector", "conf_threshold") or cfg.get("detector", "confidence_threshold") or 0.4,
        iou_threshold=cfg.get("detector", "iou_threshold") or 0.5,
        device_preference=cfg.get("system", "device") or "auto",
    )
    cap = cv2.VideoCapture(source)
    latencies = []
    count = 0
    t_total = time.perf_counter()
    while count < frames:
        ok, frame = cap.read()
        if not ok:
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
    table.add_row("Avg FPS", f"{count / elapsed:.1f}" if elapsed else "0.0")
    table.add_row("Avg inference (ms)", f"{sum(latencies) / len(latencies):.1f}" if latencies else "n/a")
    if latencies:
        table.add_row("Min / Max (ms)", f"{min(latencies):.1f} / {max(latencies):.1f}")
    console.print(table)


@cli.command("tune-thresholds")
@click.option("--events", required=True, help="Ground-truth labels JSON file.")
@click.option("--predictions", required=True, help="Predicted debug_events.jsonl file.")
def tune_thresholds(events, predictions):
    """Summarize event-label overlap to guide threshold tuning."""
    labels_path = Path(events)
    preds_path = Path(predictions)
    if not labels_path.exists() or not preds_path.exists():
        console.print("[red]Both --events and --predictions must exist.[/red]")
        sys.exit(1)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(labels, dict):
        label_frames = {int(item["frame_number"]) for item in labels.get("events", [])}
    else:
        label_frames = {int(item["frame_number"]) for item in labels}
    pred_frames = set()
    severities = []
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "confirmed":
            pred_frames.add(int(rec.get("frame_number", 0)))
            severities.append(float(rec.get("severity", 0.0)))
    tolerance = 15
    matches = sum(1 for pf in pred_frames if any(abs(pf - lf) <= tolerance for lf in label_frames))
    precision = matches / len(pred_frames) if pred_frames else 0.0
    recall = matches / len(label_frames) if label_frames else 0.0
    console.print(f"Predictions: {len(pred_frames)} | Labels: {len(label_frames)} | Matches(+/-{tolerance}f): {matches}")
    console.print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
    if severities:
        console.print(f"Severity range: {min(severities):.3f} - {max(severities):.3f}")


@cli.command("analyze-debug")
@click.option("--debug", "debug_path", required=True, help="Path to output/debug_events.jsonl.")
@click.option("--accidents", "accidents_path", default="output/accidents.csv", show_default=True)
def analyze_debug(debug_path, accidents_path):
    """Summarize accident debug JSONL for false-positive review."""
    import csv
    from collections import Counter, defaultdict

    path = Path(debug_path)
    if not path.exists():
        console.print(f"[red]Debug file not found: {path}[/red]")
        sys.exit(1)

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    type_counts = Counter(rec.get("type", "unknown") for rec in records)
    state_counts = Counter(rec.get("state", "unknown") for rec in records)
    suppressed = Counter(rec.get("suppressed_reason") for rec in records if rec.get("suppressed_reason"))
    pair_counts = Counter(tuple(rec.get("pair", [])) for rec in records if rec.get("pair"))
    active_signals = Counter(int(rec.get("active_signal_count", 0) or 0) for rec in records)
    wait_reasons = Counter(rec.get("wait_reason") for rec in records if rec.get("wait_reason"))
    severity_by_state = defaultdict(list)
    no_hard_impact = 0
    same_direction = 0
    static_related = 0

    for rec in records:
        state = rec.get("state", rec.get("type", "unknown"))
        severity_by_state[state].append(float(rec.get("severity", 0.0) or 0.0))
        reasons = set(rec.get("post_impact_reasons") or [])
        hard = {
            "post_impact_abrupt_deceleration",
            "post_impact_stalled_or_stopped",
            "post_impact_direction_change",
            "prediction_error_spike_after_contact",
        }
        if rec.get("type") in {"candidate_wait", "confirmed"} and not (reasons & hard):
            no_hard_impact += 1
        sr = rec.get("suppressed_reason", "")
        if "same_direction" in sr:
            same_direction += 1
        if "static" in sr or "parked" in sr:
            static_related += 1

    confirmed_csv_rows = 0
    acc_path = Path(accidents_path)
    if acc_path.exists():
        with open(acc_path, newline="", encoding="utf-8") as fh:
            confirmed_csv_rows = max(0, sum(1 for _ in csv.DictReader(fh)))

    table = Table(title="Accident Debug Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Debug records", str(len(records)))
    table.add_row("SUSPECT records", str(state_counts.get("SUSPECT", 0)))
    table.add_row("CANDIDATE records", str(state_counts.get("CANDIDATE", 0)))
    table.add_row("CONFIRMED records", str(type_counts.get("confirmed", 0)))
    table.add_row("accidents.csv rows", str(confirmed_csv_rows))
    table.add_row("Candidates without hard impact", str(no_hard_impact))
    table.add_row("Same-direction suppressed", str(same_direction))
    table.add_row("Static/parked suppressed", str(static_related))
    console.print(table)

    def print_counter(title: str, counter: Counter, limit: int = 8):
        sub = Table(title=title)
        sub.add_column("Key")
        sub.add_column("Count")
        for key, count in counter.most_common(limit):
            sub.add_row(str(key), str(count))
        console.print(sub)

    print_counter("Record Types", type_counts)
    print_counter("Suppressed Reasons", suppressed)
    print_counter("Wait Reasons", wait_reasons)
    print_counter("Top Pair IDs", pair_counts)
    print_counter("Active Signal Distribution", active_signals)

    sev_table = Table(title="Average Severity by State")
    sev_table.add_column("State")
    sev_table.add_column("Average")
    for state, values in sorted(severity_by_state.items()):
        avg = sum(values) / len(values) if values else 0.0
        sev_table.add_row(str(state), f"{avg:.3f}")
    console.print(sev_table)


@cli.command("export-onnx")
@click.option("--model", default="yolo11n.pt", show_default=True)
@click.option("--output", default="models/yolo11n.onnx", show_default=True)
def export_onnx(model, output):
    """Export YOLO model to ONNX for future TensorRT/GPU deployment."""
    from detectors.yolo_wrapper import YOLODetector

    det = YOLODetector(model_path=model)
    path = det.export_onnx(output)
    console.print(f"[green]ONNX export saved to: {path}[/green]")


if __name__ == "__main__":
    cli()
