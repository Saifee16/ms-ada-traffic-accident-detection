# Accident Detection Algorithm Overview

## Problem Fixed

The previous accident logic moved between two bad modes:

- Loose thresholds produced false positives from close traffic, overtaking, parked vehicles, and visual occlusion.
- Later strict thresholds suppressed valid accidents because real collisions often have noisy boxes and short contact windows.

The new detector uses a pair-level event state machine and explainable multi-signal scoring instead of a single threshold.

## Event State Pipeline

Each vehicle pair has independent state:

- `NORMAL`: no meaningful interaction.
- `SUSPECT`: proximity plus early conflict evidence.
- `CANDIDATE`: enough contact/near-overlap and motion conflict to begin confirmation.
- `CONFIRMED`: persistent evidence plus post-impact validation.
- `CLOSED`: event already emitted; prevents duplicate alerts for the same pair.

Event IDs use `ACC-{frame}-{track_a}-{track_b}-{counter}`.

## Candidate Generation

A pair becomes suspicious only when multiple preliminary conditions appear:

- Dynamic proximity based on configured pixels and bbox size.
- IoU or near-overlap edge gap.
- Relative motion.
- Predicted trajectory conflict.
- Velocity disruption or optical-flow anomaly.

Parked pairs and same-direction low-disruption traffic are suppressed early.

## Multi-Signal Fusion

The detector computes normalized scores:

- `iou_score`
- `proximity_score`
- `velocity_drop_score`
- `trajectory_conflict_score`
- `optical_flow_anomaly_score`
- `post_impact_stagnation_score`
- `direction_change_score`

Severity is a weighted score configured in `configs/default.yaml`. Every confirmed event stores signal scores and recent evidence in memory and writes debug JSONL when enabled.

## Predictive Trajectory

Recent centroid history is fit with a constant-velocity model. The detector projects both vehicles forward for `trajectory_prediction_horizon` frames and scores:

- whether future distance falls below the bbox-scale conflict radius;
- whether the pair is converging;
- whether headings conflict.

This catches collisions before or at bbox contact while avoiding normal side-by-side traffic.

## Temporal Confirmation

A single frame cannot confirm an accident. Candidate evidence must persist over `confirmation_seconds`, with `min_confirming_frames` computed from processed FPS unless configured. The default is high precision:

- `candidate_threshold: 0.42`
- `confirmed_threshold: 0.62`
- `confirmation_seconds: 1.2`
- `min_signals_required: 3`
- `cooldown_seconds: 30`

## Post-Impact Validation

High-precision mode requires post-impact validation after candidate impact:

- abrupt deceleration;
- stalled/stopped vehicle;
- sudden direction change;
- local optical-flow anomaly.

Spatial relation is recorded for debugging, but it is not enough to confirm an accident by itself. Confirmation requires a real disruption signal such as speed drop, stall, direction change, or local optical-flow anomaly.

## False Positive Suppression

The detector suppresses:

- normal overtaking and same-direction close traffic;
- lane changes without speed disruption;
- visual overlap from occlusion without motion anomaly;
- parked/static vehicles;
- bbox jitter and near-miss grazing;
- camera shake by subtracting global optical-flow baseline.

## Alert and Evidence Flow

When an accident is confirmed:

1. The pipeline finalizes plate/fallback identifiers for both tracks.
2. Evidence is saved under `output/evidence/{event_id}/`.
3. `detections.csv` and `accidents.csv` are updated.
4. WhatsApp and SMTP alerts are queued with event ID, identifiers, timestamp, camera, severity, snapshot path, and clip path.
5. Alerts deduplicate by `event_id` and retry with backoff.

## Limitations

- Depth ambiguity remains hard with one monocular camera; the detector relies on post-impact disruption to reject visual crossing.
- Pixel speed is not real-world speed until the camera is calibrated.
- Local WhatsApp media paths are not public URLs. Text alerts include paths; image upload needs a real media hosting integration.
- A short bounded smoke run verifies wiring, but final threshold tuning still needs labelled accident/non-accident clips from the deployment camera angle.
