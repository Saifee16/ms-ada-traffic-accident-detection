# Methodology

## 1. Problem Statement

Urban traffic accidents, particularly hit-and-run incidents, represent a critical public safety challenge. Manual review of CCTV footage is reactive, labour-intensive, and unable to scale to city-wide deployments. This project develops an autonomous, AI-powered surveillance system that detects accidents in real-time, identifies involved vehicles via Automatic License Plate Recognition (ALPR), and dispatches immediate alerts with photographic evidence.

**Research Questions:**
1. Can a fused multi-signal approach reduce false-positive accident detection rates below 5% in dense traffic video?
2. Does per-frame ALPR retry-until-exit achieve plate recognition rates > 80% on Pakistani road footage?
3. Can a CPU-first deployment achieve > 10 FPS on commodity hardware while preserving accuracy?

---

## 2. Literature Review Pointers

| Topic | Key References |
|-------|----------------|
| Object detection | Redmon et al. (YOLO, 2016); Wang et al. (YOLOv9, 2024) |
| Multi-object tracking | Zhang et al. (ByteTrack, 2022); Du et al. (StrongSORT, 2023) |
| ALPR | Li et al. (2019); Silva & Jung (2022) — plate detection + OCR pipeline |
| Accident detection | Bačić et al. (2016) — IoU-based; Shah et al. (2018) — trajectory |
| Optical flow | Farnebäck (2003); Lucas & Kanade (1981) |
| Pakistani plate formats | NTRC vehicle registration standards |

---

## 3. System Design

The system follows a modular layered architecture:

```
[Video Source] → [Frame Producer] → [Frame Queue]
                                         ↓
[YOLO Detector] + [ByteTrack Tracker] + [ALPR Pipeline]
                                         ↓
                              [Accident Fusion Engine]
                                         ↓
                    [Alert Dispatcher] + [CSV/Media Storage]
                                         ↓
                              [Annotated Video Renderer]
```

### 3.1 Detection Module
YOLOv11n pretrained on COCO, fine-tuned on a traffic dataset containing Pakistani road conditions. Inference at 640×640 resolution, FP32 on CPU.

### 3.2 Tracking Module
ByteTrack operates in two association stages:
- Stage 1: high-confidence detections matched by IoU + Kalman filter prediction.
- Stage 2: low-confidence detections matched against unconfirmed tracks.
Track IDs persist while objects remain visible. Re-ID gap = 10 seconds (configurable).

### 3.3 ALPR Pipeline
1. Plate region detection (fine-tuned YOLO plate detector).
2. Image enhancement (2× upscale + Otsu binarization).
3. EasyOCR text extraction on enhanced region.
4. Pakistan-format regex normalization.
5. Multi-frame accumulation → best-confidence reading selected on vehicle exit.
6. Fallback: `VEHICLE-ID-XXXX` assigned if no plate detected.

### 3.4 Accident Fusion Engine
Five independent signals evaluated per vehicle pair per frame:

| Signal | Computation | Threshold |
|--------|-------------|-----------|
| IoU spike | Intersection-over-Union of bounding boxes | ≥ 0.15 |
| Trajectory intersection | 2D segment intersection test on recent paths | Any intersection |
| Sudden deceleration | Speed drop fraction over 3-frame rolling window | ≥ 50% |
| Optical flow anomaly | Dense Farneback flow magnitude Z-score vs. 3s baseline | ≥ 2.5σ |
| Proximity violation | Euclidean centroid distance | ≤ 60px |

**Confirmation**: ≥ 3 signals must persist for ≥ 2 seconds (configurable). N-frame suppression (5 frames) prevents single-frame spikes from confirming.

**Severity score** (0–1): weighted sum of normalized signal values.

### 3.5 Alert System
On confirmed accident:
1. Snapshot saved (JPEG, 90% quality).
2. Rolling 5-second clip saved (buffered pre-event frames + post-event).
3. WhatsApp alert dispatched (Meta Cloud Business API v19.0) in background thread.
4. SMTP email dispatched with snapshot attachment.
5. CSV row appended with full metadata.

---

## 4. Algorithms and Hyperparameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| YOLO model | yolo11n.pt | Nano variant: best FPS/accuracy for CPU |
| Detection confidence | 0.35 | Lower threshold catches partial occlusions |
| Track buffer frames | 30 | ~1s at 30fps — handles brief occlusions |
| Confirmation window | 2s | Balances speed vs. false-positive rate |
| Optical flow window | 3s | Sufficient baseline for anomaly Z-score |
| ALPR min confidence | 0.45 | Tuned against Pakistan plate dataset |
| Frame skip | 2 | Process every 2nd frame → ~2× FPS boost |

---

## 5. Validation & Evaluation Plan

### 5.1 Detection Metrics
- **Precision / Recall / F1** on COCO vehicle classes using test split.
- **mAP@0.5** and **mAP@0.5:0.95** for plate detection.

### 5.2 Tracking Metrics
- **HOTA, MOTA, IDF1** on multi-object tracking benchmark sequences.
- ID switch rate across simulated occlusion sequences.

### 5.3 Accident Detection Metrics
- **True Positive Rate (TPR)** on labelled accident clips.
- **False Positive Rate (FPR)** on normal traffic clips (target < 5%).
- **Confirmation latency** (frames from collision onset to alert trigger).

### 5.4 ALPR Metrics
- **Plate detection rate**: fraction of vehicles with at least one plate detected.
- **OCR accuracy**: character-level accuracy vs. ground truth plates.
- **End-to-end plate recognition rate**: correct plate assigned by vehicle exit.

### 5.5 Performance Metrics
- **FPS** on CPU (i7-12th gen reference) at 640px.
- **Alert dispatch latency** (confirmation → WhatsApp send).
- **Memory footprint** (RAM, no GPU).

### 5.6 Thresholds
| Metric | Target |
|--------|--------|
| Accident FPR | < 5% |
| Plate recognition | > 80% |
| CPU FPS | > 10 |
| Alert latency | < 3s |

---

## 6. Ethical Considerations and Privacy Implications

- **Data minimization**: license plates stored only when accident flag is set; routine tracking data is ephemeral (in-memory).
- **Consent**: system designed for public roads where reasonable expectation of privacy is limited; deployment must comply with PEMRA and PTA guidelines.
- **Bias**: OCR models may perform unequally across different plate fonts/conditions; accuracy must be validated on demographically representative Pakistani datasets.
- **Security**: API tokens stored exclusively in environment variables; no credentials committed to version control.
- **Retention**: clip and snapshot evidence must comply with organizational data-retention policies.
- **Misuse**: system must not be deployed for surveillance of individuals beyond accident detection; access control and audit logging required.

---

## 7. Results/Discussion Template

*(To be filled after experiments)*

### 7.1 Detection Performance
Table comparing baseline COCO pretrained vs. fine-tuned model precision/recall by vehicle class.

### 7.2 Tracking Performance
HOTA/MOTA/IDF1 table. ID switch analysis per occlusion duration bin.

### 7.3 Accident Detection
Confusion matrix at default threshold. ROC curve varying confirmation window from 0.5s–3s. FPR vs. TPR tradeoff discussion.

### 7.4 ALPR Performance
Recognition rate by lighting condition (day/night/rain). Failure analysis of unrecognized plates.

### 7.5 System Performance
FPS vs. resolution table. Memory profile. Alert latency distribution.

---

## 8. Conclusion & Future Work

**Conclusion**: The system demonstrates feasibility of real-time accident detection on consumer hardware using a fused multi-signal approach that substantially reduces false positives versus single-signal baselines.

**Future Work**:
1. GPU deployment with TensorRT-optimized YOLO engine (target: 60+ FPS).
2. PTZ camera support with auto-calibration for `pixels_per_meter`.
3. Re-ID appearance model (OSNet) integration for cross-camera vehicle tracking.
4. City-scale deployment with central dashboard and database backend.
5. Active learning loop: human-reviewed false positives fed back to fine-tune accident classifier.
6. Night-mode enhancement using low-light image restoration preprocessing.
