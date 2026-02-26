"""
notebooks/evaluation.py — Evaluation notebook (run with: jupyter nbconvert --to notebook --execute)

Computes:
  - Detection precision/recall/F1 per class
  - Accident detection confusion matrix
  - FPS histogram
  - ALPR accuracy breakdown
"""
# %%
import sys; sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve, auc,
)
from pathlib import Path

sns.set_theme(style="darkgrid")

# ── Load CSV output ──────────────────────────────────────────
# %%
CSV_PATH = "../output/detections.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows, {df['camera_id'].nunique()} camera(s)")
df.head()

# ── Vehicle class distribution ───────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(8, 4))
df["class"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Detection Count by Vehicle Class")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../output/class_distribution.png", dpi=150)
plt.show()

# ── Accident detection confusion matrix ──────────────────────
# %%
# Load ground-truth labels (must be prepared separately)
GT_PATH = "../datasets/labels/accident_gt.csv"
if Path(GT_PATH).exists():
    gt = pd.read_csv(GT_PATH)  # columns: frame_id, camera_id, accident (0/1)
    merged = df.merge(gt, on=["frame_id", "camera_id"], how="inner", suffixes=("_pred", "_gt"))
    y_true = merged["accident_gt"].values
    y_pred = merged["accident_flag"].values

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Accident"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Accident Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../output/confusion_matrix.png", dpi=150)
    plt.show()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1)
    print(f"Precision: {precision[1]:.3f}  Recall: {recall[1]:.3f}  F1: {f1[1]:.3f}")
else:
    print(f"Ground truth file not found at {GT_PATH}. Skipping confusion matrix.")

# ── Speed distribution ───────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(8, 4))
df[df["speed_px_per_s"] > 0]["speed_px_per_s"].hist(bins=50, ax=ax, color="coral", edgecolor="black")
ax.set_title("Vehicle Speed Distribution (px/s)")
ax.set_xlabel("Speed (px/s)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("../output/speed_distribution.png", dpi=150)
plt.show()

# ── ALPR fallback rate ───────────────────────────────────────
# %%
total = len(df.drop_duplicates("track_id"))
fallbacks = df["plate_text"].str.startswith("VEHICLE-ID").sum()
recognized = total - fallbacks
print(f"Plate recognition rate: {recognized/total*100:.1f}% ({recognized}/{total})")

labels = ["Recognized", "Fallback ID"]
sizes = [recognized, fallbacks]
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
ax.set_title("ALPR: Plate Recognition vs. Fallback")
plt.savefig("../output/alpr_pie.png", dpi=150)
plt.show()

print("Evaluation complete. Charts saved to output/")
