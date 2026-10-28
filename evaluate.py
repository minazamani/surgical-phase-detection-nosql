import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
)

from torch.utils.data import DataLoader

from dataset import SurgicalPhaseDataset
from mstcn import MultiStageModel


# ==========================================================
# CONFIGURATION
# ==========================================================

PROCESSED_DIR = Path(
    "/home/mina/PycharmProjects/phasedetection/processed"
)

CHECKPOINT_PATH = Path(
    "training_output/best_model.pt"
)

OUTPUT_DIR = Path(
    "evaluation_output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

IGNORE_INDEX = -100

NUM_STAGES = 4
NUM_LAYERS = 10
NUM_F_MAPS = 64
FEATURE_DIM = 2048


print("Device:", DEVICE)


# ==========================================================
# LOAD LABEL MAP
# ==========================================================

with open(
    PROCESSED_DIR / "label_maps.json",
    "r"
) as f:

    label_maps = json.load(f)


phase_to_id = label_maps["phase_to_id"]

valid_classes = {
    name: idx
    for name, idx in phase_to_id.items()
    if idx >= 0
}

NUM_CLASSES = len(valid_classes)


# id -> phase name
id_to_phase = {
    idx: name
    for name, idx in valid_classes.items()
}


class_names = [
    id_to_phase[i]
    for i in range(NUM_CLASSES)
]


print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, name)


# ==========================================================
# LOAD TEST DATASET
# ==========================================================

test_dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


print(
    "\nTest video fragments:",
    len(test_dataset)
)


# ==========================================================
# CREATE MODEL
# ==========================================================

model = MultiStageModel(
    num_stages=NUM_STAGES,
    num_layers=NUM_LAYERS,
    num_f_maps=NUM_F_MAPS,
    input_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
)

model = model.to(DEVICE)


# ==========================================================
# LOAD CHECKPOINT
# ==========================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    "\nLoaded checkpoint from epoch:",
    checkpoint.get("epoch")
)

print(
    "Validation Macro-F1 of checkpoint:",
    checkpoint.get("val_macro_f1")
)


# ==========================================================
# EVALUATION
# ==========================================================

all_predictions = []
all_labels = []

video_results = []


with torch.no_grad():

    for batch in test_loader:

        features = batch[
            "features"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        video_id = batch[
            "video_id"
        ][0]

        surgery_id = batch[
            "surgery_id"
        ][0]

        # ----------------------------------------------
        # Valid annotation positions
        # ----------------------------------------------

        valid = (
            labels != IGNORE_INDEX
        )

        if valid.sum() == 0:

            print(
                f"Skipping {video_id}: "
                f"no valid labels"
            )

            continue

        # ----------------------------------------------
        # Mask for MS-TCN
        # ----------------------------------------------

        mask = (
            valid
            .float()
            .unsqueeze(1)
        )

        # ----------------------------------------------
        # Forward
        # ----------------------------------------------

        outputs = model(
            features,
            mask
        )

        # Use final MS-TCN stage
        final_output = outputs[-1]

        predictions = (
            final_output
            .argmax(dim=1)
        )

        # ----------------------------------------------
        # Remove ignored positions
        # ----------------------------------------------

        labels_valid = (
            labels[valid]
            .cpu()
            .numpy()
        )

        predictions_valid = (
            predictions[valid]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------
        # Global results
        # ----------------------------------------------

        all_labels.append(
            labels_valid
        )

        all_predictions.append(
            predictions_valid
        )

        # ----------------------------------------------
        # Per-video metrics
        # ----------------------------------------------

        video_accuracy = accuracy_score(
            labels_valid,
            predictions_valid
        )

        video_macro_f1 = f1_score(
            labels_valid,
            predictions_valid,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )

        video_results.append(
            {
                "video_id": video_id,
                "surgery_id": surgery_id,
                "num_valid_samples":
                    len(labels_valid),

                "accuracy":
                    video_accuracy,

                "macro_f1":
                    video_macro_f1,
            }
        )

        print(
            f"{video_id:20s} | "
            f"Accuracy: "
            f"{video_accuracy:.4f} | "
            f"Macro-F1: "
            f"{video_macro_f1:.4f}"
        )


# ==========================================================
# COMBINE ALL TEST VIDEOS
# ==========================================================

if len(all_labels) == 0:

    raise RuntimeError(
        "No test predictions were generated."
    )


all_labels = np.concatenate(
    all_labels
)

all_predictions = np.concatenate(
    all_predictions
)


# ==========================================================
# GLOBAL METRICS
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)


print("\n======================================")
print("TEST RESULTS")
print("======================================")

print(
    f"Accuracy:    {accuracy:.4f}"
)

print(
    f"Macro-F1:    {macro_f1:.4f}"
)

print(
    f"Weighted-F1: {weighted_f1:.4f}"
)


# ==========================================================
# PER-CLASS METRICS
# ==========================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
)


metrics_df = pd.DataFrame(
    {
        "phase_id":
            list(range(NUM_CLASSES)),

        "phase":
            class_names,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "support":
            support,
    }
)


print("\nPer-phase results:")
print(
    metrics_df.to_string(
        index=False
    )
)


metrics_df.to_csv(
    OUTPUT_DIR
    / "per_phase_metrics.csv",
    index=False
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0
)

print("\nClassification report:")
print(report)


with open(
    OUTPUT_DIR
    / "classification_report.txt",
    "w"
) as f:

    f.write(report)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)


cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)


# ==========================================================
# NORMALIZED CONFUSION MATRIX
# ==========================================================

cm_normalized = (
    cm.astype(float)
    / np.maximum(
        cm.sum(axis=1)[:, None],
        1
    )
)


# ==========================================================
# PLOT CONFUSION MATRIX
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

image = ax.imshow(
    cm_normalized
)

ax.set_xticks(
    np.arange(NUM_CLASSES)
)

ax.set_yticks(
    np.arange(NUM_CLASSES)
)

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    class_names
)

ax.set_xlabel(
    "Predicted phase"
)

ax.set_ylabel(
    "True phase"
)

ax.set_title(
    "Normalized Confusion Matrix"
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        ax.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            ha="center",
            va="center"
        )


fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "confusion_matrix.png",
    dpi=200
)

plt.close(fig)


# ==========================================================
# SAVE PER-VIDEO METRICS
# ==========================================================

video_results_df = pd.DataFrame(
    video_results
)


video_results_df.to_csv(
    OUTPUT_DIR
    / "per_video_metrics.csv",
    index=False
)


# ==========================================================
# SAVE GLOBAL SUMMARY
# ==========================================================

summary = {
    "checkpoint_epoch":
        checkpoint.get("epoch"),

    "checkpoint_validation_macro_f1":
        checkpoint.get("val_macro_f1"),

    "test_accuracy":
        float(accuracy),

    "test_macro_f1":
        float(macro_f1),

    "test_weighted_f1":
        float(weighted_f1),

    "num_test_fragments":
        len(video_results),

    "num_test_samples":
        int(len(all_labels)),
}


with open(
    OUTPUT_DIR
    / "summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR.resolve()
)