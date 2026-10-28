import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SurgicalPhaseDataset
from mstcn import MultiStageModel


# ==========================================================
# CONFIG
# ==========================================================

PROCESSED_DIR = Path(
    "/home/mina/PycharmProjects/phasedetection/processed"
)

CHECKPOINT_PATH = Path(
    "training_output/best_model.pt"
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

id_to_phase = {
    idx: name
    for name, idx in valid_classes.items()
}


# ==========================================================
# DATASET
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


# ==========================================================
# MODEL
# ==========================================================

model = MultiStageModel(
    num_stages=NUM_STAGES,
    num_layers=NUM_LAYERS,
    num_f_maps=NUM_F_MAPS,
    input_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ==========================================================
# SEGMENT EXTRACTION
# ==========================================================

def get_segments(sequence):
    """
    Convert frame/second-level labels into segments.

    Example:
        [1,1,1,2,2,3,3]

    becomes:
        labels = [1,2,3]
        starts = [0,3,5]
        ends   = [3,5,7]

    end indices are exclusive.
    """

    labels = []
    starts = []
    ends = []

    if len(sequence) == 0:
        return labels, starts, ends

    last_label = sequence[0]
    start = 0

    for i in range(1, len(sequence)):

        if sequence[i] != last_label:

            labels.append(
                int(last_label)
            )

            starts.append(
                start
            )

            ends.append(
                i
            )

            last_label = sequence[i]
            start = i

    labels.append(
        int(last_label)
    )

    starts.append(
        start
    )

    ends.append(
        len(sequence)
    )

    return (
        labels,
        starts,
        ends
    )


# ==========================================================
# LEVENSHTEIN EDIT DISTANCE
# ==========================================================

def levenshtein_distance(a, b):

    m = len(a)
    n = len(b)

    dp = np.zeros(
        (m + 1, n + 1),
        dtype=np.int32
    )

    dp[:, 0] = np.arange(
        m + 1
    )

    dp[0, :] = np.arange(
        n + 1
    )

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            cost = (
                0
                if a[i - 1] == b[j - 1]
                else 1
            )

            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost
            )

    return dp[m, n]


def edit_score(prediction, ground_truth):

    pred_labels, _, _ = get_segments(
        prediction
    )

    gt_labels, _, _ = get_segments(
        ground_truth
    )

    if len(pred_labels) == 0:
        return 0.0

    if len(gt_labels) == 0:
        return 0.0

    distance = levenshtein_distance(
        pred_labels,
        gt_labels
    )

    max_len = max(
        len(pred_labels),
        len(gt_labels)
    )

    score = (
        1.0
        - distance / max_len
    )

    return 100.0 * score


# ==========================================================
# SEGMENT F1 @ OVERLAP
# ==========================================================

def segment_f1(
    prediction,
    ground_truth,
    overlap_threshold
):

    pred_labels, pred_starts, pred_ends = (
        get_segments(prediction)
    )

    gt_labels, gt_starts, gt_ends = (
        get_segments(ground_truth)
    )

    if len(pred_labels) == 0:

        return 0, 0, len(gt_labels)

    if len(gt_labels) == 0:

        return 0, len(pred_labels), 0

    hits = np.zeros(
        len(gt_labels)
    )

    tp = 0
    fp = 0

    for j in range(
        len(pred_labels)
    ):

        intersection = np.minimum(
            pred_ends[j],
            gt_ends
        ) - np.maximum(
            pred_starts[j],
            gt_starts
        )

        intersection = np.maximum(
            intersection,
            0
        )

        union = np.maximum(
            pred_ends[j],
            gt_ends
        ) - np.minimum(
            pred_starts[j],
            gt_starts
        )

        iou = (
            intersection
            / np.maximum(union, 1e-8)
        )

        # Segment must also have same phase label
        iou = iou * (
            np.array(gt_labels)
            == pred_labels[j]
        )

        idx = np.argmax(iou)

        if (
            iou[idx] >= overlap_threshold
            and hits[idx] == 0
        ):

            tp += 1
            hits[idx] = 1

        else:

            fp += 1

    fn = len(gt_labels) - hits.sum()

    return (
        int(tp),
        int(fp),
        int(fn)
    )


def f1_from_counts(
    tp,
    fp,
    fn
):

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    if precision + recall == 0:
        return 0.0

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
    )

    return 100.0 * f1

def remove_short_segments(sequence, min_duration):
    """
    Merge segments shorter than min_duration.

    Rules:
    - If left and right neighbors have the same label,
      merge the short segment into that label.
    - Otherwise merge into the longer neighboring segment.
    - At the beginning/end, merge into the only neighbor.
    """

    sequence = np.asarray(sequence).copy()

    if min_duration <= 0:
        return sequence

    changed = True

    while changed:

        changed = False

        labels, starts, ends = get_segments(sequence)

        if len(labels) <= 1:
            break

        for i in range(len(labels)):

            start = starts[i]
            end = ends[i]

            duration = end - start

            if duration >= min_duration:
                continue

            # First segment
            if i == 0:
                new_label = labels[i + 1]

            # Last segment
            elif i == len(labels) - 1:
                new_label = labels[i - 1]

            # Middle segment
            else:
                left_label = labels[i - 1]
                right_label = labels[i + 1]

                if left_label == right_label:
                    new_label = left_label

                else:
                    left_duration = (
                        ends[i - 1] - starts[i - 1]
                    )

                    right_duration = (
                        ends[i + 1] - starts[i + 1]
                    )

                    if left_duration >= right_duration:
                        new_label = left_label
                    else:
                        new_label = right_label

            sequence[start:end] = new_label

            changed = True

            # Recompute segments after each merge
            break

    return sequence
# ==========================================================
# EVALUATE
# ==========================================================

edit_scores = []

thresholds = [
    0.10,
    0.25,
    0.50
]

counts = {
    threshold: {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
    for threshold in thresholds
}


with torch.no_grad():

    for batch in test_loader:

        features = (
            batch["features"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
        )

        video_id = (
            batch["video_id"][0]
        )

        valid = (
            labels != IGNORE_INDEX
        )

        if valid.sum() == 0:
            continue

        mask = (
            valid
            .float()
            .unsqueeze(1)
        )

        outputs = model(
            features,
            mask
        )

        predictions = (
            outputs[-1]
            .argmax(dim=1)
        )

        # ----------------------------------------------
        # Remove ignored annotation positions
        # ----------------------------------------------
        prediction = (
            predictions[valid]
            .cpu()
            .numpy()
        )

        ground_truth = (
            labels[valid]
            .cpu()
            .numpy()
        )

        # Post-processing selected using validation set
        prediction = remove_short_segments(
            prediction,
            min_duration=15
        )

        # ----------------------------------------------
        # Edit score
        # ----------------------------------------------

        video_edit = edit_score(
            prediction,
            ground_truth
        )

        edit_scores.append(
            video_edit
        )

        # ----------------------------------------------
        # Segment F1
        # ----------------------------------------------

        video_f1_values = {}

        for threshold in thresholds:

            tp, fp, fn = segment_f1(
                prediction,
                ground_truth,
                threshold
            )

            counts[threshold]["tp"] += tp
            counts[threshold]["fp"] += fp
            counts[threshold]["fn"] += fn

            video_f1_values[
                threshold
            ] = f1_from_counts(
                tp,
                fp,
                fn
            )

        print(
            f"{video_id:20s} | "
            f"Edit: {video_edit:6.2f} | "
            f"F1@10: {video_f1_values[0.10]:6.2f} | "
            f"F1@25: {video_f1_values[0.25]:6.2f} | "
            f"F1@50: {video_f1_values[0.50]:6.2f}"
        )


# ==========================================================
# GLOBAL RESULTS
# ==========================================================

mean_edit = np.mean(
    edit_scores
)

global_f1 = {}

for threshold in thresholds:

    global_f1[threshold] = (
        f1_from_counts(
            counts[threshold]["tp"],
            counts[threshold]["fp"],
            counts[threshold]["fn"]
        )
    )


print("\n======================================")
print("TEMPORAL SEGMENTATION RESULTS")
print("======================================")

print(
    f"Edit score: {mean_edit:.2f}"
)

print(
    f"F1@10:      "
    f"{global_f1[0.10]:.2f}"
)

print(
    f"F1@25:      "
    f"{global_f1[0.25]:.2f}"
)

print(
    f"F1@50:      "
    f"{global_f1[0.50]:.2f}"
)