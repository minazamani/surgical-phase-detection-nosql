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

# We will test these on VALIDATION only
MIN_SEGMENT_DURATIONS = [
    0,
    5,
    10,
    15,
    20,
    30,
]


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


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="validation"
)

loader = DataLoader(
    dataset,
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
# SEGMENT HELPERS
# ==========================================================

def get_segments(sequence):
    """
    Convert:
        [1,1,1,2,2,3,3]

    into:
        labels = [1,2,3]
        starts = [0,3,5]
        ends   = [3,5,7]
    """

    sequence = np.asarray(sequence)

    if len(sequence) == 0:
        return [], [], []

    labels = []
    starts = []
    ends = []

    current_label = sequence[0]
    start = 0

    for i in range(1, len(sequence)):

        if sequence[i] != current_label:

            labels.append(
                int(current_label)
            )

            starts.append(
                start
            )

            ends.append(
                i
            )

            current_label = sequence[i]
            start = i

    labels.append(
        int(current_label)
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
# POST-PROCESSING
# ==========================================================

def remove_short_segments(
    sequence,
    min_duration
):
    """
    Merge segments shorter than min_duration.

    Strategy:

    - If left and right neighbors have same phase,
      merge into that phase.

    - Otherwise merge into the longer neighboring segment.

    - At sequence boundaries, merge with the only neighbor.
    """

    sequence = np.asarray(
        sequence
    ).copy()

    if min_duration <= 0:
        return sequence

    changed = True

    while changed:

        changed = False

        labels, starts, ends = (
            get_segments(sequence)
        )

        if len(labels) <= 1:
            break

        for i in range(len(labels)):

            start = starts[i]
            end = ends[i]

            duration = end - start

            if duration >= min_duration:
                continue

            # ------------------------------------------
            # First segment
            # ------------------------------------------

            if i == 0:

                new_label = labels[i + 1]

            # ------------------------------------------
            # Last segment
            # ------------------------------------------

            elif i == len(labels) - 1:

                new_label = labels[i - 1]

            # ------------------------------------------
            # Middle segment
            # ------------------------------------------

            else:

                left_label = labels[i - 1]
                right_label = labels[i + 1]

                # Best case:
                # A - tiny B - A
                if left_label == right_label:

                    new_label = left_label

                else:

                    left_duration = (
                        ends[i - 1]
                        - starts[i - 1]
                    )

                    right_duration = (
                        ends[i + 1]
                        - starts[i + 1]
                    )

                    if (
                        left_duration
                        >= right_duration
                    ):
                        new_label = left_label

                    else:
                        new_label = right_label

            sequence[
                start:end
            ] = new_label

            changed = True

            # Recompute segments after every merge
            break

    return sequence


# ==========================================================
# EDIT SCORE
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


def edit_score(
    prediction,
    ground_truth
):

    pred_labels, _, _ = (
        get_segments(prediction)
    )

    gt_labels, _, _ = (
        get_segments(ground_truth)
    )

    if (
        len(pred_labels) == 0
        or len(gt_labels) == 0
    ):
        return 0.0

    distance = (
        levenshtein_distance(
            pred_labels,
            gt_labels
        )
    )

    max_len = max(
        len(pred_labels),
        len(gt_labels)
    )

    return 100.0 * (
        1.0
        - distance / max_len
    )


# ==========================================================
# SEGMENT F1
# ==========================================================

def segment_f1_counts(
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

    pred_starts = np.array(pred_starts)
    pred_ends = np.array(pred_ends)

    gt_starts = np.array(gt_starts)
    gt_ends = np.array(gt_ends)

    gt_labels = np.array(gt_labels)

    if len(pred_labels) == 0:

        return (
            0,
            0,
            len(gt_labels)
        )

    if len(gt_labels) == 0:

        return (
            0,
            len(pred_labels),
            0
        )

    hits = np.zeros(
        len(gt_labels),
        dtype=np.int32
    )

    tp = 0
    fp = 0

    for j in range(
        len(pred_labels)
    ):

        intersection = (
            np.minimum(
                pred_ends[j],
                gt_ends
            )
            -
            np.maximum(
                pred_starts[j],
                gt_starts
            )
        )

        intersection = np.maximum(
            intersection,
            0
        )

        union = (
            np.maximum(
                pred_ends[j],
                gt_ends
            )
            -
            np.minimum(
                pred_starts[j],
                gt_starts
            )
        )

        iou = (
            intersection
            / np.maximum(
                union,
                1e-8
            )
        )

        # Must be same phase
        iou = iou * (
            gt_labels
            == pred_labels[j]
        )

        idx = np.argmax(iou)

        if (
            iou[idx]
            >= overlap_threshold
            and hits[idx] == 0
        ):

            tp += 1
            hits[idx] = 1

        else:

            fp += 1

    fn = (
        len(gt_labels)
        - hits.sum()
    )

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

    if (
        precision + recall
        == 0
    ):
        return 0.0

    return (
        100
        * 2
        * precision
        * recall
        / (precision + recall)
    )


# ==========================================================
# GET RAW MODEL PREDICTIONS ON VALIDATION SET
# ==========================================================

all_sequences = []


with torch.no_grad():

    for batch in loader:

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

        all_sequences.append(
            {
                "video_id":
                    video_id,

                "prediction":
                    prediction,

                "ground_truth":
                    ground_truth,
            }
        )


# ==========================================================
# TEST DIFFERENT MINIMUM DURATIONS
# ==========================================================

print(
    "\n======================================"
)

print(
    "POST-PROCESSING VALIDATION"
)

print(
    "======================================"
)


best_threshold = None
best_f1_50 = -1


for min_duration in (
    MIN_SEGMENT_DURATIONS
):

    edit_scores = []

    counts = {
        0.10: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        },

        0.25: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        },

        0.50: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    }

    for item in all_sequences:

        prediction = (
            item["prediction"]
        )

        ground_truth = (
            item["ground_truth"]
        )

        filtered = (
            remove_short_segments(
                prediction,
                min_duration
            )
        )

        edit_scores.append(
            edit_score(
                filtered,
                ground_truth
            )
        )

        for threshold in counts:

            tp, fp, fn = (
                segment_f1_counts(
                    filtered,
                    ground_truth,
                    threshold
                )
            )

            counts[
                threshold
            ]["tp"] += tp

            counts[
                threshold
            ]["fp"] += fp

            counts[
                threshold
            ]["fn"] += fn

    mean_edit = np.mean(
        edit_scores
    )

    f1_10 = f1_from_counts(
        **counts[0.10]
    )

    f1_25 = f1_from_counts(
        **counts[0.25]
    )

    f1_50 = f1_from_counts(
        **counts[0.50]
    )

    print(
        f"\nMinimum duration: "
        f"{min_duration:2d} sec"
    )

    print(
        f"Edit:  {mean_edit:.2f}"
    )

    print(
        f"F1@10: {f1_10:.2f}"
    )

    print(
        f"F1@25: {f1_25:.2f}"
    )

    print(
        f"F1@50: {f1_50:.2f}"
    )

    if f1_50 > best_f1_50:

        best_f1_50 = f1_50
        best_threshold = (
            min_duration
        )


# ==========================================================
# BEST THRESHOLD
# ==========================================================

print(
    "\n======================================"
)

print(
    "BEST VALIDATION SETTING"
)

print(
    "======================================"
)

print(
    "Minimum segment duration:",
    best_threshold,
    "seconds"
)

print(
    "Validation F1@50:",
    f"{best_f1_50:.2f}"
)