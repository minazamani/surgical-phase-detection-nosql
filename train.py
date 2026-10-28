import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import SurgicalPhaseDataset
from mstcn import MultiStageModel


# ==========================================================
# CONFIGURATION
# ==========================================================

PROCESSED_DIR = Path(
    "/home/mina/PycharmProjects/phasedetection/processed"
)

OUTPUT_DIR = Path("training_output")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

NUM_EPOCHS = 50
BATCH_SIZE = 1

LEARNING_RATE = 5e-4

NUM_STAGES = 4
NUM_LAYERS = 10
NUM_F_MAPS = 64

FEATURE_DIM = 2048

IGNORE_INDEX = -100

# Weight of temporal smoothing term
SMOOTHING_WEIGHT = 0.15


print("Device:", DEVICE)


# ==========================================================
# LOAD CLASS INFORMATION
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


print("\nClasses:")

for name, idx in sorted(
    valid_classes.items(),
    key=lambda x: x[1]
):
    print(
        f"{idx}: {name}"
    )


print(
    "\nNumber of classes:",
    NUM_CLASSES
)


# ==========================================================
# DATASETS
# ==========================================================

train_dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="train"
)

val_dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="validation"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)


print(
    "\nTraining fragments:",
    len(train_dataset)
)

print(
    "Validation fragments:",
    len(val_dataset)
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

model = model.to(DEVICE)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# CLASSIFICATION LOSS
# ==========================================================

classification_loss = torch.nn.CrossEntropyLoss(
    ignore_index=IGNORE_INDEX
)


# ==========================================================
# TEMPORAL SMOOTHING LOSS
# ==========================================================

def temporal_smoothing_loss(
    logits,
    labels,
):
    """
    logits:
        [B, C, T]

    labels:
        [B, T]

    Encourages neighboring predictions
    to change smoothly over time.
    """

    if logits.shape[-1] <= 1:
        return torch.tensor(
            0.0,
            device=logits.device
        )

    log_probs = F.log_softmax(
        logits,
        dim=1
    )

    current = log_probs[:, :, 1:]

    previous = (
        log_probs[:, :, :-1]
        .detach()
    )

    difference = (
        current - previous
    ) ** 2

    difference = torch.clamp(
        difference,
        min=0,
        max=16
    )

    valid_current = (
        labels[:, 1:]
        != IGNORE_INDEX
    )

    valid_previous = (
        labels[:, :-1]
        != IGNORE_INDEX
    )

    valid_transition = (
        valid_current
        & valid_previous
    )

    if valid_transition.sum() == 0:
        return torch.tensor(
            0.0,
            device=logits.device
        )

    # Average over class dimension
    difference = difference.mean(
        dim=1
    )

    return difference[
        valid_transition
    ].mean()


# ==========================================================
# TOTAL LOSS
# ==========================================================

def compute_loss(
    outputs,
    labels,
):
    """
    outputs:
        [num_stages, B, C, T]

    labels:
        [B, T]
    """

    total_loss = 0.0

    for stage_output in outputs:

        ce = classification_loss(
            stage_output,
            labels
        )

        smooth = temporal_smoothing_loss(
            stage_output,
            labels
        )

        total_loss += (
            ce
            + SMOOTHING_WEIGHT * smooth
        )

    return total_loss


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(
    predictions,
    labels,
):
    valid = labels != IGNORE_INDEX

    labels = labels[valid]
    predictions = predictions[valid]

    if len(labels) == 0:

        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }

    accuracy = (
        predictions == labels
    ).mean()

    macro_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    processed_videos = 0

    for batch in train_loader:

        features = batch[
            "features"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        video_id = batch[
            "video_id"
        ][0]

        # ----------------------------------------------
        # Check valid labels
        # ----------------------------------------------

        valid = (
            labels != IGNORE_INDEX
        )

        if valid.sum() == 0:

            print(
                "Skipping:",
                video_id
            )

            continue

        # ----------------------------------------------
        # Mask
        # ----------------------------------------------

        mask = (
            valid
            .float()
            .unsqueeze(1)
        )

        # [B, 1, T]

        # ----------------------------------------------
        # Forward
        # ----------------------------------------------

        optimizer.zero_grad()

        outputs = model(
            features,
            mask
        )

        # ----------------------------------------------
        # Loss
        # ----------------------------------------------

        loss = compute_loss(
            outputs,
            labels
        )

        if not torch.isfinite(loss):

            print(
                f"WARNING: non-finite loss "
                f"for {video_id}"
            )

            continue

        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_loss += loss.item()

        processed_videos += 1

        # ----------------------------------------------
        # Prediction from final stage
        # ----------------------------------------------

        predictions = (
            outputs[-1]
            .argmax(dim=1)
        )

        valid_predictions = (
            predictions[valid]
            .detach()
            .cpu()
            .numpy()
        )

        valid_labels = (
            labels[valid]
            .detach()
            .cpu()
            .numpy()
        )

        all_predictions.append(
            valid_predictions
        )

        all_labels.append(
            valid_labels
        )

    if processed_videos == 0:

        raise RuntimeError(
            "No videos were processed "
            "during training."
        )

    all_predictions = np.concatenate(
        all_predictions
    )

    all_labels = np.concatenate(
        all_labels
    )

    metrics = calculate_metrics(
        all_predictions,
        all_labels
    )

    average_loss = (
        total_loss
        / processed_videos
    )

    return (
        average_loss,
        metrics
    )


# ==========================================================
# VALIDATION
# ==========================================================

def validate():

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    processed_videos = 0

    with torch.no_grad():

        for batch in val_loader:

            features = batch[
                "features"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

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

            loss = compute_loss(
                outputs,
                labels
            )

            if not torch.isfinite(loss):
                continue

            total_loss += loss.item()

            processed_videos += 1

            predictions = (
                outputs[-1]
                .argmax(dim=1)
            )

            all_predictions.append(
                predictions[valid]
                .cpu()
                .numpy()
            )

            all_labels.append(
                labels[valid]
                .cpu()
                .numpy()
            )

    if processed_videos == 0:

        return float("nan"), {
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }

    predictions = np.concatenate(
        all_predictions
    )

    labels = np.concatenate(
        all_labels
    )

    metrics = calculate_metrics(
        predictions,
        labels
    )

    average_loss = (
        total_loss
        / processed_videos
    )

    return (
        average_loss,
        metrics
    )


# ==========================================================
# TRAINING LOOP
# ==========================================================

best_val_f1 = -1.0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    train_loss, train_metrics = (
        train_one_epoch()
    )

    val_loss, val_metrics = (
        validate()
    )

    print(
        f"\nEpoch {epoch:03d}/{NUM_EPOCHS}"
    )

    print(
        f"Train | "
        f"Loss: {train_loss:.4f} | "
        f"Accuracy: "
        f"{train_metrics['accuracy']:.4f} | "
        f"Macro-F1: "
        f"{train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Val   | "
        f"Loss: {val_loss:.4f} | "
        f"Accuracy: "
        f"{val_metrics['accuracy']:.4f} | "
        f"Macro-F1: "
        f"{val_metrics['macro_f1']:.4f}"
    )

    # ------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics[
                "macro_f1"
            ]
        )

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_macro_f1":
                best_val_f1,

            "phase_to_id":
                phase_to_id,

            "num_classes":
                NUM_CLASSES,

            "feature_dim":
                FEATURE_DIM,
        }

        torch.save(
            checkpoint,
            OUTPUT_DIR
            / "best_model.pt"
        )

        print(
            "✓ Saved new best model "
            f"(Macro-F1 = "
            f"{best_val_f1:.4f})"
        )


# ==========================================================
# FINISHED
# ==========================================================

print(
    "\nTraining finished."
)

print(
    "Best validation Macro-F1:",
    best_val_f1
)

print(
    "Model saved at:",
    (
        OUTPUT_DIR
        / "best_model.pt"
    ).resolve()
)