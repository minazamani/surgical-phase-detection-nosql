import json

import torch

from dataset import SurgicalPhaseDataset
from mstcn import MultiStageModel


PROCESSED_DIR = (
    "/home/mina/PycharmProjects/"
    "phasedetection/processed"
)


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="train"
)

sample = dataset[0]

features = sample["features"]
labels = sample["labels"]


# ----------------------------------------------------------
# Number of classes
# ----------------------------------------------------------

with open(
    f"{PROCESSED_DIR}/label_maps.json",
    "r"
) as f:
    label_maps = json.load(f)

phase_to_id = label_maps["phase_to_id"]

valid_classes = {
    name: idx
    for name, idx in phase_to_id.items()
    if idx >= 0
}

num_classes = len(valid_classes)

print(
    "Number of classes:",
    num_classes
)


# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

model = MultiStageModel(
    num_stages=4,
    num_layers=10,
    num_f_maps=64,
    input_dim=2048,
    num_classes=num_classes,
)


# ----------------------------------------------------------
# Add batch dimension
# ----------------------------------------------------------

# features:
# [2048, T]
#
# -> [1, 2048, T]

features = features.unsqueeze(0)

T = features.shape[-1]

mask = torch.ones(
    1,
    1,
    T
)


# ----------------------------------------------------------
# Forward pass
# ----------------------------------------------------------

with torch.no_grad():

    outputs = model(
        features,
        mask
    )


print(
    "Input:",
    features.shape
)

print(
    "Output:",
    outputs.shape
)

print(
    "Labels:",
    labels.shape
)