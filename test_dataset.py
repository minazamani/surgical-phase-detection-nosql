from dataset import SurgicalPhaseDataset


PROCESSED_DIR = (
    "/home/mina/PycharmProjects/"
    "phasedetection/processed"
)


train_dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="train"
)

val_dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="validation"
)

test_dataset = SurgicalPhaseDataset(
    PROCESSED_DIR,
    split="test"
)


print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


sample = train_dataset[0]

print("\nExample sample:")
print("Video:", sample["video_id"])
print("Surgery:", sample["surgery_id"])
print(
    "Features:",
    sample["features"].shape
)
print(
    "Labels:",
    sample["labels"].shape
)

print(
    "Valid labels:",
    (sample["labels"] != -100).sum().item()
)

print(
    "Ignored labels:",
    (sample["labels"] == -100).sum().item()
)