from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = Path("/media/mina/88f340ce-0749-45be-bb45-1f8bc5f3a486/phasedetectionvideo/")
PROCESSED_DIR = Path("/home/mina/PycharmProjects/PythonProject2/processed")

SAMPLE_FPS = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD RESNET-50
# ============================================================

print(f"Using device: {DEVICE}")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

# Remove final classification layer
#
# Original ResNet:
# image -> features -> FC -> ImageNet class
#
# We want:
# image -> 2048-D features
model.fc = nn.Identity()

model = model.to(DEVICE)
model.eval()

preprocess = weights.transforms()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_from_video(video_path):
    """
    Extract one ResNet-50 feature vector per second.

    Output shape:
        [T, 2048]

    where T ~= video duration in seconds.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration_sec = frame_count / original_fps

    print(
        f"  FPS: {original_fps:.2f}"
        f" | Frames: {frame_count}"
        f" | Duration: {duration_sec:.1f} sec"
    )

    # Example:
    # video = 30 fps
    # SAMPLE_FPS = 1
    #
    # take approximately every 30th frame

    frame_step = original_fps / SAMPLE_FPS

    features = []

    sample_index = 0

    with torch.no_grad():

        while True:

            frame_number = int(
                round(sample_index * frame_step)
            )

            if frame_number >= frame_count:
                break

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            success, frame = cap.read()

            if not success:
                break

            # OpenCV = BGR
            # ResNet expects RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(frame_rgb)

            tensor = preprocess(image)

            # Add batch dimension:
            #
            # [3, H, W]
            # ->
            # [1, 3, H, W]

            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(DEVICE)

            feature = model(tensor)

            # [1, 2048] -> [2048]
            feature = (
                feature
                .squeeze(0)
                .cpu()
                .numpy()
            )

            features.append(feature)

            sample_index += 1

    cap.release()

    if len(features) == 0:
        raise RuntimeError(
            f"No features extracted from {video_path}"
        )

    features = np.stack(features)

    return features


# ============================================================
# MATCH FEATURES WITH LABELS
# ============================================================

def align_features_and_labels(
    features,
    phase_labels,
):
    """
    Ensure features and labels have identical T.

    Small differences can happen because:
      - video metadata duration is not exact
      - Excel annotation ends slightly before/after video
      - frame sampling rounding
    """

    number_features = len(features)
    number_labels = len(phase_labels)

    print(
        f"  Feature samples: {number_features}"
    )

    print(
        f"  Label samples:   {number_labels}"
    )

    if number_features == number_labels:
        print("  Alignment: OK")
        return features, phase_labels

    difference = abs(
        number_features - number_labels
    )

    print(
        f"  WARNING: difference = {difference}"
    )

    # Do not silently tolerate large differences
    if difference > 5:

        print(
            "  WARNING: large feature/label "
            "duration mismatch."
        )

        print(
            "  Please verify the video and "
            "Excel annotation."
        )

    # For now use common duration
    minimum_length = min(
        number_features,
        number_labels,
    )

    features = features[:minimum_length]
    phase_labels = phase_labels[:minimum_length]

    print(
        f"  Cropped both to: {minimum_length}"
    )

    return features, phase_labels


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    video_id = video_path.stem

    print("\n===================================")
    print(f"Processing: {video_id}")
    print("===================================")

    output_dir = PROCESSED_DIR / video_id

    if not output_dir.exists():

        print(
            f"Skipping: no processed folder found "
            f"for {video_id}"
        )

        return

    phase_path = (
        output_dir / "phase_labels.npy"
    )

    if not phase_path.exists():

        print(
            f"Skipping: phase_labels.npy missing "
            f"for {video_id}"
        )

        return

    features_path = (
        output_dir / "features.npy"
    )

    # --------------------------------------------------------
    # Optional:
    # don't calculate again if features already exist
    # --------------------------------------------------------

    if features_path.exists():

        print(
            "features.npy already exists. "
            "Skipping."
        )

        return

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    phase_labels = np.load(
        phase_path
    )

    # --------------------------------------------------------
    # Extract visual features
    # --------------------------------------------------------

    features = extract_features_from_video(
        video_path
    )

    # --------------------------------------------------------
    # Compare timeline lengths
    # --------------------------------------------------------

    features, aligned_labels = (
        align_features_and_labels(
            features,
            phase_labels,
        )
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    np.save(
        features_path,
        features.astype(np.float32),
    )

    # If cropping was necessary, save aligned labels separately.
    #
    # This avoids accidentally overwriting your original
    # ground-truth annotation.

    aligned_label_path = (
        output_dir / "phase_labels_aligned.npy"
    )

    np.save(
        aligned_label_path,
        aligned_labels.astype(np.int64),
    )

    print(
        f"  Saved features: {features.shape}"
    )

    print(
        f"  Saved to: {features_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not VIDEO_DIR.exists():

        raise FileNotFoundError(
            f"Video directory not found: "
            f"{VIDEO_DIR.resolve()}"
        )

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".mpeg",
        ".mpg",
    }

    videos = sorted(
        [
            path
            for path in VIDEO_DIR.iterdir()
            if path.suffix.lower()
            in video_extensions
        ]
    )

    print(
        f"Found {len(videos)} videos."
    )

    for video_path in videos:

        try:

            process_video(
                video_path
            )

        except Exception as error:

            print(
                f"\nERROR processing "
                f"{video_path.name}:"
            )

            print(error)


if __name__ == "__main__":
    main()