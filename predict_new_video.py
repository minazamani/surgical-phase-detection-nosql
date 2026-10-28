import json
from pathlib import Path
import os
from pymongo import MongoClient
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

from mstcn import MultiStageModel
from pymongo import MongoClient

# ==========================================================
# CONFIGURATION

# ==========================================================
# MONGODB CONFIGURATION
# ==========================================================

MONGO_URI = "mongodb://localhost:27017"

MONGO_DATABASE = "surgical_phase_detection"

MONGO_COLLECTION = "video_phase_predictions"
MONGO_URI = os.environ["MONGO_URI"]
# ==========================================================

VIDEO_PATH = Path(
    "/media/mina/88f340ce-0749-45be-bb45-1f8bc5f3a486/U141-Channel0-part-0001.mp4"
)

PROCESSED_DIR = Path(
    "/home/mina/PycharmProjects/phasedetection/processed"
)

CHECKPOINT_PATH = Path(
    "training_output/best_model.pt"
)

OUTPUT_DIR = Path(
    "new_video_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SAMPLING_RATE_FPS = 1

MIN_SEGMENT_DURATION = 5

NUM_STAGES = 4
NUM_LAYERS = 10
NUM_F_MAPS = 64
FEATURE_DIM = 2048


print("Device:", DEVICE)


# ==========================================================
# LOAD PHASE LABEL MAP
# ==========================================================

with open(
    PROCESSED_DIR / "label_maps.json",
    "r",
    encoding="utf-8"
) as f:

    label_maps = json.load(f)


phase_to_id = label_maps["phase_to_id"]


valid_classes = {
    name: idx
    for name, idx
    in phase_to_id.items()
    if idx >= 0
}


NUM_CLASSES = len(
    valid_classes
)


id_to_phase = {
    idx: name
    for name, idx
    in valid_classes.items()
}


print("\nPhases:")

for idx in sorted(id_to_phase):
    print(
        idx,
        "->",
        id_to_phase[idx]
    )

def save_to_mongodb(result):

    client = MongoClient(MONGO_URI)

    db = client[MONGO_DATABASE]

    collection = db[MONGO_COLLECTION]

    video_id = result["video_id"]

    collection.replace_one(
        {"video_id": video_id},
        result,
        upsert=True
    )

    collection.create_index(
        "video_id",
        unique=True
    )

    collection.create_index(
        "segments.phase"
    )

    collection.create_index(
        "segments.start_sec"
    )

    print("\nSaved to MongoDB")
    print("Database:", MONGO_DATABASE)
    print("Collection:", MONGO_COLLECTION)
    print("Video ID:", video_id)

    client.close()
# ==========================================================
# LOAD RESNET-50
# ==========================================================

print(
    "\nLoading ResNet-50..."
)

weights = ResNet50_Weights.DEFAULT

resnet = resnet50(
    weights=weights
)

# Remove ImageNet classification layer
resnet.fc = nn.Identity()

resnet = resnet.to(
    DEVICE
)

resnet.eval()

image_preprocess = (
    weights.transforms()
)


# ==========================================================
# LOAD MS-TCN
# ==========================================================

print(
    "Loading MS-TCN..."
)


mstcn = MultiStageModel(
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


mstcn.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


mstcn = mstcn.to(
    DEVICE
)

mstcn.eval()


print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)

print(
    "Validation Macro-F1:",
    checkpoint.get(
        "val_macro_f1",
        "unknown"
    )
)


# ==========================================================
# TIME FORMAT
# ==========================================================

def seconds_to_hms(seconds):

    seconds = int(
        round(seconds)
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ==========================================================
# EXTRACT FEATURES FROM NEW VIDEO
# ==========================================================

def extract_features(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )


    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if original_fps <= 0:

        raise RuntimeError(
            "Invalid FPS reported "
            "by the video."
        )


    duration_sec = (
        frame_count
        / original_fps
    )


    print(
        "\nVideo information:"
    )

    print(
        "FPS:",
        original_fps
    )

    print(
        "Frames:",
        frame_count
    )

    print(
        "Duration:",
        seconds_to_hms(
            duration_sec
        )
    )


    frame_step = (
        original_fps
        / SAMPLING_RATE_FPS
    )


    features = []

    sample_index = 0


    with torch.no_grad():

        while True:

            frame_number = int(
                round(
                    sample_index
                    * frame_step
                )
            )


            if (
                frame_number
                >= frame_count
            ):

                break


            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )


            success, frame = (
                cap.read()
            )


            if not success:
                break


            # OpenCV = BGR
            # ResNet expects RGB

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            image = Image.fromarray(
                frame_rgb
            )


            tensor = (
                image_preprocess(
                    image
                )
                .unsqueeze(0)
                .to(DEVICE)
            )


            feature = resnet(
                tensor
            )


            feature = (
                feature
                .squeeze(0)
                .cpu()
                .numpy()
            )


            features.append(
                feature
            )


            sample_index += 1


    cap.release()


    if len(features) == 0:

        raise RuntimeError(
            "No features were extracted."
        )


    features = np.stack(
        features
    ).astype(
        np.float32
    )


    print(
        "Extracted feature shape:",
        features.shape
    )


    return (
        features,
        duration_sec
    )


# ==========================================================
# SEGMENT EXTRACTION
# ==========================================================

def get_segments(sequence):

    sequence = np.asarray(
        sequence
    )


    if len(sequence) == 0:

        return (
            [],
            [],
            []
        )


    labels = []
    starts = []
    ends = []


    current_label = (
        sequence[0]
    )

    start = 0


    for i in range(
        1,
        len(sequence)
    ):

        if (
            sequence[i]
            != current_label
        ):

            labels.append(
                int(
                    current_label
                )
            )

            starts.append(
                start
            )

            ends.append(
                i
            )


            current_label = (
                sequence[i]
            )

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
# REMOVE SHORT SEGMENTS
# ==========================================================

def remove_short_segments(
    sequence,
    min_duration
):

    sequence = np.asarray(
        sequence
    ).copy()


    if min_duration <= 0:

        return sequence


    changed = True


    while changed:

        changed = False


        labels, starts, ends = (
            get_segments(
                sequence
            )
        )


        if len(labels) <= 1:
            break


        for i in range(
            len(labels)
        ):

            start = (
                starts[i]
            )

            end = (
                ends[i]
            )

            duration = (
                end - start
            )


            if (
                duration
                >= min_duration
            ):
                continue


            # First segment
            if i == 0:

                new_label = (
                    labels[i + 1]
                )


            # Last segment
            elif (
                i
                == len(labels) - 1
            ):

                new_label = (
                    labels[i - 1]
                )


            # Middle segment
            else:

                left_label = (
                    labels[i - 1]
                )

                right_label = (
                    labels[i + 1]
                )


                if (
                    left_label
                    == right_label
                ):

                    new_label = (
                        left_label
                    )

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

                        new_label = (
                            left_label
                        )

                    else:

                        new_label = (
                            right_label
                        )


            sequence[
                start:end
            ] = new_label


            changed = True

            break


    return sequence


# ==========================================================
# MS-TCN PREDICTION
# ==========================================================

def predict_phases(
    features
):

    # Input features:
    #
    # [T, 2048]
    #
    # Model expects:
    #
    # [B, 2048, T]

    features_tensor = (
        torch.from_numpy(
            features
        )
        .transpose(
            0,
            1
        )
        .unsqueeze(
            0
        )
        .to(
            DEVICE
        )
    )


    T = (
        features_tensor
        .shape[-1]
    )


    mask = torch.ones(
        1,
        1,
        T,
        device=DEVICE,
        dtype=torch.float32
    )


    with torch.no_grad():

        outputs = mstcn(
            features_tensor,
            mask
        )


        predictions = (
            outputs[-1]
            .argmax(
                dim=1
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )


    return predictions

# ==========================================================
# CONVERT PREDICTIONS TO INTERVALS
# ==========================================================

def predictions_to_intervals(
    predictions
):

    predictions = (
        remove_short_segments(
            predictions,
            MIN_SEGMENT_DURATION
        )
    )


    labels, starts, ends = (
        get_segments(
            predictions
        )
    )


    segments = []


    for (
        label,
        start,
        end
    ) in zip(
        labels,
        starts,
        ends
    ):

        start_sec = (
            start
            / SAMPLING_RATE_FPS
        )

        end_sec = (
            end
            / SAMPLING_RATE_FPS
        )

        duration_sec = (
            end_sec
            - start_sec
        )


        segments.append(
            {
                "phase_id":
                    int(label),

                "phase":
                    id_to_phase[
                        int(label)
                    ],

                "start_sec":
                    float(
                        start_sec
                    ),

                "end_sec":
                    float(
                        end_sec
                    ),

                "duration_sec":
                    float(
                        duration_sec
                    ),

                "start_time":
                    seconds_to_hms(
                        start_sec
                    ),

                "end_time":
                    seconds_to_hms(
                        end_sec
                    ),

                "duration":
                    seconds_to_hms(
                        duration_sec
                    ),
            }
        )


    return segments

# ==========================================================
# MAIN
# ==========================================================

def main():

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{VIDEO_PATH}"
        )


    print(
        "\n======================================"
    )

    print(
        "NEW VIDEO PHASE DETECTION"
    )

    print(
        "======================================"
    )


    print(
        "Video:",
        VIDEO_PATH
    )


    # ------------------------------------------------------
    # 1. Extract ResNet features
    # ------------------------------------------------------

    features, real_duration_sec = (
        extract_features(
            VIDEO_PATH
        )
    )


    # ------------------------------------------------------
    # 2. MS-TCN prediction
    # ------------------------------------------------------

    predictions = predict_phases(
        features
    )


    # ------------------------------------------------------
    # 3. Convert to intervals
    # ------------------------------------------------------

    segments = (
        predictions_to_intervals(
            predictions
        )
    )


    # ------------------------------------------------------
    # 4. Display
    # ------------------------------------------------------

    print(
        "\nPredicted phases:"
    )


    for segment in segments:

        print(
            segment[
                "start_time"
            ],
            "->",
            segment[
                "end_time"
            ],
            "|",
            segment[
                "phase"
            ]
        )


    # ------------------------------------------------------
    # 5. Save JSON
    # ------------------------------------------------------

    video_id = (
        VIDEO_PATH.stem
    )


    result = {

        "video_id":
            video_id,

        "source_video":
            VIDEO_PATH.name,

        "model": {
            "visual_encoder":
                "ResNet50",

            "temporal_model":
                "MS-TCN",

            "checkpoint":
                CHECKPOINT_PATH.name,

            "checkpoint_epoch":
                checkpoint.get(
                    "epoch"
                ),

            "validation_macro_f1":
                checkpoint.get(
                    "val_macro_f1"
                ),
        },

        "sampling": {
            "fps":
                SAMPLING_RATE_FPS,

            "number_timesteps":
                int(
                    len(predictions)
                ),
        },

        "video_duration": {
            "seconds":
                float(
                    real_duration_sec
                ),

            "time":
                seconds_to_hms(
                    real_duration_sec
                ),
        },

        "postprocessing": {
            "minimum_segment_duration_sec":
                MIN_SEGMENT_DURATION,

            "selected_on":
                "validation set"
        },

        "number_segments":
            len(segments),

        "segments":
            segments
    }


    output_path = (
        OUTPUT_DIR
        / f"{video_id}_phases.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    save_to_mongodb(
        result
    )
    print(
        "\n======================================"
    )

    print(
        "Finished"
    )

    print(
        "======================================"
    )


    print(
        "Detected segments:",
        len(segments)
    )

    print(
        "Result saved to:"
    )

    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()