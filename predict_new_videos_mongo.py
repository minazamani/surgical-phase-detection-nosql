import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pymongo import MongoClient
from torchvision.models import resnet50, ResNet50_Weights

from mstcn import MultiStageModel


# ==========================================================
# CONFIGURATION
# ==========================================================

# Folder containing NEW videos to predict
VIDEO_FOLDER = Path(
    "/media/mina/88f340ce-0749-45be-bb45-1f8bc5f3a486"
)

# Folder containing label_maps.json
PROCESSED_DIR = Path(
    "/home/mina/PycharmProjects/phasedetection/processed"
)

# Trained MS-TCN model
CHECKPOINT_PATH = Path(
    "training_output/best_model.pt"
)

# JSON prediction output
OUTPUT_DIR = Path(
    "new_video_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# VIDEO CONFIGURATION
# ==========================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}

# Same sampling rate used for training
SAMPLING_RATE_FPS = 1

# Temporal cleanup
MIN_SEGMENT_DURATION = 5


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

NUM_STAGES = 4
NUM_LAYERS = 10
NUM_F_MAPS = 64
FEATURE_DIM = 2048


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    DEVICE
)


# ==========================================================
# MONGODB CONFIGURATION
# ==========================================================

MONGO_DATABASE = (
    "surgical_phase_detection"
)

MONGO_COLLECTION = (
    "video_phase_predictions"
)


MONGO_URI = os.getenv(
    "MONGO_URI"
)


if not MONGO_URI:

    raise RuntimeError(
        "\nMONGO_URI environment variable "
        "is not set.\n\n"
        "Example:\n"
        "export MONGO_URI="
        "'mongodb://mongoAdmin:"
        "PASSWORD@127.0.0.1:27017/"
        "?authSource=admin'"
    )


# ==========================================================
# LOAD LABEL MAP
# ==========================================================

label_map_path = (
    PROCESSED_DIR
    / "label_maps.json"
)


if not label_map_path.exists():

    raise FileNotFoundError(
        f"label_maps.json not found:\n"
        f"{label_map_path}"
    )


with open(
    label_map_path,
    "r",
    encoding="utf-8"
) as f:

    label_maps = json.load(
        f
    )


phase_to_id = (
    label_maps[
        "phase_to_id"
    ]
)


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


print(
    "\nNumber of phases:",
    NUM_CLASSES
)


for idx in sorted(
    id_to_phase
):

    print(
        idx,
        "->",
        id_to_phase[idx]
    )


# ==========================================================
# LOAD RESNET-50
# ==========================================================

print(
    "\nLoading ResNet-50..."
)


weights = (
    ResNet50_Weights.DEFAULT
)


resnet = resnet50(
    weights=weights
)


# Remove ImageNet
# classification layer
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


if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
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

def seconds_to_hms(
    seconds
):

    seconds = int(
        round(seconds)
    )


    hours = (
        seconds // 3600
    )


    minutes = (
        (seconds % 3600)
        // 60
    )


    secs = (
        seconds % 60
    )


    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ==========================================================
# EXTRACT FEATURES
# ==========================================================

def extract_features(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
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

        cap.release()

        raise RuntimeError(
            "Invalid FPS reported "
            "by the video."
        )


    duration_sec = (
        frame_count
        / original_fps
    )


    print(
        "\nVideo information"
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


    # Example:
    #
    # original video = 30 fps
    # sampling       = 1 fps
    #
    # select approximately
    # every 30 frames

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

                print(
                    "Could not read frame:",
                    frame_number
                )

                break


            # OpenCV is BGR
            # ResNet expects RGB

            frame_rgb = (
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
            )


            image = (
                Image.fromarray(
                    frame_rgb
                )
            )


            tensor = (
                image_preprocess(
                    image
                )
                .unsqueeze(0)
                .to(DEVICE)
            )


            feature = (
                resnet(
                    tensor
                )
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
            "No features were "
            "extracted."
        )


    features = (
        np.stack(
            features
        )
        .astype(
            np.float32
        )
    )


    print(
        "Feature shape:",
        features.shape
    )


    return (
        features,
        duration_sec
    )


# ==========================================================
# GET SEGMENTS
# ==========================================================

def get_segments(
    sequence
):

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


    # Final segment

    labels.append(
        int(
            current_label
        )
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

    sequence = (
        np.asarray(
            sequence
        )
        .copy()
    )


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
                    labels[
                        i + 1
                    ]
                )


            # Last segment

            elif (
                i
                == len(labels) - 1
            ):

                new_label = (
                    labels[
                        i - 1
                    ]
                )


            # Middle segment

            else:

                left_label = (
                    labels[
                        i - 1
                    ]
                )


                right_label = (
                    labels[
                        i + 1
                    ]
                )


                # A - short B - A

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


            # Recalculate segments
            # after every merge

            break


    return sequence


# ==========================================================
# MS-TCN PREDICTION
# ==========================================================

def predict_phases(
    features
):

    # features.npy format:
    #
    # [T, 2048]
    #
    # MS-TCN expects:
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


        # Use final MS-TCN stage

        final_output = (
            outputs[-1]
        )


        predictions = (
            final_output
            .argmax(
                dim=1
            )
            .squeeze(
                0
            )
            .cpu()
            .numpy()
        )


    return predictions


# ==========================================================
# PREDICTIONS -> INTERVALS
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


        segment = {

            "phase_id":
                int(
                    label
                ),

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


        segments.append(
            segment
        )


    return segments


# ==========================================================
# SAVE TO MONGODB
# ==========================================================

def save_to_mongodb(
    result
):

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )


    try:

        # Verify connection
        client.admin.command(
            "ping"
        )


        db = client[
            MONGO_DATABASE
        ]


        collection = db[
            MONGO_COLLECTION
        ]


        video_id = (
            result[
                "video_id"
            ]
        )


        # Update existing video
        # or insert if new

        collection.replace_one(
            {
                "video_id":
                    video_id
            },
            result,
            upsert=True
        )


        print(
            "\nMongoDB:"
        )

        print(
            "Database:",
            MONGO_DATABASE
        )

        print(
            "Collection:",
            MONGO_COLLECTION
        )

        print(
            "Video ID:",
            video_id
        )


    finally:

        client.close()


# ==========================================================
# PROCESS ONE VIDEO
# ==========================================================

def process_video(
    video_path
):

    print(
        "\n======================================"
    )

    print(
        "PROCESSING"
    )

    print(
        video_path.name
    )

    print(
        "======================================"
    )


    # ------------------------------------------------------
    # 1. Feature extraction
    # ------------------------------------------------------

    (
        features,
        real_duration_sec
    ) = extract_features(
        video_path
    )


    # ------------------------------------------------------
    # 2. Phase prediction
    # ------------------------------------------------------

    predictions = (
        predict_phases(
            features
        )
    )


    # ------------------------------------------------------
    # 3. Temporal intervals
    # ------------------------------------------------------

    segments = (
        predictions_to_intervals(
            predictions
        )
    )


    # ------------------------------------------------------
    # 4. Display result
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
    # 5. Build result document
    # ------------------------------------------------------

    video_id = (
        video_path.stem
    )


    result = {

        "video_id":
            video_id,

        "source_video":
            video_path.name,

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
                    len(
                        predictions
                    )
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
            len(
                segments
            ),

        "segments":
            segments
    }


    # ------------------------------------------------------
    # 6. Save JSON
    # ------------------------------------------------------

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


    print(
        "\nJSON saved:"
    )

    print(
        output_path.resolve()
    )


    # ------------------------------------------------------
    # 7. Save MongoDB
    # ------------------------------------------------------

    save_to_mongodb(
        result
    )


    print(
        "\nVideo completed:"
    )

    print(
        video_path.name
    )


    return result


# ==========================================================
# FIND ALL VIDEOS
# ==========================================================

def find_videos():

    if not VIDEO_FOLDER.exists():

        raise FileNotFoundError(
            f"Video folder does not exist:\n"
            f"{VIDEO_FOLDER}"
        )


    videos = sorted(
        [
            path

            for path
            in VIDEO_FOLDER.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]
    )


    return videos


# ==========================================================
# MAIN
# ==========================================================

def main():

    video_paths = (
        find_videos()
    )


    if len(video_paths) == 0:

        raise RuntimeError(
            f"No supported video files "
            f"found in:\n"
            f"{VIDEO_FOLDER}"
        )


    print(
        "\n======================================"
    )

    print(
        "BATCH SURGICAL PHASE DETECTION"
    )

    print(
        "======================================"
    )


    print(
        "Input folder:"
    )

    print(
        VIDEO_FOLDER
    )


    print(
        "\nVideos found:",
        len(
            video_paths
        )
    )


    print(
        "\nFiles:"
    )


    for (
        index,
        video_path
    ) in enumerate(
        video_paths,
        start=1
    ):

        print(
            f"{index:3d}. "
            f"{video_path.name}"
        )


    # ======================================================
    # PROCESS ALL VIDEOS
    # ======================================================

    successful = 0
    failed = 0

    failed_videos = []


    for (
        index,
        video_path
    ) in enumerate(
        video_paths,
        start=1
    ):

        print(
            "\n\n######################################"
        )

        print(
            f"VIDEO "
            f"{index}/"
            f"{len(video_paths)}"
        )

        print(
            video_path.name
        )

        print(
            "######################################"
        )


        try:

            process_video(
                video_path
            )


            successful += 1


        except Exception as error:

            failed += 1


            failed_videos.append(
                {
                    "video":
                        video_path.name,

                    "error":
                        str(error)
                }
            )


            print(
                "\nERROR"
            )

            print(
                "Video:",
                video_path.name
            )

            print(
                "Reason:",
                error
            )


            print(
                "Continuing to "
                "next video..."
            )


            continue


    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print(
        "\n\n======================================"
    )

    print(
        "BATCH FINISHED"
    )

    print(
        "======================================"
    )


    print(
        "Total videos:",
        len(
            video_paths
        )
    )


    print(
        "Successful:",
        successful
    )


    print(
        "Failed:",
        failed
    )


    print(
        "\nJSON output:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


    print(
        "\nMongoDB database:",
        MONGO_DATABASE
    )


    print(
        "MongoDB collection:",
        MONGO_COLLECTION
    )


    # ------------------------------------------------------
    # Show failures
    # ------------------------------------------------------

    if failed_videos:

        print(
            "\nFailed videos:"
        )


        for item in failed_videos:

            print(
                "-",
                item[
                    "video"
                ]
            )

            print(
                " ",
                item[
                    "error"
                ]
            )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()