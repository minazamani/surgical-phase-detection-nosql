import json
from pathlib import Path

import numpy as np
import torch

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
    "predicted_intervals"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Model configuration
NUM_STAGES = 4
NUM_LAYERS = 10
NUM_F_MAPS = 64
FEATURE_DIM = 2048

# Your features were extracted at 1 frame / second
SAMPLING_RATE_FPS = 1

# Selected using validation set
MIN_SEGMENT_DURATION = 15


print("Device:", DEVICE)


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

    label_maps = json.load(f)


phase_to_id = (
    label_maps["phase_to_id"]
)


# Ignore special negative IDs such as -100
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
    "Number of phases:",
    NUM_CLASSES
)


for idx in sorted(id_to_phase):

    print(
        idx,
        "->",
        id_to_phase[idx]
    )


# ==========================================================
# LOAD MODEL
# ==========================================================

model = MultiStageModel(
    num_stages=NUM_STAGES,
    num_layers=NUM_LAYERS,
    num_f_maps=NUM_F_MAPS,
    input_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
)


if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_PATH.resolve()}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    DEVICE
)

model.eval()


print(
    "\nLoaded model:"
)

print(
    CHECKPOINT_PATH.resolve()
)

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
# TIME CONVERSION
# ==========================================================

def seconds_to_hms(seconds):
    """
    Convert seconds to HH:MM:SS.

    Example:
        1041 -> 00:17:21
    """

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
# SEGMENT EXTRACTION
# ==========================================================

def get_segments(sequence):
    """
    Convert a second-by-second sequence:

        [1,1,1,2,2,3,3]

    into:

        labels = [1,2,3]
        starts = [0,3,5]
        ends   = [3,5,7]

    End indices are exclusive.
    """

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
                int(current_label)
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
    Remove very short predicted segments.

    Rules:

    1. If a short segment is between two segments
       having the same phase, merge it into them.

       A A A | B | A A A
              ↓
       A A A A A A A

    2. Otherwise merge the short segment into the
       longer neighboring segment.

    3. At the start/end of the video, merge it into
       the only neighboring segment.

    min_duration is expressed in sampled timesteps.

    Since this project uses 1 fps:
        15 timesteps = 15 seconds.
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


            # ------------------------------------------
            # First segment
            # ------------------------------------------

            if i == 0:

                new_label = (
                    labels[i + 1]
                )


            # ------------------------------------------
            # Last segment
            # ------------------------------------------

            elif (
                i
                == len(labels) - 1
            ):

                new_label = (
                    labels[i - 1]
                )


            # ------------------------------------------
            # Middle segment
            # ------------------------------------------

            else:

                left_label = (
                    labels[i - 1]
                )

                right_label = (
                    labels[i + 1]
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

            # Recompute segments after each merge
            break


    return sequence


# ==========================================================
# CONVERT MODEL TIMESTEPS TO SECONDS
# ==========================================================

def timestep_to_seconds(
    timestep
):
    """
    Convert model timestep to seconds.

    At 1 fps:
        timestep 100 -> 100 sec

    Keeping this function makes the code safe if
    sampling frequency changes later.
    """

    return (
        timestep
        / SAMPLING_RATE_FPS
    )


# ==========================================================
# PREDICT ONE VIDEO
# ==========================================================

def predict_video(
    video_id
):

    folder = (
        PROCESSED_DIR
        / video_id
    )


    features_path = (
        folder
        / "features.npy"
    )


    if not features_path.exists():

        raise FileNotFoundError(
            f"features.npy not found "
            f"for {video_id}"
        )


    # ------------------------------------------------------
    # LOAD FEATURES
    # ------------------------------------------------------

    features = np.load(
        features_path
    ).astype(
        np.float32
    )


    if features.ndim != 2:

        raise ValueError(
            f"{video_id}: expected "
            f"2D features, got "
            f"{features.shape}"
        )


    if (
        features.shape[1]
        != FEATURE_DIM
    ):

        raise ValueError(
            f"{video_id}: expected "
            f"[T, {FEATURE_DIM}], "
            f"got {features.shape}"
        )


    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            f"{video_id}: features "
            f"contain NaN or infinity"
        )


    number_timesteps = (
        features.shape[0]
    )


    print(
        f"\nProcessing: "
        f"{video_id}"
    )

    print(
        "Feature shape:",
        features.shape
    )


    # ------------------------------------------------------
    # PREPARE MODEL INPUT
    # ------------------------------------------------------

    # features.npy:
    #
    # [T, 2048]
    #
    # MS-TCN input:
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


    # For an unseen video all timesteps
    # are considered valid.
    mask = torch.ones(
        1,
        1,
        number_timesteps,
        device=DEVICE,
        dtype=torch.float32
    )


    # ------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            features_tensor,
            mask
        )


        # outputs:
        #
        # [num_stages, B, C, T]
        #
        # Take final MS-TCN stage.

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


    # ------------------------------------------------------
    # TEMPORAL POST-PROCESSING
    # ------------------------------------------------------

    predictions = (
        remove_short_segments(
            predictions,
            MIN_SEGMENT_DURATION
        )
    )


    # ------------------------------------------------------
    # CREATE INTERVALS
    # ------------------------------------------------------

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

        # ----------------------------------------------
        # Convert sampled timesteps to actual seconds
        # ----------------------------------------------

        start_sec = (
            timestep_to_seconds(
                start
            )
        )

        end_sec = (
            timestep_to_seconds(
                end
            )
        )

        duration_sec = (
            end_sec
            - start_sec
        )


        segment = {

            "phase_id":
                int(label),

            "phase":
                id_to_phase[
                    int(label)
                ],

            # Machine-friendly values
            "start_sec":
                float(start_sec),

            "end_sec":
                float(end_sec),

            "duration_sec":
                float(duration_sec),

            # Human-friendly values
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


    return {
        "number_timesteps":
            number_timesteps,

        "segments":
            segments
    }


# ==========================================================
# SAVE ONE VIDEO RESULT
# ==========================================================

def save_prediction(
    video_id,
    prediction_result
):

    number_timesteps = (
        prediction_result[
            "number_timesteps"
        ]
    )

    segments = (
        prediction_result[
            "segments"
        ]
    )


    video_duration_sec = (
        number_timesteps
        / SAMPLING_RATE_FPS
    )


    result = {

        "video_id":
            video_id,

        "model": {
            "visual_features":
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
                number_timesteps,
        },

        "video_duration": {
            "seconds":
                float(
                    video_duration_sec
                ),

            "time":
                seconds_to_hms(
                    video_duration_sec
                ),
        },

        "postprocessing": {
            "minimum_segment_duration_sec":
                MIN_SEGMENT_DURATION,

            "selection_method":
                "selected on validation set"
        },

        "number_segments":
            len(segments),

        "segments":
            segments,
    }


    output_path = (
        OUTPUT_DIR
        / f"{video_id}.json"
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


    return output_path


# ==========================================================
# FIND ALL VIDEOS
# ==========================================================

def find_video_folders():

    video_folders = []


    for folder in sorted(
        PROCESSED_DIR.iterdir()
    ):

        if not folder.is_dir():
            continue


        if not (
            folder
            / "features.npy"
        ).exists():

            continue


        video_folders.append(
            folder
        )


    return video_folders


# ==========================================================
# MAIN
# ==========================================================

def main():

    video_folders = (
        find_video_folders()
    )


    if len(video_folders) == 0:

        raise RuntimeError(
            "No video folders containing "
            "features.npy were found in:\n"
            f"{PROCESSED_DIR.resolve()}"
        )


    print(
        "\n======================================"
    )

    print(
        "PHASE INTERVAL PREDICTION"
    )

    print(
        "======================================"
    )


    print(
        "Videos found:",
        len(video_folders)
    )

    print(
        "Minimum segment duration:",
        MIN_SEGMENT_DURATION,
        "seconds"
    )


    successful = 0
    failed = 0

    total_segments = 0


    for folder in video_folders:

        video_id = (
            folder.name
        )


        try:

            prediction_result = (
                predict_video(
                    video_id
                )
            )


            output_path = (
                save_prediction(
                    video_id,
                    prediction_result
                )
            )


            segments = (
                prediction_result[
                    "segments"
                ]
            )


            print(
                f"✓ {video_id}"
            )

            print(
                "  Segments:",
                len(segments)
            )


            # Display intervals
            for segment in segments:

                print(
                    " ",
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


            print(
                "  Saved:",
                output_path
            )


            successful += 1

            total_segments += (
                len(segments)
            )


        except Exception as error:

            print(
                f"\n✗ ERROR: "
                f"{video_id}"
            )

            print(
                " ",
                error
            )


            failed += 1


    # ======================================================
    # SUMMARY
    # ======================================================

    print(
        "\n======================================"
    )

    print(
        "FINISHED"
    )

    print(
        "======================================"
    )


    print(
        "Videos processed successfully:",
        successful
    )

    print(
        "Videos failed:",
        failed
    )

    print(
        "Total predicted segments:",
        total_segments
    )

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()