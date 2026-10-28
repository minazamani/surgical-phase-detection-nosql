from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EXCEL_PATH = "Surgeryphasesdistribution_DataClassification.xlsx"
SHEET_NAME = 0                   # first sheet, or put sheet name here
OUTPUT_DIR = Path("processed")

FPS = 1                          # temporal sampling rate for labels

IGNORE_LABEL = "IGNORE"
OUTSIDE_PATIENT_LABEL = "Outside patient"

# Exact values expected in the "Surgery tags" column
UNKNOWN_BREAK_TAG = "Unknown break"
OUTSIDE_PATIENT_TAG = "Outside patient"

# Dataset split by SURGERY, never by fragment
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    """
    Convert Excel cell to clean string.
    Empty / NaN -> None.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


def time_to_seconds(value):
    """
    Convert an Excel time value to seconds.

    Supports:
        00:12:43
        0:12:43
        pandas Timedelta
        datetime.time
        numeric Excel fractions of a day
    """

    if pd.isna(value):
        return None

    # pandas Timedelta
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()

    # datetime.time
    if hasattr(value, "hour") and hasattr(value, "minute"):
        hour = getattr(value, "hour", 0)
        minute = getattr(value, "minute", 0)
        second = getattr(value, "second", 0)
        microsecond = getattr(value, "microsecond", 0)

        return (
            hour * 3600
            + minute * 60
            + second
            + microsecond / 1_000_000
        )

    # Excel may store time as fraction of one day
    if isinstance(value, (int, float, np.number)):
        return float(value) * 24 * 3600

    # String
    text = str(value).strip()

    parts = text.split(":")

    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])

        return minutes * 60 + seconds

    raise ValueError(f"Cannot parse time value: {value}")


def normalize_dataframe(df):
    """
    Keep and clean columns needed for the ML dataset.
    """

    required_columns = [
        "Surgery",
        "Video /Fragment",
        "Start Time",
        "End Time",
        "Surgery Phases",
        "Surgery tags",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in Excel: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    df["Surgery"] = df["Surgery"].apply(clean_text)
    df["Video /Fragment"] = df["Video /Fragment"].apply(clean_text)
    df["Surgery Phases"] = df["Surgery Phases"].apply(clean_text)
    df["Surgery tags"] = df["Surgery tags"].apply(clean_text)

    # Remove rows without surgery or fragment IDs
    df = df[
        df["Surgery"].notna()
        & df["Video /Fragment"].notna()
    ].copy()

    df["start_sec"] = df["Start Time"].apply(time_to_seconds)
    df["end_sec"] = df["End Time"].apply(time_to_seconds)

    return df


# ============================================================
# DISCOVER PHASES / TAGS
# ============================================================

def build_label_maps(df):
    """
    Build phase and tag dictionaries automatically from Excel.

    This means when U58, U59, etc. are added later,
    new phase names are automatically discovered.
    """

    phases = set()
    tags = set()

    for _, row in df.iterrows():

        phase = row["Surgery Phases"]
        tag = row["Surgery tags"]

        if phase:
            phases.add(phase)

        if tag:
            tags.add(tag)

    # Outside patient is treated as a phase-like visual class
    phases.add(OUTSIDE_PATIENT_LABEL)

    # IGNORE is special
    # We use -100 because PyTorch CrossEntropyLoss can ignore it.
    phase_to_id = {
        phase: i
        for i, phase in enumerate(sorted(phases))
    }

    phase_to_id[IGNORE_LABEL] = -100

    tag_to_id = {
        tag: i + 1
        for i, tag in enumerate(sorted(tags))
    }

    # 0 = no tag
    tag_to_id["NO_TAG"] = 0

    return phase_to_id, tag_to_id


# ============================================================
# CREATE LABEL SEQUENCE FOR ONE VIDEO FRAGMENT
# ============================================================

def create_fragment_labels(
    fragment_df,
    phase_to_id,
    tag_to_id,
    sampling_fps=1,
):
    """
    Generate one phase label and one tag label per sampled time point.

    Example at 1 fps:

        second 0 -> Hilar dissection
        second 1 -> Hilar dissection
        ...
    """

    max_end_sec = fragment_df["end_sec"].max()

    number_of_samples = int(np.ceil(max_end_sec * sampling_fps))

    # Default:
    # phase = IGNORE
    # tag = NO_TAG

    phase_labels = np.full(
        number_of_samples,
        phase_to_id[IGNORE_LABEL],
        dtype=np.int64,
    )

    tag_labels = np.full(
        number_of_samples,
        tag_to_id["NO_TAG"],
        dtype=np.int64,
    )

    timeline_phase_name = np.full(
        number_of_samples,
        IGNORE_LABEL,
        dtype=object,
    )

    timeline_tag_name = np.full(
        number_of_samples,
        "NO_TAG",
        dtype=object,
    )

    # --------------------------------------------------------
    # Fill intervals
    # --------------------------------------------------------

    for _, row in fragment_df.iterrows():

        start_sec = row["start_sec"]
        end_sec = row["end_sec"]

        phase = row["Surgery Phases"]
        tag = row["Surgery tags"]

        if start_sec is None or end_sec is None:
            continue

        start_idx = int(np.floor(start_sec * sampling_fps))
        end_idx = int(np.ceil(end_sec * sampling_fps))

        start_idx = max(0, start_idx)
        end_idx = min(number_of_samples, end_idx)

        if end_idx <= start_idx:
            continue

        # ====================================================
        # PHASE LOGIC
        # ====================================================

        # Case 1:
        # Real surgical phase exists
        if phase is not None:

            phase_id = phase_to_id[phase]

            phase_labels[start_idx:end_idx] = phase_id
            timeline_phase_name[start_idx:end_idx] = phase

        # Case 2:
        # No phase but camera is outside patient
        elif tag == OUTSIDE_PATIENT_TAG:

            phase_id = phase_to_id[OUTSIDE_PATIENT_LABEL]

            phase_labels[start_idx:end_idx] = phase_id

            timeline_phase_name[start_idx:end_idx] = (
                OUTSIDE_PATIENT_LABEL
            )

        # Case 3:
        # Unknown break
        #
        # Do NOT teach the model that "Unknown break"
        # is a surgical phase.
        elif tag == UNKNOWN_BREAK_TAG:

            phase_labels[start_idx:end_idx] = (
                phase_to_id[IGNORE_LABEL]
            )

            timeline_phase_name[start_idx:end_idx] = (
                IGNORE_LABEL
            )

        # Case 4:
        # no phase information
        else:

            phase_labels[start_idx:end_idx] = (
                phase_to_id[IGNORE_LABEL]
            )

            timeline_phase_name[start_idx:end_idx] = (
                IGNORE_LABEL
            )

        # ====================================================
        # TAG LOGIC
        # ====================================================

        if tag is not None:

            if tag not in tag_to_id:
                raise ValueError(
                    f"Unknown tag found: {tag}"
                )

            tag_labels[start_idx:end_idx] = tag_to_id[tag]
            timeline_tag_name[start_idx:end_idx] = tag

    return (
        phase_labels,
        tag_labels,
        timeline_phase_name,
        timeline_tag_name,
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_surgery_splits(df):
    """
    Split by complete surgery.

    All fragments belonging to U57 stay in the same split.
    """

    surgeries = sorted(df["Surgery"].unique())

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(surgeries)

    n = len(surgeries)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = surgeries[:n_train]

    val = surgeries[n_train:n_train + n_val]

    test = surgeries[n_train + n_val:]

    split_map = {}

    for surgery in train:
        split_map[surgery] = "train"

    for surgery in val:
        split_map[surgery] = "validation"

    for surgery in test:
        split_map[surgery] = "test"

    return split_map


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Read Excel
    # --------------------------------------------------------

    print("Reading Excel...")

    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name=SHEET_NAME,
    )
    for i, value in df["Start Time"].items():
        if "day" in str(value):
            print(
                "Suspicious Start Time:",
                i,
                value
            )

    for i, value in df["End Time"].items():
        if "day" in str(value):
            print(
                "Suspicious End Time:",
                i,
                value
            )
    df = normalize_dataframe(df)

    print(f"Rows loaded: {len(df)}")

    print(
        f"Surgeries found: "
        f"{df['Surgery'].nunique()}"
    )

    # --------------------------------------------------------
    # 2. Discover labels
    # --------------------------------------------------------

    phase_to_id, tag_to_id = build_label_maps(df)

    print("\nPhase labels:")
    for phase, idx in phase_to_id.items():
        print(f"  {idx:>4}: {phase}")

    print("\nTag labels:")
    for tag, idx in tag_to_id.items():
        print(f"  {idx:>4}: {tag}")

    # Save dictionaries

    with open(
        OUTPUT_DIR / "label_maps.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "phase_to_id": phase_to_id,
                "tag_to_id": tag_to_id,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 3. Surgery-level train / validation / test split
    # --------------------------------------------------------

    split_map = create_surgery_splits(df)

    split_df = pd.DataFrame(
        [
            {
                "surgery": surgery,
                "split": split_name,
            }
            for surgery, split_name in split_map.items()
        ]
    )

    split_df.to_csv(
        OUTPUT_DIR / "splits.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Process each surgery + fragment
    # --------------------------------------------------------

    manifest_rows = []

    grouped = df.groupby(
        ["Surgery", "Video /Fragment"],
        sort=False,
    )

    for (surgery, fragment), fragment_df in grouped:

        surgery = str(surgery)

        # Clean fragment ID
        fragment = str(fragment)

        # Excel sometimes gives 1.0 instead of 1
        if fragment.endswith(".0"):
            fragment = fragment[:-2]

        video_id = f"{surgery}_{fragment}"

        print(f"\nProcessing {video_id}")

        fragment_output_dir = (
            OUTPUT_DIR / video_id
        )

        fragment_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            phase_labels,
            tag_labels,
            phase_names,
            tag_names,
        ) = create_fragment_labels(
            fragment_df=fragment_df,
            phase_to_id=phase_to_id,
            tag_to_id=tag_to_id,
            sampling_fps=FPS,
        )

        # ----------------------------------------------------
        # Save NumPy arrays
        # ----------------------------------------------------

        np.save(
            fragment_output_dir / "phase_labels.npy",
            phase_labels,
        )

        np.save(
            fragment_output_dir / "tag_labels.npy",
            tag_labels,
        )

        # ----------------------------------------------------
        # Human-readable timeline
        # ----------------------------------------------------

        timeline = pd.DataFrame(
            {
                "sample_index": np.arange(
                    len(phase_labels)
                ),

                "time_sec": (
                    np.arange(len(phase_labels))
                    / FPS
                ),

                "phase_id": phase_labels,

                "phase": phase_names,

                "tag_id": tag_labels,

                "tag": tag_names,
            }
        )

        timeline.to_csv(
            fragment_output_dir / "timeline.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        duration_sec = (
            len(phase_labels) / FPS
        )

        number_valid_labels = int(
            np.sum(
                phase_labels
                != phase_to_id[IGNORE_LABEL]
            )
        )

        manifest_rows.append(
            {
                "video_id": video_id,
                "surgery": surgery,
                "fragment": fragment,
                "duration_sec": duration_sec,
                "number_samples": len(phase_labels),
                "valid_phase_samples": number_valid_labels,
                "split": split_map[surgery],
                "phase_labels_path": str(
                    fragment_output_dir
                    / "phase_labels.npy"
                ),
                "tag_labels_path": str(
                    fragment_output_dir
                    / "tag_labels.npy"
                ),
            }
        )

    # --------------------------------------------------------
    # 5. Save global manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(manifest_rows)

    manifest.to_csv(
        OUTPUT_DIR / "manifest.csv",
        index=False,
    )

    print("\n======================================")
    print("Finished.")
    print("======================================")

    print(
        f"Processed fragments: {len(manifest)}"
    )

    print("\nDataset splits:")

    if not manifest.empty:
        print(
            manifest.groupby("split")[
                "surgery"
            ].nunique()
        )

    print(
        f"\nOutput directory: "
        f"{OUTPUT_DIR.resolve()}"
    )



if __name__ == "__main__":
    main()