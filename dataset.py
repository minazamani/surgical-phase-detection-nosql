from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SurgicalPhaseDataset(Dataset):

    def __init__(
        self,
        processed_dir,
        split="train",
    ):
        """
        processed_dir should contain:

        processed/
        ├── splits.csv
        ├── label_maps.json
        ├── U57L_1_raw/
        │   ├── features.npy
        │   └── phase_labels_aligned.npy
        ├── U57L_2_raw/
        └── ...

        split:
            "train"
            "validation"
            "test"
        """

        self.processed_dir = Path(processed_dir)
        self.split = split

        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed directory does not exist:\n"
                f"{self.processed_dir}"
            )

        # --------------------------------------------------
        # Read split file
        # --------------------------------------------------

        split_file = self.processed_dir / "splits.csv"

        if not split_file.exists():
            raise FileNotFoundError(
                f"splits.csv not found:\n{split_file}"
            )

        splits = pd.read_csv(split_file)

        splits["surgery"] = (
            splits["surgery"]
            .astype(str)
            .str.strip()
        )

        splits["split"] = (
            splits["split"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        split = split.lower()

        if split == "val":
            split = "validation"

        self.surgeries = set(
            splits.loc[
                splits["split"] == split,
                "surgery"
            ]
        )

        if len(self.surgeries) == 0:
            raise RuntimeError(
                f"No surgeries found for split '{split}'"
            )

        # --------------------------------------------------
        # Find valid video folders
        # --------------------------------------------------

        self.video_ids = []

        for folder in sorted(self.processed_dir.iterdir()):

            if not folder.is_dir():
                continue

            video_id = folder.name
            surgery_id = self._extract_surgery_id(video_id)

            if surgery_id not in self.surgeries:
                continue

            features_path = folder / "features.npy"
            labels_path = folder / "phase_labels_aligned.npy"

            if not features_path.exists():
                print(
                    f"Skipping {video_id}: "
                    f"features.npy missing"
                )
                continue

            if not labels_path.exists():
                print(
                    f"Skipping {video_id}: "
                    f"phase_labels_aligned.npy missing"
                )
                continue

            # Validate before accepting
            is_valid = self._validate_sample(
                video_id,
                features_path,
                labels_path
            )

            if not is_valid:
                print(
                    f"Skipping {video_id}: "
                    f"no valid phase labels"
                )
                continue

            self.video_ids.append(video_id)

        if len(self.video_ids) == 0:
            raise RuntimeError(
                f"No usable videos found for split '{split}'"
            )

        print(
            f"{split}: "
            f"{len(self.video_ids)} video fragments "
            f"from {len(self.surgeries)} surgeries"
        )

    @staticmethod
    def _extract_surgery_id(video_id):
        """
        Examples:

        U57L_1_raw -> U57
        U79L_4_raw -> U79
        U63L_5_raw -> U63
        """

        match = re.match(
            r"(U\d+)",
            video_id
        )

        if match is None:
            raise ValueError(
                f"Cannot extract surgery ID from "
                f"folder name: {video_id}"
            )

        return match.group(1)

    @staticmethod
    def _validate_sample(
        video_id,
        features_path,
        labels_path
    ):
        """
        Verify that the sample is safe to use.
        """

        features = np.load(
            features_path,
            mmap_mode="r"
        )

        labels = np.load(
            labels_path,
            mmap_mode="r"
        )

        # ----------------------------------------------
        # Feature shape
        # ----------------------------------------------

        if features.ndim != 2:
            raise ValueError(
                f"{video_id}: features must be 2D, "
                f"got {features.shape}"
            )

        if features.shape[1] != 2048:
            raise ValueError(
                f"{video_id}: expected features "
                f"[T, 2048], got {features.shape}"
            )

        # ----------------------------------------------
        # Label shape
        # ----------------------------------------------

        if labels.ndim != 1:
            raise ValueError(
                f"{video_id}: labels must be 1D, "
                f"got {labels.shape}"
            )

        # ----------------------------------------------
        # Temporal alignment
        # ----------------------------------------------

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{video_id}: temporal mismatch:\n"
                f"features = {features.shape}\n"
                f"labels   = {labels.shape}"
            )

        # ----------------------------------------------
        # NaN / infinity check
        # ----------------------------------------------

        if not np.isfinite(features).all():
            raise ValueError(
                f"{video_id}: features contain "
                f"NaN or infinity"
            )

        # ----------------------------------------------
        # At least one usable label
        # ----------------------------------------------

        valid_labels = labels[labels != -100]

        if len(valid_labels) == 0:
            return False
        return True

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):

        video_id = self.video_ids[index]

        folder = self.processed_dir / video_id

        features = np.load(
            folder / "features.npy"
        ).astype(np.float32)

        labels = np.load(
            folder / "phase_labels_aligned.npy"
        ).astype(np.int64)

        # Current:
        # features = [T, 2048]
        #
        # MS-TCN wants:
        # features = [2048, T]

        features = torch.from_numpy(
            features
        ).transpose(0, 1)

        labels = torch.from_numpy(
            labels
        )

        return {
            "video_id": video_id,
            "surgery_id": self._extract_surgery_id(
                video_id
            ),
            "features": features,
            "labels": labels,
        }