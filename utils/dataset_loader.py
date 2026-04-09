"""
UrbanSound8K dataset loader for the Hazard Sound Detection project.

Expected folder structure
─────────────────────────
dataset/
└── UrbanSound8K/
    ├── audio/
    │   ├── fold1/
    │   │   └── *.wav
    │   ├── fold2/ … fold10/
    └── metadata/
        └── UrbanSound8K.csv

Download the dataset from https://urbansounddataset.weebly.com/urbansound8k.html
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .feature_extraction import FeatureExtractor


# ── Class mappings ──────────────────────────────────────────────────────────
HAZARD_CLASSES = {
    "dog_bark": 3,
    "gun_shot": 6,
    "siren": 8,
}
ALL_CLASS_IDS = set(range(10))
NORMAL_CLASS_IDS = ALL_CLASS_IDS - set(HAZARD_CLASSES.values())

LABEL_NAMES = ["dog_bark", "gun_shot", "siren", "normal"]


def _dataset_root(base_dir: str = None) -> str:
    """Return the absolute path to the UrbanSound8K root directory."""
    if base_dir:
        return base_dir
    # Walk up from this file to find dataset/UrbanSound8K
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    return os.path.join(project_root, "dataset", "UrbanSound8K")


def _check_dataset(dataset_root: str) -> None:
    """Raise a descriptive error if the dataset is not found."""
    metadata_csv = os.path.join(dataset_root, "metadata", "UrbanSound8K.csv")
    audio_dir = os.path.join(dataset_root, "audio")
    if not os.path.isfile(metadata_csv) or not os.path.isdir(audio_dir):
        print(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ❌  UrbanSound8K dataset NOT found!\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "\n"
            "  1. Download from: https://urbansounddataset.weebly.com/urbansound8k.html\n"
            "  2. Extract the archive.\n"
            "  3. Place the folder so the project tree looks like:\n"
            "\n"
            "       <project-root>/\n"
            "       └── dataset/\n"
            "           └── UrbanSound8K/\n"
            "               ├── audio/\n"
            "               │   ├── fold1/\n"
            "               │   │   └── *.wav\n"
            "               │   └── … fold10/\n"
            "               └── metadata/\n"
            "                   └── UrbanSound8K.csv\n"
            "\n"
            "  Then re-run:  python train_model.py\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        sys.exit(1)


class DatasetLoader:
    """
    Loads UrbanSound8K, extracts features, and returns train/test splits.

    Parameters
    ----------
    dataset_root : str, optional
        Path to the UrbanSound8K root directory.  If *None* the default
        ``<project-root>/dataset/UrbanSound8K`` path is used.
    extractor : FeatureExtractor, optional
        Pre-configured feature extractor.  A default one is created when
        *None* is supplied.
    test_size : float
        Fraction of samples reserved for the test set (default 0.2).
    normal_ratio : float
        Fraction of non-hazard samples to include as the "normal" class
        (default 0.3 — keeps the class balanced with hazard classes).
    random_state : int
        Seed for reproducible train/test splits.
    """

    def __init__(
        self,
        dataset_root: str = None,
        extractor: FeatureExtractor = None,
        test_size: float = 0.2,
        normal_ratio: float = 0.3,
        random_state: int = 42,
    ):
        self.dataset_root = _dataset_root(dataset_root)
        self.extractor = extractor or FeatureExtractor()
        self.test_size = test_size
        self.normal_ratio = normal_ratio
        self.random_state = random_state

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABEL_NAMES)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _read_metadata(self) -> pd.DataFrame:
        _check_dataset(self.dataset_root)
        csv_path = os.path.join(self.dataset_root, "metadata", "UrbanSound8K.csv")
        df = pd.read_csv(csv_path)
        # Build full path column
        df["file_path"] = df.apply(
            lambda row: os.path.join(
                self.dataset_root, "audio", f"fold{row['fold']}", row["slice_file_name"]
            ),
            axis=1,
        )
        return df

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep hazard classes plus a balanced subset of normal samples."""
        hazard_ids = list(HAZARD_CLASSES.values())
        hazard_df = df[df["classID"].isin(hazard_ids)].copy()
        normal_df = df[~df["classID"].isin(hazard_ids)].copy()

        # Sample a fraction of normal sounds so the dataset stays balanced
        n_normal = max(1, int(len(normal_df) * self.normal_ratio))
        normal_df = normal_df.sample(n=n_normal, random_state=self.random_state)

        # Assign string labels via classID for reliable mapping
        hazard_df["label"] = hazard_df["classID"].map(
            {v: k for k, v in HAZARD_CLASSES.items()}
        )
        normal_df["label"] = "normal"

        combined = pd.concat([hazard_df, normal_df], ignore_index=True)
        # Drop rows whose audio file does not exist
        combined = combined[combined["file_path"].apply(os.path.isfile)]
        combined = combined.reset_index(drop=True)
        return combined

    def _extract_features(self, df: pd.DataFrame):
        """Extract features for every row; return (X, y) arrays."""
        features = []
        labels = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            try:
                feat = self.extractor.extract_from_file(row["file_path"])
                features.append(feat)
                labels.append(row["label"])
            except Exception as exc:  # noqa: BLE001
                # Skip files that cannot be read (corrupted, etc.)
                print(f"  [WARN] Skipping {row['file_path']}: {exc}")

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)
        return X, y

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self):
        """
        Load dataset, extract features, and return train/test splits.

        Returns
        -------
        X_train, X_test : np.ndarray  shape (N, n_mels, max_frames, 1)
        y_train, y_test : np.ndarray  integer-encoded labels
        label_names     : list[str]   ordered label names
        """
        print("📂  Reading UrbanSound8K metadata …")
        df = self._read_metadata()

        print("🔍  Filtering hazard + normal classes …")
        df = self._filter_rows(df)
        print(f"    Total samples selected: {len(df)}")
        print(df["label"].value_counts().to_string())

        print("\n🎵  Extracting audio features …")
        X, y_str = self._extract_features(df)

        y = self.label_encoder.transform(y_str)

        print(f"\n✅  Feature matrix shape : {X.shape}")
        print(f"    Labels (encoded)     : {np.unique(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test, LABEL_NAMES
