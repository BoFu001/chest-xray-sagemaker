"""Step 1: download the Chest X-Ray pneumonia dataset from Kaggle into local data/.

Run: python 01_download_data.py
Requires ~/.kaggle/kaggle.json to be in place.
"""
import os
from config import LOCAL_DATA_DIR

DATASET = "paultimothymooney/chest-xray-pneumonia"


def main():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    # Authenticate: automatically reads ~/.kaggle/kaggle.json
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} -> {LOCAL_DATA_DIR}/  (~1.2GB, downloads and unzips, please wait)...")
    api.dataset_download_files(DATASET, path=LOCAL_DATA_DIR, unzip=True)

    print("\nDone. Directory structure:")
    for root, dirs, files in os.walk(LOCAL_DATA_DIR):
        depth = root.replace(LOCAL_DATA_DIR, "").count(os.sep)
        if depth <= 3:
            print("  " * depth + f"{os.path.basename(root) or LOCAL_DATA_DIR}/  ({len(files)} files)")


if __name__ == "__main__":
    main()
