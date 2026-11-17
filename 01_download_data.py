"""Step 1: download the Kaggle 'Dogs vs. Cats' competition data into local data/.

Run: python 01_download_data.py
Requires ~/.kaggle/kaggle.json. You must also accept the competition rules once at
https://www.kaggle.com/c/dogs-vs-cats/rules — otherwise the download returns 403.
"""
import os
import zipfile
from config import LOCAL_DATA_DIR

COMPETITION = "dogs-vs-cats"


def main():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    # Authenticate: automatically reads ~/.kaggle/kaggle.json
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading competition '{COMPETITION}' -> {LOCAL_DATA_DIR}/  (~800MB, please wait)...")
    api.competition_download_files(COMPETITION, path=LOCAL_DATA_DIR, quiet=False)

    # Outer archive -> train.zip, test1.zip, sampleSubmission.csv
    outer = os.path.join(LOCAL_DATA_DIR, f"{COMPETITION}.zip")
    with zipfile.ZipFile(outer) as z:
        z.extractall(LOCAL_DATA_DIR)

    # Only the labelled train set is usable (test1 has no labels)
    train_zip = os.path.join(LOCAL_DATA_DIR, "train.zip")
    with zipfile.ZipFile(train_zip) as z:
        z.extractall(LOCAL_DATA_DIR)

    # Drop archives we no longer need, to save disk
    for f in [outer, train_zip, os.path.join(LOCAL_DATA_DIR, "test1.zip")]:
        if os.path.exists(f):
            os.remove(f)

    train_dir = os.path.join(LOCAL_DATA_DIR, "train")
    n = len(os.listdir(train_dir))
    print(f"\nDone. {n} images in {train_dir}/  (named cat.N.jpg / dog.N.jpg)")


if __name__ == "__main__":
    main()
