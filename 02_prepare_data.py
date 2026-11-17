"""Step 2: organise the raw cats-vs-dogs images and split into train/val/test.

Kaggle's train set is 25,000 images in ONE folder, labelled only by filename
(cat.0.jpg / dog.3.jpg); the competition test set is unlabelled, so it is unusable.
Here we sort images into cats/ and dogs/ by filename, take PER_CLASS of each, then make
a stratified 70/15/15 train/val/test split (reproducible, seed=42).

Output: data/prepared/{train,val,test}/{cats,dogs}
Run: python 02_prepare_data.py
"""
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from config import LOCAL_DATA_DIR, CLASS_NAMES

SRC = Path(LOCAL_DATA_DIR) / "train"        # raw unzipped images (cat.N.jpg / dog.N.jpg)
DST = Path(LOCAL_DATA_DIR) / "prepared"     # re-split output
PER_CLASS = 10000                           # images per class (matches the earlier notebook)
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

PREFIX = {"cats": "cat", "dogs": "dog"}     # class name -> filename prefix


def collect(cls):
    """All raw images of one class, numerically sorted (cat.2 before cat.10), capped at PER_CLASS."""
    files = sorted(SRC.glob(f"{PREFIX[cls]}.*.jpg"), key=lambda p: int(p.stem.split(".")[1]))
    return files[:PER_CLASS]


def copy_files(files, split, cls):
    out = DST / split / cls
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out / f.name)


def main():
    if DST.exists():
        shutil.rmtree(DST)

    summary = {}
    for cls in CLASS_NAMES:
        pool = collect(cls)
        # 70/15/15: split off test first, then split the remainder into train / val
        trainval, te = train_test_split(pool, test_size=TEST_RATIO, random_state=SEED)
        tr, va = train_test_split(trainval, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=SEED)
        copy_files(tr, "train", cls)
        copy_files(va, "val", cls)
        copy_files(te, "test", cls)
        summary[cls] = (len(tr), len(va), len(te))

    print(f"Split complete (seed={SEED}, 70/15/15):\n")
    print(f"  {'Class':<8}{'train':>8}{'val':>8}{'test':>8}")
    for cls, (a, b, c) in summary.items():
        print(f"  {cls:<8}{a:>8}{b:>8}{c:>8}")
    print(f"\nOutput: {DST}/")


if __name__ == "__main__":
    main()
