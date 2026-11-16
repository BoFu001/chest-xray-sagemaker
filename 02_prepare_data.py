"""Step 2: re-split the dataset (stratified by class).

The original val set has only 16 images, too small. We merge original train + original
val and carve out a stratified 15% validation set; test (624 images) is held out and
untouched.

Output: data/prepared/{train,val,test}/{NORMAL,PNEUMONIA}
Run: python 02_prepare_data.py
"""
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from config import LOCAL_DATA_DIR, CLASS_NAMES

SRC = Path(LOCAL_DATA_DIR) / "chest_xray"   # raw data after download/unzip
DST = Path(LOCAL_DATA_DIR) / "prepared"     # re-split output
VAL_RATIO = 0.15
SEED = 42
IMG_EXT = {".jpeg", ".jpg", ".png"}


def collect(split, cls):
    d = SRC / split / cls
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


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
        pool = collect("train", cls) + collect("val", cls)
        tr, va = train_test_split(pool, test_size=VAL_RATIO, random_state=SEED)
        copy_files(tr, "train", cls)
        copy_files(va, "val", cls)
        te = collect("test", cls)
        copy_files(te, "test", cls)
        summary[cls] = (len(tr), len(va), len(te))

    print(f"Split complete (seed={SEED}, val ratio={VAL_RATIO}):\n")
    print(f"  {'Class':<12}{'train':>8}{'val':>8}{'test':>8}")
    for cls, (a, b, c) in summary.items():
        print(f"  {cls:<12}{a:>8}{b:>8}{c:>8}")
    print(f"\nOutput: {DST}/")


if __name__ == "__main__":
    main()
