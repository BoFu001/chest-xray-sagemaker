"""Step 2: re-split the dataset.

The original val set has only 16 images, too small to evaluate reliably. Approach:
  - keep test as-is (final test set, 624 images, never touched during training)
  - merge original train + original val into one pool, carve out 15% as a
    stratified validation set
  - output to data/prepared/{train,val,test}/{NORMAL,PNEUMONIA}

Stratified = after splitting, the NORMAL:PNEUMONIA ratio in each subset matches
the original, so the validation set is representative.

Run: python 02_prepare_data.py
"""
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import LOCAL_DATA_DIR, CLASS_NAMES

SRC = Path(LOCAL_DATA_DIR) / "chest_xray"   # raw data after download/unzip
DST = Path(LOCAL_DATA_DIR) / "prepared"     # re-split output
VAL_RATIO = 0.15                            # carve 15% of train as validation
SEED = 42                                   # fixed seed -> reproducible split

IMG_EXT = {".jpeg", ".jpg", ".png"}


def collect(split, cls):
    """Collect all image paths under a given split/class."""
    d = SRC / split / cls
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def copy_files(files, split, cls):
    out = DST / split / cls
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out / f.name)


def main():
    if DST.exists():
        shutil.rmtree(DST)   # clear previous output so the run is repeatable

    summary = {}
    for cls in CLASS_NAMES:
        # merge original train + original val into one pool
        pool = collect("train", cls) + collect("val", cls)

        # split into train / val within each class (stratified by splitting per-class)
        tr, va = train_test_split(pool, test_size=VAL_RATIO, random_state=SEED)
        copy_files(tr, "train", cls)
        copy_files(va, "val", cls)

        # copy test as-is
        te = collect("test", cls)
        copy_files(te, "test", cls)

        summary[cls] = (len(tr), len(va), len(te))

    print(f"Split complete (seed={SEED}, val ratio={VAL_RATIO}):\n")
    print(f"  {'Class':<12}{'train':>8}{'val':>8}{'test':>8}")
    tot = [0, 0, 0]
    for cls, (a, b, c) in summary.items():
        print(f"  {cls:<12}{a:>8}{b:>8}{c:>8}")
        tot = [tot[0]+a, tot[1]+b, tot[2]+c]
    print(f"  {'Total':<12}{tot[0]:>8}{tot[1]:>8}{tot[2]:>8}")
    print(f"\nOutput directory: {DST}/")


if __name__ == "__main__":
    main()
