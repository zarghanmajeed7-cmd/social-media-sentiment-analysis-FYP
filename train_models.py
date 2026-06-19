# ============================================================
# train_models.py
# Run this ONCE before launching the dashboard:
#
#     python train_models.py
#
# What it does:
#   1. Downloads Sentiment140 dataset (~80 MB zip)
#   2. Trains Logistic Regression + Random Forest
#   3. Saves models to  models/
#   4. Prints accuracy + F1 score for both models
#
# Optional: use fewer rows for faster training:
#     python train_models.py --sample 50000
# ============================================================

import os
import sys
import urllib.request
import zipfile

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
DATASET_PATH  = os.path.join(DATA_DIR, "sentiment140.csv")
ZIP_PATH      = os.path.join(DATA_DIR, "_sentiment140.zip")
DATASET_URL   = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATASET_PATH):
        size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
        print(f"[DATA] Dataset already present ({size_mb:.0f} MB) — skipping download.")
        return

    print(f"[DATA] Downloading Sentiment140 (~80 MB) ...")
    print(f"       URL : {DATASET_URL}")
    print("       Please wait ...\n")

    def _progress(count, block, total):
        done = min(int(count * block * 50 / total), 50)
        sys.stdout.write(f"\r       [{'█' * done}{'░' * (50 - done)}] "
                         f"{min(count * block * 100 // total, 100)}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, reporthook=_progress)
    print("\n[DATA] Download complete. Extracting ...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist():
            if "training" in name.lower() and name.endswith(".csv"):
                z.extract(name, DATA_DIR)
                extracted = os.path.join(DATA_DIR, name)
                os.rename(extracted, DATASET_PATH)
                print(f"[DATA] Saved to:  {DATASET_PATH}")
                break

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)


def print_results(results: dict):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TRAINING RESULTS")
    print(sep)
    for name, m in results.items():
        cm = m["confusion_matrix"]
        print(f"\n  {name}")
        print(f"    Accuracy  : {m['accuracy'] * 100:.2f}%")
        print(f"    F1 Score  : {m['f1_score']  * 100:.2f}%")
        print(f"    Confusion Matrix:")
        print(f"      True Neg  = {cm[0][0]:>7,}   False Pos = {cm[0][1]:>7,}")
        print(f"      False Neg = {cm[1][0]:>7,}   True Pos  = {cm[1][1]:>7,}")
    print(f"\n{sep}")
    print("  Done! Start the dashboard with:  streamlit run dashboard.py")
    print(sep + "\n")


if __name__ == "__main__":
    # Parse --sample argument
    sample_size = 100_000
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        try:
            sample_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("[WARN] Invalid --sample value. Using default 100000.")

    print("=" * 60)
    print(f"  ML Training Pipeline  |  sample_size = {sample_size:,}")
    print("=" * 60)

    download_dataset()

    from ml_models import train_and_save
    results = train_and_save(sample_size=sample_size)
    print_results(results)
