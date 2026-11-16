import os 
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152ea5b71d23423dd4d7/GTSRB_Final_Training_Images.zip"

PROJECT_ROOT=Path(__file__).resolve().parent
DATA_DIR =PROJECT_ROOT / "data"
ZIP_PATH=DATA_DIR/"GTSRB_Final_Training_Images.zip"


def download_progress_hook(block_num,block_size,total_size):
    """Callback function to print a live download progress bar in the terminal."""
    downloaded = block_num * block_size
    percent = min(100.0, (downloaded / total_size) * 100.0)
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(
        f"\r[Downloading] {percent:5.1f}% ({downloaded_mb:5.1f} MB / {total_mb:5.1f} MB)",
        end="",
        flush=True,
    )

def download_and_extract_gtsrb():
    """Download the GTSRB dataset archive and extract images into data/."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Connecting to dataset server: {DATA_URL}")
        urllib.request.urlretrieve(
            DATA_URL, ZIP_PATH, reporthook=download_progress_hook
        )
        print("\n[Download Complete] Archive downloaded successfully.")
    else:
        print(f"[Found] Archive already exists at {ZIP_PATH.name}")

    extract_target = DATA_DIR / "GTSRB"
    if not extract_target.exists():
        print("Extracting archive into data/ folder...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("[Extraction Complete] GTSRB dataset ready to use!")
    else:
        print("[Found] Dataset already extracted in data/GTSRB.")


if __name__ == "__main__":
    download_and_extract_gtsrb()