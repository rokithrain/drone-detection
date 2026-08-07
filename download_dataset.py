"""
download_dataset.py -- Fetch drone detection dataset from Roboflow Universe

Dataset: "Drone Detection" dataset (single class: drone)
Source: Roboflow Universe (public, CC BY 4.0)
Format: YOLOv8 / Ultralytics compatible
"""

import os
import sys
# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import shutil
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"

# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# Roboflow Universe public datasets (YOLOv8 format, single-class "drone")
# We try multiple known-good datasets in order of preference.
# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
DATASETS = [
    {
        "name": "Drone Detection (godworkspace)",
        "workspace": "godworkspace",
        "project": "drone-detection-dvhol",
        "version": 2,
        "license": "CC BY 4.0",
        "approx_images": 932,
    },
    {
        # Well-known community drone dataset
        # https://universe.roboflow.com/drone-detection-kdtnf/drone-detection-yolo
        "name": "Drone Detection YOLO (community)",
        "workspace": "drone-detection-kdtnf",
        "project": "drone-detection-yolo",
        "version": 3,
        "license": "CC BY 4.0",
        "approx_images": 580,
    },
    {
        # Backup: DroneDet dataset (December 2023 upload)
        "name": "DroneDet (backup)",
        "workspace": "dronedet",
        "project": "dronedet-zy9iu",
        "version": 1,
        "license": "CC BY 4.0",
        "approx_images": 420,
    },
]

# ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def try_roboflow_sdk(api_key: str, dataset_info: dict) -> bool:
    """Download using the roboflow Python SDK."""
    try:
        import roboflow
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
        version = project.version(dataset_info["version"])
        dataset = version.download("yolov8", location=str(DATASET_DIR), overwrite=True)
        print(f"  ??? Downloaded via Roboflow SDK to: {DATASET_DIR}")
        return True
    except Exception as e:
        print(f"  ??? SDK download failed: {e}")
        return False


def try_roboflow_export_url(api_key: str, dataset_info: dict) -> bool:
    """Download using Roboflow's export URL (requires API key)."""
    workspace = dataset_info["workspace"]
    project = dataset_info["project"]
    version = dataset_info["version"]
    url = (
        f"https://universe.roboflow.com/{workspace}/{project}"
        f"/dataset/{version}/download/yolov8?api_key={api_key}"
    )
    zip_path = PROJECT_DIR / "dataset.zip"
    try:
        print(f"  Trying export URL for {project} v{version}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATASET_DIR)
        zip_path.unlink(missing_ok=True)
        print(f"  ??? Downloaded and extracted to: {DATASET_DIR}")
        return True
    except Exception as e:
        print(f"  ??? Export URL failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def inspect_and_fix_structure():
    """Ensure dataset has train/valid/test folders with images and labels."""
    print("\n[Inspect] Checking dataset structure...")

    # Sometimes Roboflow zips have an extra top-level folder
    contents = list(DATASET_DIR.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        inner = contents[0]
        print(f"  Moving contents of '{inner.name}' up one level...")
        for item in inner.iterdir():
            shutil.move(str(item), str(DATASET_DIR / item.name))
        inner.rmdir()

    # Report what we have
    splits = ["train", "valid", "test"]
    total_images = 0
    for split in splits:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*")))
            n_lbls = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
            total_images += n_imgs
            print(f"  {split:6s}: {n_imgs} images, {n_lbls} labels")

    print(f"  Total images: {total_images}")

    if total_images == 0:
        print("  ??? No images found! Dataset structure may be unexpected.")
        print(f"  Contents of {DATASET_DIR}:")
        for p in DATASET_DIR.rglob("*"):
            if p.is_file():
                print(f"    {p.relative_to(DATASET_DIR)}")
        return False
    return True


def generate_data_yaml():
    """Generate a data.yaml if not present or fix the existing one."""
    yaml_path = DATASET_DIR / "data.yaml"

    # Check if valid/val exists ??? Roboflow uses 'valid', Ultralytics expects 'val' or 'valid'
    valid_exists = (DATASET_DIR / "valid" / "images").exists()
    val_exists = (DATASET_DIR / "val" / "images").exists()
    test_exists = (DATASET_DIR / "test" / "images").exists()

    val_dir = "valid/images" if valid_exists else ("val/images" if val_exists else None)
    if val_dir is None:
        print("  ??? No validation split found!")
        return False

    yaml_content = f"""# Drone Detection Dataset
# Source: Roboflow Universe ??? single class "drone"
# License: CC BY 4.0

path: {str(DATASET_DIR).replace(chr(92), '/')}
train: train/images
val: {val_dir}
"""
    if test_exists:
        yaml_content += "test: test/images\n"

    yaml_content += """
nc: 1
names:
  - drone
"""
    yaml_path.write_text(yaml_content)
    print(f"\n??? data.yaml written to: {yaml_path}")
    print(f"  Contents:\n{'???'*40}")
    print(yaml_content)
    print("???"*40)
    return True


def main():
    print("=" * 60)
    print("  Drone Detection ??? Dataset Downloader")
    print("=" * 60)

    if DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.jpg")):
        print(f"\n??? Dataset already exists at {DATASET_DIR}")
        print("  Skipping download. Delete ./dataset to re-download.")
        inspect_and_fix_structure()
        generate_data_yaml()
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Get API key
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("\n+----------------------------------------------------------+")
        print("|  Roboflow API key required to download public datasets.  |")
        print("|                                                          |")
        print("|  Get a FREE key (60 seconds):                            |")
        print("|  1. Go to https://roboflow.com and sign up / log in      |")
        print("|  2. Go to Settings -> API Keys                           |")
        print("|  3. Copy your key and paste it below                     |")
        print("+----------------------------------------------------------+")
        api_key = input("\nPaste your Roboflow API key: ").strip()
        if not api_key:
            print("[FAIL] No API key provided. Cannot proceed.")
            sys.exit(1)

    print(f"\n[OK] API key received (length: {len(api_key)} chars)")

    # Try each dataset
    success = False
    for ds_info in DATASETS:
        print(f"\n????????? Trying: {ds_info['name']} ?????????")
        print(f"    License: {ds_info['license']}, ~{ds_info['approx_images']} images")

        # Try SDK first, then URL
        if try_roboflow_sdk(api_key, ds_info):
            success = True
            break
        if try_roboflow_export_url(api_key, ds_info):
            success = True
            break

    if not success:
        print("\n??? All download attempts failed.")
        print("  Please check your API key and internet connection.")
        sys.exit(1)

    ok = inspect_and_fix_structure()
    if not ok:
        sys.exit(1)

    generate_data_yaml()
    print("\n??? Dataset ready. Next step: python train.py")


if __name__ == "__main__":
    main()

