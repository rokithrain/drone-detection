"""
train.py — Train YOLOv8n for drone detection

Model   : YOLOv8n (nano — fastest, fits 8GB VRAM)
Dataset : Single-class "drone" in YOLO format
Hardware: RTX 4060 (8GB VRAM), CUDA
"""

import sys
import time
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_YAML   = PROJECT_DIR / "dataset" / "data.yaml"
RUNS_DIR    = PROJECT_DIR / "runs"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
MODEL       = "yolov8n.pt"          # Nano — smallest/fastest YOLOv8
EPOCHS      = 50
IMGSZ       = 640
OPTIMIZER   = "AdamW"
INITIAL_LR  = 0.001
PROJECT_NAME = str(RUNS_DIR / "detect")
EXP_NAME    = "drone_yolov8n"
BATCH_SIZES = [16, 8]               # Try 16 first, fall back to 8 on OOM

# ─────────────────────────────────────────────────────────────────────────────

def verify_cuda():
    import torch
    print("─" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n✗ CUDA not available. Check your drivers and PyTorch install.")
        print("  Expected: RTX 4060 with CUDA 12.x")
        sys.exit(1)
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM            : {mem_gb:.1f} GB")
    print("─" * 60)
    return mem_gb


def estimate_training_time(n_images: int, batch_size: int, epochs: int) -> None:
    """Print a rough estimate before training starts."""
    # YOLOv8n on RTX 4060 at 640px: ~25-40ms per image (forward+backward)
    # This is empirically derived from community benchmarks
    ms_per_img = 35      # ms per image (forward + backward, RTX 4060, YOLOv8n)
    steps_per_epoch = math.ceil(n_images / batch_size)
    seconds_per_epoch = (steps_per_epoch * batch_size * ms_per_img) / 1000
    total_seconds = seconds_per_epoch * epochs

    print(f"\n{'─'*60}")
    print(f"  ⏱  Training Time Estimate")
    print(f"{'─'*60}")
    print(f"  Dataset size    : {n_images} training images")
    print(f"  Batch size      : {batch_size}")
    print(f"  Steps/epoch     : {steps_per_epoch}")
    print(f"  Est. time/epoch : {seconds_per_epoch:.0f}s ({seconds_per_epoch/60:.1f} min)")
    print(f"  Total epochs    : {epochs}")
    print(f"  Est. total time : {total_seconds/60:.0f} min  ({total_seconds/3600:.1f} hrs)")
    print(f"{'─'*60}")
    if total_seconds / 60 > 90:
        print("  ⚠  Might take >90 min. Consider reducing epochs if needed.")
    print()


def count_training_images() -> int:
    train_dir = PROJECT_DIR / "dataset" / "train" / "images"
    if not train_dir.exists():
        return 500  # fallback estimate
    return len(list(train_dir.glob("*.[jJpP][pPnN][gG]")))


def train_with_batch(batch_size: int):
    from ultralytics import YOLO
    import torch

    print(f"\n🚀 Starting training with batch_size={batch_size}...")
    model = YOLO(MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch_size,
        optimizer=OPTIMIZER,
        lr0=INITIAL_LR,
        device=0,                    # CUDA GPU 0
        project=PROJECT_NAME,
        name=EXP_NAME,
        exist_ok=True,
        plots=True,                  # Save training plots
        save=True,                   # Save best + last weights
        verbose=True,
        patience=10,                 # Early stopping patience
        amp=True,                    # Mixed precision (saves VRAM)
        cache=False,                 # Don't cache to RAM (save memory)
        workers=4,
        seed=42,
    )
    return results


def main():
    print("=" * 60)
    print("  Drone Detection — YOLOv8n Training")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"✗ data.yaml not found at {DATA_YAML}")
        print("  Run: python download_dataset.py  first.")
        sys.exit(1)

    vram_gb = verify_cuda()
    n_train = count_training_images()

    # Choose batch size based on VRAM
    if vram_gb < 7:
        BATCH_SIZES.insert(0, 4)
        print(f"⚠  Only {vram_gb:.1f}GB VRAM detected — starting with smaller batches")

    # Estimate time before starting
    estimate_training_time(n_train, BATCH_SIZES[0], EPOCHS)

    # Try training, fall back on OOM
    results = None
    for batch_size in BATCH_SIZES:
        try:
            t_start = time.time()
            results = train_with_batch(batch_size)
            t_end = time.time()
            elapsed = t_end - t_start
            print(f"\n✓ Training complete in {elapsed/60:.1f} minutes (batch={batch_size})")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"\n⚠  OOM with batch={batch_size}. Retrying with smaller batch...")
                import torch
                torch.cuda.empty_cache()
                continue
            else:
                raise

    if results is None:
        print("✗ Training failed on all batch sizes.")
        sys.exit(1)

    # Print save location
    save_dir = Path(PROJECT_NAME) / EXP_NAME
    if not save_dir.exists():
        save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else RUNS_DIR / "detect" / EXP_NAME

    print(f"\n📁 Results saved to: {save_dir}")
    print(f"   Best weights   : {save_dir}/weights/best.pt")
    print(f"   Training plots : {save_dir}/*.png")
    print(f"\nNext step: python print_metrics.py")
    print(f"           python track.py")


if __name__ == "__main__":
    main()
