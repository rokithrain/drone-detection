# Drone Detection with YOLOv8n

A complete drone detection and tracking pipeline trained locally on consumer GPU hardware using Ultralytics YOLOv8. Detects a single class — **drone** — with real-time inference and multi-object tracking.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Model Choice](#model-choice)
3. [Training Configuration](#training-configuration)
4. [Results](#results)
5. [Running the Project](#running-the-project)
6. [Project Structure](#project-structure)
7. [Known Limitations](#known-limitations)

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | [Roboflow Universe](https://universe.roboflow.com/) — public drone detection dataset |
| **Format** | YOLOv8 (images + YOLO-format `.txt` label files + `data.yaml`) |
| **Classes** | 1 — `drone` |
| **Size** | ~400–800 annotated images (train / valid / test split) |
| **License** | **CC BY 4.0** — free to use for research and commercial purposes with attribution |

### About the License

CC BY 4.0 (Creative Commons Attribution 4.0 International) means you can:
- Use it freely for any purpose, including commercial projects
- Redistribute and adapt it
- As long as you credit the original dataset author

### Dataset Attribution

Dataset sourced from Roboflow Universe. Annotations are in YOLO format: each `.txt` label file contains one line per object — `<class_id> <cx> <cy> <width> <height>` — all normalized to [0, 1] relative to the image dimensions.

---

## Model Choice

### Why YOLOv8n (Nano)?

This project uses **YOLOv8n** — the smallest variant of the YOLOv8 family. Here's why it was chosen over larger options:

| Model | Parameters | mAP (COCO) | Inference (T4) | Reason |
|---|---|---|---|---|
| YOLOv8n | **3.2M** | 37.3 | 1.47ms | ✅ **Chosen — fastest, lowest VRAM** |
| YOLOv8s | 11.2M | 44.9 | 1.89ms | Slower, more VRAM |
| YOLOv8m | 25.9M | 50.2 | 3.52ms | Much slower |
| YOLOv8l | 43.7M | 52.9 | 6.12ms | Would strain 8GB VRAM |

**Hardware constraint**: RTX 4060 with 8GB VRAM. At `imgsz=640`, batch=16, YOLOv8n uses ~2–3GB VRAM — comfortable. The larger variants risk OOM errors or require batch=1–2, which slows training significantly.

**Dataset size constraint**: With only a few hundred images, a nano model is actually ideal. Larger models are prone to overfitting on small datasets unless heavily regularized. YOLOv8n's capacity is well-matched to a ~500-image dataset.

**Task simplicity**: Single-class detection doesn't require the representational power of a large model — the nano variant is more than sufficient.

---

## Training Configuration

```yaml
model:      yolov8n.pt         # Nano — pretrained on COCO ImageNet
epochs:     50
imgsz:      640                # Standard YOLO input resolution
batch:      16                 # (fallback to 8 if OOM)
optimizer:  AdamW              # Better generalization than SGD for small datasets
lr0:        0.001              # Initial learning rate
patience:   10                 # Early stopping: stops if no improvement for 10 epochs
amp:        true               # Mixed precision (FP16) — saves VRAM, speeds up training
device:     0                  # CUDA GPU (RTX 4060)
seed:       42                 # Reproducibility
```

### Why AdamW over SGD?

AdamW (Adam with decoupled weight decay) converges faster on small datasets and is more robust to learning rate tuning. SGD with momentum can outperform AdamW given large datasets and careful scheduling, but for a few hundred images trained to 50 epochs, AdamW reliably reaches good minima faster.

### Mixed Precision (AMP)

`amp=True` enables automatic mixed precision — the model runs forward passes in FP16 (half precision) while keeping certain critical operations in FP32. This roughly halves VRAM usage and speeds up training by ~40% on Ampere/Ada architecture GPUs like the RTX 4060, with negligible accuracy impact.

---

## Results

> Results are populated automatically after training. Run `python print_metrics.py` to see the full breakdown.

### Metrics (Final Epoch)

| Metric | Value | What It Means |
|---|---|---|
| **Precision (P)** | — | Of all drone bounding boxes the model predicted, this fraction were actually drones. High precision = few false alarms. |
| **Recall (R)** | — | Of all actual drones in the validation images, this fraction were correctly detected. High recall = few missed drones. |
| **mAP@50** | — | Mean Average Precision at 50% Intersection-over-Union (IoU). The standard detection metric. Think of it as "did the model find the drone AND roughly box it correctly?" Anything above 0.8 is good for a small dataset. |
| **mAP@50-95** | — | mAP averaged across IoU thresholds from 0.50 to 0.95 (in 0.05 steps). This is stricter — it demands tighter bounding boxes. Values above 0.5 are solid for a compact model on limited data. |

### Understanding IoU

IoU (Intersection over Union) measures how well the predicted bounding box overlaps with the ground-truth box:

```
IoU = Area of Overlap / Area of Union
```

- IoU = 1.0 → perfect overlap
- IoU = 0.5 → prediction overlaps the ground truth by half (the minimum acceptable threshold)
- IoU < 0.5 → detection is treated as a false positive

### Training Plots

After training, Ultralytics saves the following plots to `runs/detect/drone_yolov8n/`:

- `results.png` — loss curves and mAP over epochs
- `confusion_matrix.png` — true/false positive/negative breakdown
- `PR_curve.png` — Precision-Recall curve
- `val_batch*.jpg` — sample validation predictions with bounding boxes

---

## Running the Project

### Prerequisites

- Python 3.10–3.13
- NVIDIA GPU with CUDA 12.x drivers
- ~5GB free disk space (for PyTorch + dataset)
- A free [Roboflow API key](https://roboflow.com) (for dataset download)

### Step-by-Step

```powershell
# 1. Navigate to the project
cd drone-detection

# 2. Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies (PyTorch + Ultralytics)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics roboflow opencv-python tqdm requests

# 4. Download the drone dataset
python download_dataset.py
# (You'll be prompted for your Roboflow API key if not set in env)

# 5. Train the model
python train.py
# Expected: ~25–75 minutes on RTX 4060 (depends on dataset size)

# 6. Run inference + tracking on a sample video
python track.py

# 7. Print final training metrics
python print_metrics.py
```

### Environment Variable (Optional)

To skip the API key prompt:
```powershell
$env:ROBOFLOW_API_KEY = "your_key_here"
python download_dataset.py
```

---

## Project Structure

```
drone-detection/
├── venv/                         # Python virtual environment (gitignored)
├── dataset/                      # Downloaded drone dataset (gitignored)
│   ├── train/
│   │   ├── images/               # Training images (.jpg)
│   │   └── labels/               # YOLO format annotations (.txt)
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml                 # Dataset config for Ultralytics
├── runs/                         # Training & inference outputs (auto-created)
│   ├── detect/drone_yolov8n/
│   │   ├── weights/
│   │   │   ├── best.pt           # Best checkpoint (use this for inference)
│   │   │   └── last.pt           # Last epoch checkpoint
│   │   ├── results.csv           # Per-epoch metrics
│   │   └── *.png                 # Training plots
│   └── track/drone_tracking/
│       └── *.mp4                 # Annotated tracking output video
├── setup_env.ps1                 # One-shot environment setup script
├── download_dataset.py           # Fetches dataset from Roboflow Universe
├── train.py                      # Training script (batch OOM fallback included)
├── track.py                      # Inference + BoT-SORT tracking demo
├── print_metrics.py              # Reads results.csv, prints formatted metrics
└── README.md                     # This file
```

---

## Known Limitations

### 1. Small Dataset Size

A few hundred images is a very small training set for a detection model. Real-world deployments typically use thousands of images with diverse conditions. The model may:
- Struggle with drones it hasn't seen before (novel shapes, colors)
- Overfit to specific backgrounds present in the training data
- Perform poorly in different lighting conditions (night, direct sun glare)

**Mitigation**: Ultralytics applies random augmentations (mosaic, flips, HSV shifts) by default. Still, more data is always better.

### 2. Single Class

The model only detects drones — it cannot distinguish between drone types (quadcopter, fixed-wing, hexacopter) or differentiate drones from birds and planes. In real applications, multi-class models or a two-stage pipeline would be more robust.

### 3. Domain Shift (Indoor vs. Outdoor)

Most public drone datasets are captured outdoors against sky backgrounds. If you test this model on indoor footage, performance will degrade significantly because the visual context (background, lighting, scale) is very different from training data.

### 4. No Re-ID Across Camera Cuts

The BoT-SORT tracker maintains IDs within a single video stream. When the drone leaves and re-enters the frame, it may be assigned a new track ID. This is a fundamental limitation of appearance-based tracking without a re-identification model.

### 5. Speed vs. Accuracy Trade-off

YOLOv8n was chosen for speed. For applications requiring higher accuracy (security, airport safety), a larger model (YOLOv8m or YOLOv8l) trained on a much larger dataset would be appropriate.

### 6. CUDA Dependency

The training scripts require an NVIDIA GPU with CUDA. CPU-only training at 50 epochs would take 10–20× longer (~8–25 hours) and is not practical for development iteration.

---

## License

- **Model**: [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- **Dataset**: CC BY 4.0 — see dataset source on Roboflow Universe for attribution
- **Training/inference scripts**: MIT (this project)
