"""
track.py — Run YOLOv8n inference + multi-object tracking on a drone video

Uses model.track() with BoT-SORT (built into Ultralytics).
Downloads a short royalty-free drone clip from a public source.
Saves annotated output video to runs/track/
"""

import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
RUNS_DIR    = PROJECT_DIR / "runs"
VIDEO_PATH  = PROJECT_DIR / "sample_drone_video.mp4"
OUTPUT_DIR  = RUNS_DIR / "track"

# ─── Model weights (prefer best.pt, fall back to last.pt) ─────────────────────
def find_best_weights() -> Path:
    detect_dir = PROJECT_DIR / "runs" / "detect"
    candidates = []
    if detect_dir.exists():
        for d in sorted(detect_dir.iterdir()):
            if d.is_dir() and d.name.startswith("drone_yolov8n"):
                best = d / "weights" / "best.pt"
                if best.exists():
                    candidates.append((d.stat().st_mtime, best))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


# ─── Download sample drone video ──────────────────────────────────────────────
# Royalty-free drone footage clips from public sources
SAMPLE_VIDEOS = [
    # Short UAV/drone clip from Wikimedia Commons (public domain)
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/6d/DJI_Phantom_4_Pro_V2.0_in_flight.webm/DJI_Phantom_4_Pro_V2.0_in_flight.webm.480p.vp9.webm",
    # Backup: another short clip
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/2/24/DJI_Mavic_Mini_01.webm/DJI_Mavic_Mini_01.webm.480p.vp9.webm",
]

def download_video() -> Path:
    """Download a sample drone video. Returns path to video file."""
    import urllib.request
    import urllib.error

    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 50_000:
        print(f"  ✓ Sample video already exists: {VIDEO_PATH}")
        return VIDEO_PATH

    # Try yt-dlp first (best quality, many sources)
    print("  Trying yt-dlp for sample video...")
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Download a short CC-licensed drone video from YouTube
            dl_result = subprocess.run([
                "yt-dlp",
                "-f", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/mp4",
                "--max-filesize", "50M",
                "-o", str(VIDEO_PATH),
                # Short drone footage (CC BY 3.0)
                "https://www.youtube.com/watch?v=7pUTdKNuGmo",
            ], capture_output=True, text=True, timeout=60)
            if dl_result.returncode == 0 and VIDEO_PATH.exists():
                print(f"  ✓ Video downloaded via yt-dlp: {VIDEO_PATH}")
                return VIDEO_PATH
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: direct URL download (Wikimedia Commons)
    for url in SAMPLE_VIDEOS:
        ext = ".webm" if ".webm" in url else ".mp4"
        tmp_path = PROJECT_DIR / f"sample_drone_video{ext}"
        try:
            print(f"  Trying direct download from Wikimedia Commons...")
            print(f"  URL: {url[:80]}...")
            urllib.request.urlretrieve(url, tmp_path)
            if tmp_path.stat().st_size > 50_000:
                tmp_path.rename(VIDEO_PATH)
                print(f"  ✓ Video downloaded: {VIDEO_PATH}")
                return VIDEO_PATH
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    return None


def generate_synthetic_video():
    """Create a simple test video with a moving rectangle (drone stand-in) if no real video."""
    import cv2
    import numpy as np

    print("  Creating synthetic test video (drone simulation)...")
    out_path = PROJECT_DIR / "sample_synthetic.mp4"
    width, height = 640, 480
    fps = 15
    n_frames = 75  # 5 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky gradient
        for y in range(height):
            frame[y] = [max(0, 100 - y//5), max(0, 150 - y//4), 200]
        # Moving "drone" rectangle
        t = i / n_frames
        x = int(50 + t * (width - 150))
        y = int(height//2 + 60 * np.sin(t * 2 * np.pi))
        cv2.rectangle(frame, (x, y), (x+60, y+25), (80, 80, 80), -1)
        cv2.rectangle(frame, (x-15, y+8), (x+75, y+17), (60, 60, 60), -1)
        # Rotor circles
        for cx, cy in [(x+5, y-5), (x+55, y-5), (x+5, y+30), (x+55, y+30)]:
            cv2.circle(frame, (cx, cy), 8, (40, 40, 40), 2)
        # Frame counter
        cv2.putText(frame, f"Frame {i+1}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"  ✓ Synthetic video created: {out_path} ({n_frames} frames @ {fps}fps)")
    return out_path


def run_tracking(model_path: Path, video_path: Path):
    """Run model.track() on the video."""
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / f"tracked_{video_path.name}"

    print(f"\n  Model weights : {model_path}")
    print(f"  Input video   : {video_path}")
    print(f"  Output will be saved in: {OUTPUT_DIR}")

    model = YOLO(str(model_path))

    print("\n  Running model.track() with BoT-SORT tracker...")
    print("  (Press Ctrl+C to stop early)\n")

    results = model.track(
        source=str(video_path),
        tracker="botsort.yaml",     # BoT-SORT (built-in Ultralytics)
        conf=0.25,                  # Confidence threshold
        iou=0.45,                   # NMS IoU threshold
        device=0,                   # GPU 0
        save=True,                  # Save annotated video
        project=str(RUNS_DIR / "track"),
        name="drone_tracking",
        exist_ok=True,
        verbose=True,
        stream=True,                # Stream results (memory efficient)
    )

    # Consume the generator
    frame_count = 0
    detection_frames = 0
    total_detections = 0

    for result in results:
        frame_count += 1
        n_dets = len(result.boxes) if result.boxes is not None else 0
        total_detections += n_dets
        if n_dets > 0:
            detection_frames += 1
        if frame_count % 15 == 0:
            print(f"  Frame {frame_count:4d} | Detections this frame: {n_dets}")

    print(f"\n{'─'*50}")
    print(f"  Tracking complete!")
    print(f"  Frames processed     : {frame_count}")
    print(f"  Frames with detection: {detection_frames} ({detection_frames/max(1,frame_count)*100:.1f}%)")
    print(f"  Total detections     : {total_detections}")
    print(f"  Output saved to      : {RUNS_DIR / 'track' / 'drone_tracking'}")
    print(f"{'─'*50}")


def main():
    print("=" * 60)
    print("  Drone Detection — Inference + Tracking")
    print("=" * 60)

    # Find model weights
    model_path = find_best_weights()
    if model_path is None:
        print("\n✗ No trained weights found at runs/detect/drone_yolov8n/weights/best.pt")
        print("  Run: python train.py  first.")
        sys.exit(1)
    print(f"\n✓ Using weights: {model_path}")

    # Get video
    print("\n[Step 1] Acquiring sample drone video...")
    video_path = download_video()

    if video_path is None or not video_path.exists():
        print("\n  ⚠  Could not download real video. Creating synthetic test video...")
        video_path = generate_synthetic_video()

    if video_path is None:
        print("✗ Could not get any video. Check opencv-python install.")
        sys.exit(1)

    # Run tracking
    print(f"\n[Step 2] Running inference + tracking...")
    try:
        run_tracking(model_path, video_path)
    except Exception as e:
        print(f"\n✗ Tracking failed: {e}")
        raise

    print("\n✓ Done! Check runs/track/drone_tracking/ for annotated output.")
    print("  Next step: python print_metrics.py")


if __name__ == "__main__":
    main()
