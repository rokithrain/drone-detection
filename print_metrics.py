"""
print_metrics.py — Report final training metrics from results.csv

Reads the Ultralytics training results CSV and prints the final epoch metrics
with plain-English explanations.
"""

import sys
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
RUNS_DIR    = PROJECT_DIR / "runs" / "detect" / "drone_yolov8n"

def find_results_csv() -> Path:
    """Find results.csv — handles Ultralytics naming (drone_yolov8n, drone_yolov8n2, etc.)"""
    detect_dir = PROJECT_DIR / "runs" / "detect"
    candidates = []
    if detect_dir.exists():
        for d in sorted(detect_dir.iterdir()):
            if d.is_dir() and d.name.startswith("drone_yolov8n"):
                csv = d / "results.csv"
                if csv.exists():
                    candidates.append((d.stat().st_mtime, csv))
    if not candidates:
        return None
    # Return the most recently modified
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_results(csv_path: Path) -> dict:
    """Parse results.csv and return the last (best) epoch row."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    # Strip whitespace from keys/values
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}
    return last


def find_best_row(csv_path: Path) -> dict:
    """Return the row with the best mAP@50."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    if not rows:
        return {}
    map50_key = next((k for k in rows[0] if "map50" in k.lower() and "95" not in k.lower()), None)
    if not map50_key:
        return rows[-1]
    best = max(rows, key=lambda r: float(r.get(map50_key, 0) or 0))
    return best


def print_metric_block(label: str, value: str, explanation: str, bar_width: int = 30):
    try:
        val = float(value)
        bar = "█" * int(val * bar_width) + "░" * (bar_width - int(val * bar_width))
        print(f"\n  {label}")
        print(f"  [{bar}] {val:.4f} ({val*100:.1f}%)")
        print(f"  → {explanation}")
    except (ValueError, TypeError):
        print(f"\n  {label}: {value}")
        print(f"  → {explanation}")


def main():
    print("=" * 65)
    print("  Drone Detection — Training Metrics")
    print("=" * 65)

    csv_path = find_results_csv()
    if csv_path is None:
        print(f"\n✗ results.csv not found in runs/detect/")
        print("  Make sure training has completed: python train.py")
        sys.exit(1)

    print(f"\n  Results file: {csv_path}")

    last_row = parse_results(csv_path)
    best_row = find_best_row(csv_path)

    # Find column names (Ultralytics may have spaces in headers)
    def get(row, *keys):
        for k in keys:
            for col in row:
                if k.lower() in col.lower():
                    return row[col]
        return "N/A"

    # Total epochs trained
    epoch_key = next((k for k in last_row if "epoch" in k.lower()), None)
    total_epochs = last_row.get(epoch_key, "?") if epoch_key else "?"

    print(f"\n  Epochs trained  : {total_epochs}")
    print(f"  Results at last epoch:")
    print("─" * 65)

    # ── Final epoch metrics ────────────────────────────────────────────────
    precision  = get(last_row, "metrics/precision")
    recall     = get(last_row, "metrics/recall")
    map50      = get(last_row, "metrics/mAP50(B)")
    map50_95   = get(last_row, "metrics/mAP50-95(B)")

    print_metric_block(
        "Precision (P)",
        precision,
        "Of all drone detections made, this fraction were actually drones.\n"
        "  High precision = few false alarms (no ghost drones)."
    )
    print_metric_block(
        "Recall (R)",
        recall,
        "Of all real drones in the images, this fraction were detected.\n"
        "  High recall = few missed drones."
    )
    print_metric_block(
        "mAP@50",
        map50,
        "Mean Average Precision at 50% IoU overlap threshold.\n"
        "  The primary detection metric. >0.8 = good, >0.9 = excellent."
    )
    print_metric_block(
        "mAP@50-95",
        map50_95,
        "mAP averaged across IoU thresholds 0.50–0.95 (stricter).\n"
        "  Measures localization quality. >0.5 = solid for a small dataset."
    )

    # ── Best epoch metrics ─────────────────────────────────────────────────
    print("\n" + "─" * 65)
    best_epoch = get(best_row, "epoch") if best_row else "?"
    best_map50 = get(best_row, "metrics/mAP50(B)") if best_row else "?"
    best_map50_95 = get(best_row, "metrics/mAP50-95(B)") if best_row else "?"
    print(f"\n  Best epoch      : {best_epoch}")
    print(f"  Best mAP@50     : {best_map50}")
    print(f"  Best mAP@50-95  : {best_map50_95}")

    print("\n" + "─" * 65)
    print(f"\n  Best weights saved at:")
    print(f"  {csv_path.parent / 'weights' / 'best.pt'}")
    print(f"\n  Training plots at: {csv_path.parent}")
    print("\n" + "=" * 65)

    # ── Quick quality assessment ───────────────────────────────────────────
    try:
        m = float(map50)
        if m >= 0.90:
            verdict = "🏆 Excellent! Model is production-ready for single-class detection."
        elif m >= 0.75:
            verdict = "✅ Good. More data or epochs would push it further."
        elif m >= 0.50:
            verdict = "⚠️  Decent. Model detects drones but has room to improve."
        else:
            verdict = "❌ Low accuracy. Dataset may be too small or unbalanced."
        print(f"\n  Assessment: {verdict}")
    except (ValueError, TypeError):
        pass

    print()


if __name__ == "__main__":
    main()
