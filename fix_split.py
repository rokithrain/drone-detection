import random
import shutil
from pathlib import Path

random.seed(42)

root = Path("dataset")
train_img = root / "train" / "images"
train_lbl = root / "train" / "labels"
valid_img = root / "valid" / "images"
valid_lbl = root / "valid" / "labels"
test_img = root / "test" / "images"
test_lbl = root / "test" / "labels"

for d in [valid_img, valid_lbl, test_img, test_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# Find training images that have NON-EMPTY labels
pairs = []

for label in train_lbl.glob("*.txt"):
    if label.stat().st_size == 0:
        continue

    stem = label.stem

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img = train_img / (stem + ext)
        if img.exists():
            pairs.append((img, label))
            break

print(f"Found {len(pairs)} labeled training image/label pairs.")

random.shuffle(pairs)

n_valid = int(len(pairs) * 0.10)
n_test = int(len(pairs) * 0.10)

valid_pairs = pairs[:n_valid]
test_pairs = pairs[n_valid:n_valid + n_test]

def move_pairs(items, img_dst, lbl_dst):
    for img, lbl in items:
        shutil.move(str(img), str(img_dst / img.name))
        shutil.move(str(lbl), str(lbl_dst / lbl.name))

move_pairs(valid_pairs, valid_img, valid_lbl)
move_pairs(test_pairs, test_img, test_lbl)

print(f"Moved to validation: {len(valid_pairs)}")
print(f"Moved to test:       {len(test_pairs)}")
print(f"Remaining training:  {len(pairs) - len(valid_pairs) - len(test_pairs)}")
print("DONE.")