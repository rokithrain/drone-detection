# setup_env.ps1 — Drone Detection Project Environment Setup
# RTX 4060 (CUDA 12.x), Python venv, PyTorch + Ultralytics

Write-Host "=== Drone Detection — Environment Setup ===" -ForegroundColor Cyan

# Step 1: Create virtual environment
Write-Host "`n[1/5] Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv
if (-not $?) { Write-Error "Failed to create venv. Is Python installed?"; exit 1 }
Write-Host "    venv created at ./venv" -ForegroundColor Green

# Step 2: Activate venv
Write-Host "`n[2/5] Activating venv..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

# Step 3: Upgrade pip
Write-Host "`n[3/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Step 4: Install PyTorch with CUDA 12.1 support (compatible with RTX 4060)
Write-Host "`n[4/5] Installing PyTorch (CUDA 12.1) + Ultralytics + dependencies..." -ForegroundColor Yellow
Write-Host "    This may take a few minutes (PyTorch is ~2.5GB)..." -ForegroundColor Gray

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if (-not $?) { Write-Error "PyTorch installation failed"; exit 1 }

pip install ultralytics roboflow opencv-python tqdm requests --quiet
if (-not $?) { Write-Error "Ultralytics/dependencies installation failed"; exit 1 }

Write-Host "    Packages installed successfully." -ForegroundColor Green

# Step 5: Verify CUDA
Write-Host "`n[5/5] Verifying CUDA availability..." -ForegroundColor Yellow
$cuda_check = python -c @"
import torch
print(f'PyTorch version  : {torch.__version__}')
print(f'CUDA available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version     : {torch.version.cuda}')
    print(f'GPU name         : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM             : {mem:.1f} GB')
    print('STATUS: CUDA OK — ready to train!')
else:
    print('STATUS: CUDA NOT FOUND — check your drivers!')
    exit(1)
"@
Write-Host $cuda_check

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Next step: run 'python download_dataset.py'" -ForegroundColor White
