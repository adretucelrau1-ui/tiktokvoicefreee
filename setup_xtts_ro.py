"""
setup_xtts_ro.py - Auto-setup for Romanian XTTS voice pipeline
Run once from the project folder:
    python setup_xtts_ro.py
"""
import os
import sys
import subprocess

def run(cmd, check=True):
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"\nERROR: command failed with exit code {result.returncode}")
        input("Press Enter to exit...")
        sys.exit(1)
    return result.returncode

print("=" * 60)
print(" XTTS RO - Auto-setup for Romanian XTTS voice pipeline")
print("=" * 60)
print()

# --- 1. Check Python version ---
print(f"Found: Python {sys.version.split()[0]}")
print()

# --- 2. Create virtual environment ---
venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
if not os.path.exists(venv_dir):
    print("Creating virtual environment (.venv)...")
    run([sys.executable, "-m", "venv", ".venv"])
    print("Virtual environment created.")
else:
    print("Virtual environment (.venv) already exists.")

# Determine pip/python paths inside venv
if sys.platform == "win32":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
else:
    venv_python = os.path.join(venv_dir, "bin", "python")
    venv_pip = os.path.join(venv_dir, "bin", "pip")

print(f"Using venv: {venv_dir}")
print()

# --- 3. Upgrade pip ---
print("Upgrading pip...")
run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
print()

# --- 4. Install PyTorch ---
print("Installing PyTorch...")
use_cpu = os.environ.get("USE_CPU", "")
if use_cpu:
    print("USE_CPU set — installing CPU-only PyTorch.")
    run([venv_pip, "install", "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"])
else:
    # Try CUDA 12.8 first (RTX 5000 series - sm_120 / Blackwell architecture)
    print("Trying PyTorch with CUDA 12.8 (RTX 5000 series support)...")
    ret = run([venv_pip, "install", "torch", "torchvision", "torchaudio",
               "--index-url", "https://download.pytorch.org/whl/cu128", "--quiet"], check=False)
    if ret != 0:
        # Fall back to CUDA 12.1
        print("CUDA 12.8 not available — trying CUDA 12.1...")
        ret = run([venv_pip, "install", "torch", "torchvision", "torchaudio",
                   "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], check=False)
    if ret != 0:
        print("CUDA build failed — falling back to CPU-only PyTorch.")
        run([venv_pip, "install", "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"])
print("PyTorch installed.")
print()

# --- 5. Install compatible transformers (Coqui TTS requires BeamSearchScorer) ---
print("Installing compatible transformers version...")
run([venv_pip, "install", "transformers==4.33.3", "--quiet"])
print("transformers installed.")
print()

# --- 6. Install Coqui TTS (without overriding transformers) ---
print("Installing Coqui TTS (this may take a few minutes)...")
run([venv_pip, "install", "TTS", "--no-deps", "--quiet"])
# Install TTS dependencies except transformers
run([venv_pip, "install", "trainer", "coqpit", "inflect", "anyascii",
     "bangla", "bnnumerizer", "bnunicodenormalizer", "gruut", "jamo",
     "jieba", "pypinyin", "tqdm", "librosa", "unidecode", "pysbd",
     "encodec", "--quiet"], check=False)
print("Coqui TTS installed.")
print()

# --- 7. Install soundfile ---
print("Installing soundfile...")
run([venv_pip, "install", "soundfile", "--quiet"])
print("soundfile installed.")
print()

# --- 8. Install scipy ---
print("Installing scipy...")
run([venv_pip, "install", "scipy", "--quiet"])
print("scipy installed.")
print()

# --- 9. Install from requirements.txt if present ---
req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
if os.path.exists(req_file):
    print("Installing from requirements.txt...")
    run([venv_pip, "install", "-r", req_file, "--quiet"])
    print("Done.")
    print()

# --- 10. Verify ---
print("Verifying installation...")
run([venv_python, "-c", "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"])
run([venv_python, "-c", "from TTS.api import TTS; print('  TTS: OK')"])
run([venv_python, "-c", "import soundfile; print('  soundfile: OK')"])
print()

# --- 11. Next steps ---
print("=" * 60)
print(" Setup complete!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Set 'model_path' in xtts_ro_config.json (or leave empty to")
print("     auto-download ~1.8 GB on first run).")
print()
print("  2. Place a Romanian speaker WAV (6-30s) next to this folder")
print("     and set 'speaker_ref_path' in xtts_ro_config.json.")
print()
print("  3. Run the GUI:")
if sys.platform == "win32":
    print("       .venv\\Scripts\\python cliptic.py")
else:
    print("       .venv/bin/python cliptic.py")
print()
input("Press Enter to exit...")
