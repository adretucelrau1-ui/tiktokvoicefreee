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

# --- 9. Install huggingface_hub (needed to download Romanian model) ---
print("Installing huggingface_hub...")
run([venv_pip, "install", "huggingface_hub", "--quiet"])
print("huggingface_hub installed.")
print()

# --- 10. Install from requirements.txt if present ---
req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
if os.path.exists(req_file):
    print("Installing from requirements.txt...")
    run([venv_pip, "install", "-r", req_file, "--quiet"])
    print("Done.")
    print()

# --- 11. Download Romanian fine-tuned XTTS v2 model ---
print("=" * 60)
print(" Downloading Romanian fine-tuned XTTS v2 model")
print(" Source: https://huggingface.co/eduardem/xtts-v2-romanian")
print("=" * 60)
print()
print("This model is fine-tuned specifically for Romanian speech.")
print("It natively supports the 'ro' language — much better quality")
print("than the generic XTTS v2 model for Romanian text.")
print()
print("Download size: ~1.8 GB  (one-time, saved to models/xtts-v2-romanian/)")
print()

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models", "xtts-v2-romanian")
config_file = os.path.join(model_dir, "config.json")
model_file = os.path.join(model_dir, "model.pth")

if os.path.exists(config_file) and os.path.exists(model_file):
    print(f"Romanian model already downloaded at: {model_dir}")
    print("Skipping download.")
else:
    print("Downloading model from HuggingFace...")
    download_script = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "  repo_id='eduardem/xtts-v2-romanian',"
        f" local_dir=r'{model_dir}',"
        "  ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*']"
        ")"
    )
    ret = run([venv_python, "-c", download_script], check=False)
    if ret == 0 and os.path.exists(config_file):
        print(f"\nRomanian model downloaded successfully: {model_dir}")
    else:
        print("\nWARNING: Romanian model download failed.")
        print("The script will fall back to the generic XTTS v2 model on first run.")
        print("You can retry manually:")
        print(f'  python -c "from huggingface_hub import snapshot_download; '
              f'snapshot_download(repo_id=\'eduardem/xtts-v2-romanian\', local_dir=\'{model_dir}\')"')
print()

# --- 12. Update xtts_ro_config.json with model_path ---
config_json_path = os.path.join(script_dir, "xtts_ro_config.json")
if os.path.exists(config_file) and os.path.exists(model_file):
    import json
    try:
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}
        # Only set model_path if it is currently empty (don't override user setting)
        if not cfg.get("model_path"):
            cfg["model_path"] = model_dir
            with open(config_json_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
            print(f"xtts_ro_config.json updated: model_path = {model_dir}")
        else:
            print(f"xtts_ro_config.json already has model_path: {cfg['model_path']} — not changed.")
    except Exception as e:
        print(f"WARNING: Could not update xtts_ro_config.json: {e}")
    print()

# --- 13. Verify ---
print("Verifying installation...")
run([venv_python, "-c", "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"])
run([venv_python, "-c", "from TTS.api import TTS; print('  TTS: OK')"])
run([venv_python, "-c", "import soundfile; print('  soundfile: OK')"])
run([venv_python, "-c", "import huggingface_hub; print('  huggingface_hub:', huggingface_hub.__version__)"])
print()

# --- 14. Next steps ---
print("=" * 60)
print(" Setup complete!")
print("=" * 60)
print()
if os.path.exists(config_file) and os.path.exists(model_file):
    print("Romanian fine-tuned XTTS v2 model is ready.")
    print(f"  model_path = {model_dir}")
    print()
print("Next steps:")
print("  1. Place a Romanian speaker WAV (6–30s) in the voices/ folder:")
print("       python manage_voices.py --add \"path\\to\\speaker.wav\" --name \"vocea_mea\"")
print("       python manage_voices.py --select \"vocea_mea\"")
print()
print("  2. Enable 'xtts ro voice' in the Cliptic UI, then process your video.")
print()
print("  3. Run the GUI:")
if sys.platform == "win32":
    print("       .venv\\Scripts\\python cliptic.py")
else:
    print("       .venv/bin/python cliptic.py")
print()
input("Press Enter to exit...")
