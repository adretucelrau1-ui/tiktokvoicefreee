@echo off
:: setup_xtts_ro.bat - Auto-setup for Romanian XTTS voice pipeline
:: Run this script once from the project root to install all requirements.
::
:: GPU (CUDA) vs CPU:
::   - If an NVIDIA GPU with CUDA is present, PyTorch will use it automatically.
::   - If no GPU is found, synthesis falls back to CPU (slower but fully functional).
::   - To force CPU even if GPU is available, set USE_CPU=1 before running.
::
:: Usage:
::   cd path\to\tiktokvoicefreee
::   setup_xtts_ro.bat

setlocal enabledelayedexpansion

echo ============================================================
echo  XTTS RO - Auto-setup for Romanian XTTS voice pipeline
echo ============================================================
echo.

:: --- 1. Locate Python ---------------------------------------------------------
set PYTHON=python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found in PATH.
    echo Please install Python 3.9+ from https://python.org and re-run this script.
    pause
    exit /b 1
)
for /f "tokens=*" %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo Found: %PYVER%
echo.

:: --- 2. Create / activate virtual environment --------------------------------
if not exist ".venv" (
    echo Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment (.venv) already exists.
)

call .venv\Scripts\activate.bat
echo Activated .venv
echo.

:: --- 3. Upgrade pip -----------------------------------------------------------
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

:: --- 4. Install PyTorch (with CUDA if available, else CPU) -------------------
echo Installing PyTorch...
if defined USE_CPU goto install_cpu_torch
:: Try CUDA 12.1 build first; fall back to CPU if it fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 goto install_cpu_torch
goto torch_done
:install_cpu_torch
echo Installing CPU-only PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
:torch_done
echo PyTorch installed.
echo.

:: --- 5. Install Coqui TTS (XTTS v2) ------------------------------------------
echo Installing Coqui TTS (this may take a few minutes)...
pip install TTS --quiet
if errorlevel 1 (
    echo ERROR: Failed to install TTS package. Check your internet connection and try again.
    pause
    exit /b 1
)
echo Coqui TTS installed.
echo.

:: --- 6. Install soundfile (WAV I/O without torchaudio codec) -----------------
echo Installing soundfile...
pip install soundfile --quiet
echo soundfile installed.
echo.

:: --- 7. Install scipy (optional — used for resampling) -----------------------
echo Installing scipy...
pip install scipy --quiet
echo scipy installed.
echo.

:: --- 8. Install huggingface_hub (needed to download Romanian model) ----------
echo Installing huggingface_hub...
pip install huggingface_hub --quiet
echo huggingface_hub installed.
echo.

:: --- 9. Install remaining project requirements --------------------------------
if exist "requirements.txt" (
    echo Installing project requirements from requirements.txt...
    pip install -r requirements.txt --quiet
    echo Done.
    echo.
)

:: --- 10. Verify installation ---------------------------------------------------
echo Verifying installation...
python -c "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "from TTS.api import TTS; print('  TTS: OK')"
python -c "import soundfile; print('  soundfile: OK')"
python -c "import huggingface_hub; print('  huggingface_hub:', huggingface_hub.__version__)"
echo.

:: --- 11. Download Romanian fine-tuned XTTS v2 model --------------------------
echo ============================================================
echo  Downloading Romanian fine-tuned XTTS v2 model
echo  Source: https://huggingface.co/eduardem/xtts-v2-romanian
echo ============================================================
echo.
echo This model is fine-tuned specifically for Romanian speech.
echo It natively supports the 'ro' language — much better quality
echo than the generic XTTS v2 model for Romanian text.
echo.
echo Download size: ~1.8 GB  (one-time, saved to models\xtts-v2-romanian\)
echo.

set MODEL_DIR=%~dp0models\xtts-v2-romanian

if exist "%MODEL_DIR%\config.json" (
    if exist "%MODEL_DIR%\model.pth" (
        echo Romanian model already downloaded at: %MODEL_DIR%
        echo Skipping download.
        goto model_done
    )
)

echo Downloading model from HuggingFace...
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='eduardem/xtts-v2-romanian', local_dir=r'%MODEL_DIR%', ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'])"
if errorlevel 1 (
    echo.
    echo WARNING: Romanian model download failed.
    echo The script will fall back to the generic XTTS v2 model on first run.
    echo You can retry manually:
    echo   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='eduardem/xtts-v2-romanian', local_dir='models\xtts-v2-romanian')"
    goto model_done
)
echo Romanian model downloaded successfully: %MODEL_DIR%
echo.

:: Update xtts_ro_config.json with model_path (only if currently empty)
python -c "import json,os; p=r'%~dp0xtts_ro_config.json'; cfg=json.load(open(p)) if os.path.exists(p) else {}; cfg.setdefault('model_path','') or (cfg.update({'model_path':r'%MODEL_DIR%'}) or json.dump(cfg,open(p,'w'),indent=4,ensure_ascii=False)) if not cfg.get('model_path') else None; print('model_path:', cfg.get('model_path'))"

:model_done
echo.

:: --- 12. Show next steps -------------------------------------------------------
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Place a Romanian speaker WAV (6-30s) in the voices\ folder:
echo        python manage_voices.py --add "path\to\speaker.wav" --name "vocea_mea"
echo        python manage_voices.py --select "vocea_mea"
echo.
echo   2. Enable 'xtts ro voice' in the Cliptic UI, then process your video.
echo.
echo   3. Run the GUI:
echo        .venv\Scripts\python cliptic
echo.
echo   GPU note: CUDA is used automatically when available.
echo   To force CPU, set USE_CPU=1 before re-running setup.
echo.
pause
