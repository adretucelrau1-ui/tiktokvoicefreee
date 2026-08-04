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
if defined USE_CPU (
    echo USE_CPU=1 set — installing CPU-only PyTorch.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    :: Try CUDA 12.1 build first; fall back to CPU if it fails
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    if errorlevel 1 (
        echo CUDA build failed — falling back to CPU-only PyTorch.
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    )
)
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

:: --- 8. Install remaining project requirements --------------------------------
if exist "requirements.txt" (
    echo Installing project requirements from requirements.txt...
    pip install -r requirements.txt --quiet
    echo Done.
    echo.
)

:: --- 9. Verify installation ---------------------------------------------------
echo Verifying installation...
python -c "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "from TTS.api import TTS; print('  TTS: OK')"
python -c "import soundfile; print('  soundfile: OK')"
echo.

:: --- 10. Show next steps -------------------------------------------------------
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Place your XTTS v2 model directory (containing model.pth,
echo      config.json, vocab.json) somewhere accessible and set
echo      "model_path" in xtts_ro_config.json.
echo      Leave "model_path" empty to auto-download the model on
echo      first run (requires internet, ~1.8 GB).
echo.
echo   2. Place a Romanian speaker reference WAV file next to
echo      cliptic (e.g. xtts_speaker_ref.wav) and set
echo      "speaker_ref_path" in xtts_ro_config.json.
echo      The reference audio should be 6-30 seconds of clean,
echo      noise-free Romanian speech.
echo.
echo   3. Run the GUI:
echo        .venv\Scripts\python cliptic
echo.
echo   4. In the GUI:
echo        a) Check "xtts ro voice" to enable local XTTS RO synthesis.
echo        b) Check "Replace voice with AI (TTS)" if not already enabled.
echo        c) Process your video as usual.
echo.
echo   GPU note: CUDA is used automatically when available.
echo   To force CPU, set USE_CPU=1 before re-running setup.
echo.
pause
