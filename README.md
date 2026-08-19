# tiktokvoicefreee

TikTok-style video generator with GUI, translation, captions, and AI voice-over.

---

## Quick Start

```
python cliptic
```

---

## Romanian XTTS Voice ("xtts ro voice")

The project ships with a local **Romanian XTTS** voice pipeline that replaces
the online GenAI Pro TTS engine when enabled.

### One-time setup (Windows)

```bat
cd path\to\tiktokvoicefreee
setup_xtts_ro.bat
```

This script:
- Creates a `.venv` virtual environment
- Installs PyTorch (CUDA if an NVIDIA GPU is detected, otherwise CPU)
- Installs Coqui TTS, soundfile, scipy, and all project dependencies
- Prints next-steps instructions

**GPU vs CPU:** CUDA is used automatically when available.  
To force CPU, run `set USE_CPU=1` before executing the script.

### Model and speaker reference files

| File | Description |
|------|-------------|
| `xtts_ro_config.json` | Config: model path, speaker ref, tuning params |
| `xtts_speaker_ref.wav` | Romanian speaker reference audio (6-30 s, clean speech) |
| *(model dir)* | XTTS v2 checkpoint (`model.pth`, `config.json`, `vocab.json`) |

**Where to place files:**
- Put `xtts_speaker_ref.wav` in the same folder as `cliptic`.
- Set `"model_path"` in `xtts_ro_config.json` to your model directory.
  Leave it empty (`""`) to auto-download the model on first run (~1.8 GB).

### Using XTTS RO in the GUI

1. Launch the GUI: `.venv\Scripts\python cliptic`
2. In the **Translation & AI Voice** section:
   - Check **"xtts ro voice"** — enables local Romanian XTTS synthesis.
   - Check **"Replace voice with AI (TTS)"** (auto-enabled when you tick XTTS RO).
3. Process your video.  Translated text flows through OpenAI → XTTS RO → video mix.

### Configuration (`xtts_ro_config.json`)

```json
{
    "model_path": "",           // Path to XTTS v2 model dir, or "" to auto-download
    "speaker_ref_path": "xtts_speaker_ref.wav",
    "language": "ro",
    "max_chars_per_chunk": 220, // Long text is split into chunks of this size
    "crossfade_ms": 80,         // Crossfade between chunks (ms) to avoid audible gaps
    "temperature": 0.65,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 10.0,
    "sample_rate": 24000,
    "output_sample_rate": 24000
}
```

If your XTTS checkpoint does not support `"ro"` in its tokenizer, synthesis
automatically falls back to `"en"` to avoid chunk failures.

### Standalone CLI

```bash
python xtts_ro.py "Acesta este un test." --out output.wav
python xtts_ro.py "Text lung..." --model path/to/model --speaker ref.wav --max-chars 200
```

---

## GenAI Pro (original online TTS)

The original GenAI Pro TTS path is still available.  Keep **"xtts ro voice"**
unchecked and configure your JWT token in `tts_config.json` to use it.
