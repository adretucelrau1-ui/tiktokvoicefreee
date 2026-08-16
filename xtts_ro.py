"""
xtts_ro.py — Romanian XTTS voice generation pipeline.

Features:
  - Auto device selection (CUDA → CPU)
  - Romanian diacritic normalization (cedilla/comma variants)
  - Long-text chunking with configurable max chars per chunk
  - Crossfade concatenation to avoid audible pauses
  - soundfile-based WAV output (avoids torchaudio/torchcodec pitfalls)
  - Configurable via xtts_ro_config.json (model path, speaker ref, tuning params)

Usage (standalone):
    python xtts_ro.py "Acesta este un test." --out test_out.wav

Usage (from cliptic):
    from xtts_ro import generate_xtts_ro
    path = generate_xtts_ro(text, output_path=output_path, log=log)
"""

import os
import json
import math
import logging
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _THIS_DIR / "xtts_ro_config.json"

# Built-in XTTS v2 "studio" speakers bundled with the public checkpoint's
# speakers_xtts.pth.  These are ready-made voices that can be used without
# cloning anything — handy when the user doesn't want to use a custom
# cloned voice.  Kept as a static list so the UI can populate a speaker
# picker instantly, without having to load the (large) model first.  The
# authoritative list is always re-read from the loaded model when possible
# (see list_builtin_speakers()).
BUILTIN_SPEAKERS = [
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
    "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
    "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper",
    "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka",
    "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy",
    "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah",
    "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos",
    "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando",
    "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger",
    "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
    "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María",
    "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman",
    "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı",
    "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski",
]
DEFAULT_BUILTIN_SPEAKER = "Claribel Dervla"

_DEFAULT_CONFIG = {
    "model_path": "",           # Path to XTTS v2 model dir (or leave empty to auto-download)
    "voice_mode": "custom",     # "custom" = use active_voice/speaker_ref_path, "builtin" = use builtin_speaker
    "active_voice": "",         # Name of active voice profile from voices/ folder
    "speaker_ref_path": "",     # Path to reference WAV file for voice cloning
    "builtin_speaker": DEFAULT_BUILTIN_SPEAKER,  # Built-in XTTS studio speaker name (used when voice_mode == "builtin" or no custom voice is set)
    "language": "ro",
    "max_chars_per_chunk": 220, # Maximum characters per synthesis chunk
    "crossfade_ms": 80,         # Crossfade duration between chunks (milliseconds)
    "temperature": 0.65,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 10.0,
    "sample_rate": 24000,       # XTTS native sample rate
    "output_sample_rate": 24000 # Output WAV sample rate
}


def _load_config():
    """Load config from xtts_ro_config.json, falling back to defaults."""
    cfg = dict(_DEFAULT_CONFIG)
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            cfg.update({k: v for k, v in user_cfg.items() if k in _DEFAULT_CONFIG})
    except Exception as exc:
        logging.warning(f"[XTTS RO] Could not read config {_CONFIG_PATH}: {exc}")
    return cfg


# ---------------------------------------------------------------------------
# Romanian text normalization
# ---------------------------------------------------------------------------

def normalize_romanian(text: str) -> str:
    """
    Normalize Romanian diacritics: map cedilla variants (ş/Ş, ţ/Ţ) to
    comma-below variants (ș/Ș, ț/Ț) used by the XTTS Romanian model.
    Also replaces s-comma/t-comma with the model-friendly cedilla forms
    depending on direction required.  The XTTS ro model is trained on
    comma-below forms, so we always produce comma-below output.
    """
    replacements = {
        "\u015f": "\u0219",  # ş → ș (s-cedilla → s-comma)
        "\u015e": "\u0218",  # Ş → Ș
        "\u0163": "\u021b",  # ţ → ț (t-cedilla → t-comma)
        "\u0162": "\u021a",  # Ţ → Ț
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, max_chars: int) -> list:
    """
    Split text into chunks of at most *max_chars* characters, breaking
    preferentially at sentence boundaries (. ! ?) then at commas/semicolons,
    then at word boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    import re
    # Split by sentence-ending punctuation first
    sentence_re = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_re.split(text)

    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single sentence is too long, split at commas
            if len(sent) > max_chars:
                sub_parts = re.split(r'(?<=[,;])\s+', sent)
                sub_current = ""
                for part in sub_parts:
                    sub_candidate = (sub_current + " " + part).strip() if sub_current else part
                    if len(sub_candidate) <= max_chars:
                        sub_current = sub_candidate
                    else:
                        if sub_current:
                            chunks.append(sub_current)
                        # Last resort: break by word
                        words = part.split()
                        sub_current = ""
                        for w in words:
                            wc = (sub_current + " " + w).strip() if sub_current else w
                            if len(wc) <= max_chars:
                                sub_current = wc
                            else:
                                if sub_current:
                                    chunks.append(sub_current)
                                sub_current = w
                        sub_current = sub_current.strip()
                if sub_current:
                    chunks.append(sub_current)
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device(log=None):
    """Return 'cuda' if a usable GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            # Quick sanity check — allocate a tiny tensor
            _ = torch.zeros(1, device="cuda")
            if log:
                log(f"[XTTS RO] Device: cuda ({torch.cuda.get_device_name(0)})")
            return "cuda"
    except Exception as exc:
        if log:
            log(f"[XTTS RO] CUDA not usable ({exc}), falling back to CPU")
    if log:
        log("[XTTS RO] Device: cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# Model loader (lazy, cached)
# ---------------------------------------------------------------------------

_xtts_model_cache = {}  # key: (model_path, device) → model instance


def _load_xtts_model(model_path: str, device: str, log=None):
    """Load (and cache) the XTTS v2 model."""
    cache_key = (model_path, device)
    if cache_key in _xtts_model_cache:
        return _xtts_model_cache[cache_key]

    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        raise RuntimeError(
            "[XTTS RO] TTS package not found. "
            "Install with: pip install TTS"
        )

    if log:
        log("[XTTS RO] Loading XTTS v2 model…")

    if model_path and os.path.isdir(model_path):
        # Load from local checkpoint directory
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"[XTTS RO] config.json not found in model_path: {model_path}"
            )
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        xtts_config = XttsConfig()
        xtts_config.load_json(config_file)
        model = Xtts.init_from_config(xtts_config)
        checkpoint_path = os.path.join(model_path, "model.pth")
        vocab_path = os.path.join(model_path, "vocab.json")
        speakers_path = (
            os.path.join(model_path, "speakers_xtts.pth")
            if os.path.exists(os.path.join(model_path, "speakers_xtts.pth"))
            else None
        )
        model.load_checkpoint(
            xtts_config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            speaker_file_path=speakers_path,
            eval=True,
        )
        import torch
        model.to(device)
    else:
        # Auto-download the official XTTS v2 model via the TTS API.
        # NOTE: TTS.api.TTS's `gpu=` constructor kwarg is deprecated in favor of
        # explicitly moving the model with `.to(device)`, which is what we do here.
        model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    if log:
        log("[XTTS RO] Model loaded.")
    _xtts_model_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Speaker conditioning
# ---------------------------------------------------------------------------

def _get_speaker_ref(speaker_ref_path: str, log=None, active_voice: str = ""):
    """
    Resolve and validate the speaker reference audio path.
    If active_voice is set, looks in the voices/ folder first.
    Returns the path string if valid, else None.
    """
    # Try active_voice from voices/ folder first
    if active_voice:
        voices_dir = _THIS_DIR / "voices"
        voice_file = voices_dir / f"{active_voice}.wav"
        if voice_file.exists():
            if log:
                log(f"[XTTS RO] Using voice profile: {active_voice} ({voice_file})")
            return str(voice_file)
        else:
            if log:
                log(f"[XTTS RO] ⚠ Voice profile '{active_voice}' not found in voices/ — falling back to speaker_ref_path")

    if not speaker_ref_path:
        if log:
            log("[XTTS RO] No speaker_ref_path configured — using XTTS default voice")
        return None

    # Make absolute relative to repo root
    if not os.path.isabs(speaker_ref_path):
        candidate = str(_THIS_DIR / speaker_ref_path)
    else:
        candidate = speaker_ref_path

    if not os.path.exists(candidate):
        if log:
            log(f"[XTTS RO] ⚠ Speaker ref not found: {candidate} — using default voice")
        return None

    if log:
        log(f"[XTTS RO] Speaker reference: {candidate}")
    return candidate


def _load_speaker_ref_as_numpy(speaker_ref_path: str):
    """
    Load reference audio as (numpy float32 array, sample_rate) using
    soundfile to avoid torchaudio/torchcodec dependency.
    """
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(speaker_ref_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    return data, sr


# ---------------------------------------------------------------------------
# Voice listing helpers (used by cliptic's UI and manage_voices.py)
# ---------------------------------------------------------------------------

def list_custom_voices():
    """Return the sorted list of cloned voice names available in voices/."""
    voices_dir = _THIS_DIR / "voices"
    if not voices_dir.exists():
        return []
    return sorted(p.stem for p in voices_dir.iterdir() if p.suffix.lower() == ".wav")


def list_builtin_speakers(model=None):
    """
    Return the list of built-in XTTS v2 studio speaker names.

    If *model* (an already-loaded XTTS model instance) is provided and
    exposes a speaker manager, the authoritative list is read from it.
    Otherwise the static BUILTIN_SPEAKERS fallback is returned, which lets
    callers (e.g. a GUI dropdown) list the built-in voices without paying
    the cost of loading the model first.
    """
    if model is not None:
        try:
            names = list(model.speaker_manager.speakers.keys())
            if names:
                return sorted(names)
        except Exception:
            pass
    return list(BUILTIN_SPEAKERS)


def list_available_voices():
    """Return {'custom': [...], 'builtin': [...]} voice names."""
    return {"custom": list_custom_voices(), "builtin": list_builtin_speakers()}


# ---------------------------------------------------------------------------
# Audio concatenation with crossfade
# ---------------------------------------------------------------------------

def _crossfade_concat(arrays: list, sample_rate: int, crossfade_ms: int):
    """
    Concatenate a list of 1-D float32 numpy arrays with a linear crossfade.
    Returns a single float32 numpy array.
    """
    import numpy as np

    if not arrays:
        return np.array([], dtype=np.float32)
    if len(arrays) == 1:
        return arrays[0].astype(np.float32)

    cf_samples = int(sample_rate * crossfade_ms / 1000)
    result = arrays[0].astype(np.float32)

    for nxt in arrays[1:]:
        nxt = nxt.astype(np.float32)
        if cf_samples > 0 and len(result) >= cf_samples and len(nxt) >= cf_samples:
            fade_out = np.linspace(1.0, 0.0, cf_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, cf_samples, dtype=np.float32)
            # Apply crossfade on the tail of result and head of nxt
            overlap = result[-cf_samples:] * fade_out + nxt[:cf_samples] * fade_in
            result = np.concatenate([result[:-cf_samples], overlap, nxt[cf_samples:]])
        else:
            result = np.concatenate([result, nxt])

    return result


# ---------------------------------------------------------------------------
# Core synthesis function
# ---------------------------------------------------------------------------

def generate_xtts_ro(
    text: str,
    output_path: str = None,
    log=None,
    config: dict = None,
    speed: float = None,
) -> str:
    """
    Generate Romanian XTTS voice for *text*.

    Parameters
    ----------
    text : str
        Romanian text to synthesize (any length — will be chunked automatically).
    output_path : str, optional
        Destination WAV file path.  A temp file is created when None.
    log : callable, optional
        Logging callback ``log(message: str)``.
    config : dict, optional
        Override config values (merged on top of xtts_ro_config.json).
    speed : float, optional
        Playback speed multiplier (0.5 = slower, 1.0 = normal, 2.0 = faster).
        Applied via resampling after synthesis.

    Returns
    -------
    str
        Path to the generated WAV file, or None on failure.
    """
    if not text or not text.strip():
        return None

    cfg = _load_config()
    if config:
        cfg.update(config)

    # Normalize Romanian diacritics
    text = normalize_romanian(text.strip())

    device = _select_device(log=log)

    # Resolve paths
    model_path = cfg.get("model_path", "")
    active_voice = cfg.get("active_voice", "")
    voice_mode = cfg.get("voice_mode", "custom")

    speaker_ref = None
    if voice_mode != "builtin":
        speaker_ref = _get_speaker_ref(cfg.get("speaker_ref_path", ""), log=log, active_voice=active_voice)

    # XTTS v2 is a multi-speaker model: calling it without EITHER a speaker
    # reference (voice cloning) OR a named built-in speaker raises an error
    # inside model.tts()/model.inference().  To make sure a voice is always
    # produced (and never silently falls back to the original video audio),
    # resolve a built-in studio speaker whenever no custom voice is active.
    builtin_speaker_name = None
    if not speaker_ref:
        builtin_speaker_name = cfg.get("builtin_speaker") or DEFAULT_BUILTIN_SPEAKER
        if log:
            log(f"[XTTS RO] No custom voice selected — using built-in speaker: {builtin_speaker_name}")

    language = cfg.get("language", "ro")
    max_chars = int(cfg.get("max_chars_per_chunk", 220))
    crossfade_ms = int(cfg.get("crossfade_ms", 80))
    sample_rate = int(cfg.get("sample_rate", 24000))
    out_sr = int(cfg.get("output_sample_rate", 24000))
    temperature = float(cfg.get("temperature", 0.65))
    top_p = float(cfg.get("top_p", 0.85))
    top_k = int(cfg.get("top_k", 50))
    rep_penalty = float(cfg.get("repetition_penalty", 10.0))

    # Ensure output path
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="xtts_ro_")
        os.close(fd)

    try:
        import numpy as np
        import soundfile as sf

        model = _load_xtts_model(model_path, device, log=log)

        # Build conditioning latents from speaker ref (custom cloned voice)
        if speaker_ref:
            if log:
                log("[XTTS RO] Extracting speaker conditioning latents…")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[speaker_ref]
            )
        else:
            # Voice cloning conditioning is only used for custom voices.  For
            # built-in speakers we rely on the model's own speaker bank via
            # model.tts(..., speaker=builtin_speaker_name) below.
            gpt_cond_latent = None
            speaker_embedding = None

            # Validate that the requested built-in speaker actually exists in
            # THIS checkpoint's speaker bank (names can differ across model
            # versions).  If not, fall back to one that does, instead of
            # letting every chunk raise and silently producing no audio.
            try:
                available_builtin = list(getattr(model.speaker_manager, "speakers", {}).keys())
            except Exception:
                available_builtin = []
            if available_builtin and builtin_speaker_name not in available_builtin:
                fallback_speaker = sorted(available_builtin)[0]
                if log:
                    log(f"[XTTS RO] ⚠ Built-in speaker '{builtin_speaker_name}' not found in this checkpoint — using '{fallback_speaker}'")
                builtin_speaker_name = fallback_speaker
            elif not available_builtin:
                if log:
                    log("[XTTS RO] ⚠ Model has no built-in speaker bank — synthesis may fail without a custom voice")

        # Split text into chunks
        chunks = _split_into_chunks(text, max_chars)
        total = len(chunks)
        if log:
            log(f"[XTTS RO] Text length: {len(text)} chars → {total} chunk(s)")

        chunk_arrays = []
        for i, chunk in enumerate(chunks):
            if log:
                log(f"[XTTS RO] Synthesising chunk {i+1}/{total}: {chunk[:60]}{'…' if len(chunk)>60 else ''}")
            try:
                if gpt_cond_latent is not None:
                    out = model.inference(
                        chunk,
                        language,
                        gpt_cond_latent,
                        speaker_embedding,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=rep_penalty,
                    )
                    wav = out["wav"]
                    # Convert tensor → numpy if needed
                    if hasattr(wav, "cpu"):
                        wav = wav.cpu().numpy()
                    wav = np.array(wav, dtype=np.float32).squeeze()
                else:
                    # Built-in studio speaker — the high-level API resolves the
                    # speaker embedding internally from the model's speaker bank.
                    # NOTE: XTTS v2 is multi-speaker and *requires* a `speaker`
                    # (or `speaker_wav`) argument; builtin_speaker_name is always
                    # set at this point (see resolution above) so this never
                    # silently fails and falls back to the un-replaced voice.
                    wav = model.tts(text=chunk, language=language, speaker=builtin_speaker_name)
                    wav = np.array(wav, dtype=np.float32).squeeze()

                chunk_arrays.append(wav)
            except Exception as chunk_exc:
                if log:
                    log(f"[XTTS RO] ⚠ Chunk {i+1} failed: {chunk_exc}")
                import traceback
                if log:
                    log(traceback.format_exc())

        if not chunk_arrays:
            if log:
                log("[XTTS RO] ❌ All chunks failed — no audio produced")
            return None

        # Concatenate with crossfade
        if log and total > 1:
            log(f"[XTTS RO] Concatenating {len(chunk_arrays)} chunk(s) with {crossfade_ms}ms crossfade…")
        final_wav = _crossfade_concat(chunk_arrays, sample_rate, crossfade_ms)

        # Resample if output sample rate differs
        if out_sr != sample_rate and len(final_wav) > 0:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(out_sr, sample_rate)
                final_wav = resample_poly(final_wav, out_sr // g, sample_rate // g).astype(np.float32)
                if log:
                    log(f"[XTTS RO] Resampled {sample_rate} Hz → {out_sr} Hz")
            except Exception:
                pass  # keep original sample rate

        # Apply speed adjustment via resampling (speed > 1 = faster, < 1 = slower)
        effective_speed = float(speed) if speed is not None else 1.0
        effective_speed = max(0.25, min(4.0, effective_speed))
        if abs(effective_speed - 1.0) > 0.01 and len(final_wav) > 0:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                # To speed up, resample to a lower virtual sample rate then play at out_sr
                # Equivalent: stretch array length by 1/speed
                orig_len = len(final_wav)
                new_len = max(1, int(round(orig_len / effective_speed)))
                # Use resample_poly for quality resampling
                speed_num = int(round(effective_speed * 1000))
                speed_den = 1000
                g = gcd(speed_num, speed_den)
                final_wav = resample_poly(final_wav, speed_den // g, speed_num // g).astype(np.float32)
                if log:
                    log(f"[XTTS RO] Speed {effective_speed:.2f}x applied ({orig_len} → {len(final_wav)} samples)")
            except Exception as speed_exc:
                if log:
                    log(f"[XTTS RO] ⚠ Speed adjustment failed: {speed_exc}")

        # Clip to [-1, 1] to prevent clipping artifacts
        final_wav = np.clip(final_wav, -1.0, 1.0)

        # Save as PCM16 WAV using soundfile (no torchaudio)
        sf.write(output_path, final_wav, out_sr, subtype="PCM_16")

        if log:
            duration = len(final_wav) / out_sr
            log(f"[XTTS RO] ✅ Saved {duration:.2f}s WAV → {output_path}")
        return output_path

    except Exception as exc:
        if log:
            log(f"[XTTS RO] ❌ Generation failed: {exc}")
        import traceback
        if log:
            log(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Romanian XTTS voice generator")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--out", default=None, help="Output WAV path")
    parser.add_argument("--model", default=None, help="Path to XTTS v2 model directory")
    parser.add_argument("--speaker", default=None, help="Path to speaker reference WAV")
    parser.add_argument("--max-chars", type=int, default=None, help="Max chars per chunk")
    parser.add_argument("--crossfade-ms", type=int, default=None, help="Crossfade ms")
    args = parser.parse_args()

    override = {}
    if args.model:
        override["model_path"] = args.model
    if args.speaker:
        override["speaker_ref_path"] = args.speaker
    if args.max_chars:
        override["max_chars_per_chunk"] = args.max_chars
    if args.crossfade_ms is not None:
        override["crossfade_ms"] = args.crossfade_ms

    result = generate_xtts_ro(
        args.text,
        output_path=args.out,
        log=print,
        config=override if override else None,
    )
    if result:
        print(f"Output: {result}")
    else:
        print("Generation failed.")
        raise SystemExit(1)
