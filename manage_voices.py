"""
manage_voices.py - Gestionare voci pentru XTTS RO

Comenzi disponibile:
    python manage_voices.py --list
        Afiseaza toate vocile disponibile si care este activa.

    python manage_voices.py --add "C:\\cale\\catre\\voce.wav" --name "maria"
        Copiaza fisierul WAV in folderul voices/ si il salveaza ca "maria".

    python manage_voices.py --select "maria"
        Seteaza "maria" ca voce activa in xtts_ro_config.json.

    python manage_voices.py --delete "maria"
        Sterge vocea "maria" din folderul voices/.

    python manage_voices.py --test "maria"
        Genereaza un fisier audio scurt pentru a testa vocea.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VOICES_DIR = SCRIPT_DIR / "voices"
CONFIG_PATH = SCRIPT_DIR / "xtts_ro_config.json"

DEFAULT_CONFIG = {
    "_comment": "Romanian XTTS configuration — edit paths and tuning params here.",
    "model_path": "",
    "speaker_ref_path": "",
    "active_voice": "",
    "language": "ro",
    "max_chars_per_chunk": 220,
    "crossfade_ms": 80,
    "temperature": 0.65,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 10.0,
    "sample_rate": 24000,
    "output_sample_rate": 24000
}


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def get_voices():
    """Returneaza lista de voci disponibile (fisiere .wav din voices/)."""
    VOICES_DIR.mkdir(exist_ok=True)
    voices = []
    for f in sorted(VOICES_DIR.iterdir()):
        if f.suffix.lower() == ".wav":
            voices.append(f.stem)
    return voices


def list_voices():
    cfg = load_config()
    active = cfg.get("active_voice", "")
    voices = get_voices()

    print("=" * 50)
    print(" Voci disponibile:")
    print("=" * 50)
    if not voices:
        print("  (nicio voce adaugata)")
        print()
        print("Adauga o voce cu:")
        print('  python manage_voices.py --add "C:\\cale\\voce.wav" --name "nume_voce"')
    else:
        for v in voices:
            marker = " <<< ACTIVA" if v == active else ""
            print(f"  - {v}{marker}")
    print()
    if active:
        print(f"Vocea activa: {active}")
    else:
        print("Nicio voce selectata. Foloseste --select pentru a alege una.")
    print()


def add_voice(wav_path: str, name: str):
    """Copiaza un fisier WAV in folderul voices/ cu numele specificat."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        print(f"EROARE: Fisierul nu exista: {wav_path}")
        sys.exit(1)
    if wav_path.suffix.lower() != ".wav":
        print(f"EROARE: Fisierul trebuie sa fie .wav (ai dat: {wav_path.suffix})")
        sys.exit(1)

    # Curata numele (fara spatii/caractere speciale)
    safe_name = name.strip().replace(" ", "_")
    dest = VOICES_DIR / f"{safe_name}.wav"
    VOICES_DIR.mkdir(exist_ok=True)

    shutil.copy2(str(wav_path), str(dest))
    print(f"Vocea '{safe_name}' a fost adaugata: {dest}")

    # Daca e prima voce, o selectam automat
    cfg = load_config()
    if not cfg.get("active_voice"):
        cfg["active_voice"] = safe_name
        cfg["speaker_ref_path"] = str(dest)
        save_config(cfg)
        print(f"Vocea '{safe_name}' a fost setata automat ca voce activa.")
    else:
        print(f"Pentru a o activa: python manage_voices.py --select \"{safe_name}\"")
    print()


def select_voice(name: str):
    """Seteaza vocea activa in config."""
    voices = get_voices()
    if name not in voices:
        print(f"EROARE: Vocea '{name}' nu exista.")
        print(f"Voci disponibile: {', '.join(voices) if voices else '(niciuna)'}")
        sys.exit(1)

    cfg = load_config()
    cfg["active_voice"] = name
    cfg["speaker_ref_path"] = str(VOICES_DIR / f"{name}.wav")
    save_config(cfg)
    print(f"Vocea activa a fost schimbata la: {name}")
    print(f"speaker_ref_path = {cfg['speaker_ref_path']}")
    print()


def delete_voice(name: str):
    """Sterge o voce din folderul voices/."""
    voice_file = VOICES_DIR / f"{name}.wav"
    if not voice_file.exists():
        print(f"EROARE: Vocea '{name}' nu exista.")
        sys.exit(1)

    confirm = input(f"Esti sigur ca vrei sa stergi vocea '{name}'? (da/nu): ").strip().lower()
    if confirm not in ("da", "d", "yes", "y"):
        print("Anulat.")
        return

    voice_file.unlink()
    print(f"Vocea '{name}' a fost stearsa.")

    # Daca era activa, reseteaza
    cfg = load_config()
    if cfg.get("active_voice") == name:
        cfg["active_voice"] = ""
        cfg["speaker_ref_path"] = ""
        save_config(cfg)
        print("Vocea activa a fost resetata. Selecteaza alta voce cu --select.")

    # Sugereaza o alta voce
    voices = get_voices()
    if voices:
        print(f"Voci ramase: {', '.join(voices)}")
    print()


def test_voice(name: str):
    """Genereaza un clip audio scurt pentru a testa vocea."""
    voices = get_voices()
    if name not in voices:
        print(f"EROARE: Vocea '{name}' nu exista.")
        sys.exit(1)

    voice_wav = str(VOICES_DIR / f"{name}.wav")
    output_path = str(SCRIPT_DIR / f"test_{name}.wav")
    test_text = "Salut! Aceasta este o voce de test pentru sistemul XTTS Romanian."

    print(f"Testez vocea '{name}'...")
    print(f"Text: {test_text}")
    print()

    try:
        from xtts_ro import generate_xtts_ro
        result = generate_xtts_ro(
            test_text,
            output_path=output_path,
            log=print,
            config={"speaker_ref_path": voice_wav}
        )
        if result:
            print(f"\nTest reusit! Fisier generat: {result}")
        else:
            print("\nTest esuat.")
    except ImportError:
        print("EROARE: xtts_ro.py nu a fost gasit. Asigura-te ca esti in folderul corect.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Gestionare voci XTTS RO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemple:
  python manage_voices.py --list
  python manage_voices.py --add "C:\\Users\\Trofin\\voce.wav" --name "trofin"
  python manage_voices.py --select "trofin"
  python manage_voices.py --test "trofin"
  python manage_voices.py --delete "trofin"
"""
    )
    parser.add_argument("--list", action="store_true", help="Afiseaza vocile disponibile")
    parser.add_argument("--add", metavar="WAV_PATH", help="Adauga o voce noua (cale catre fisier .wav)")
    parser.add_argument("--name", metavar="NUME", help="Numele vocii (folosit cu --add)")
    parser.add_argument("--select", metavar="NUME", help="Selecteaza vocea activa")
    parser.add_argument("--delete", metavar="NUME", help="Sterge o voce")
    parser.add_argument("--test", metavar="NUME", help="Testeaza o voce")

    args = parser.parse_args()

    if args.list:
        list_voices()
    elif args.add:
        if not args.name:
            print("EROARE: Trebuie sa specifici --name cand folosesti --add.")
            print('Exemplu: python manage_voices.py --add "voce.wav" --name "maria"')
            sys.exit(1)
        add_voice(args.add, args.name)
    elif args.select:
        select_voice(args.select)
    elif args.delete:
        delete_voice(args.delete)
    elif args.test:
        test_voice(args.test)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
