"""


Plays audio warnings through the Pi's audio output when a threat is detected.

Modes:
  NOTIFY    → silent (no deterrent)
  ALERT     → voice warning ("You are being recorded...")
  EMERGENCY → loud siren

Requires:
  - Speaker or buzzer connected to Pi audio jack or GPIO
  - pygame for audio playback
    pip install pygame --break-system-packages

Audio files expected in:
  assets/sounds/
    ├── warning.mp3    (voice warning)
    └── siren.mp3      (emergency siren)

If no audio files exist, falls back to Pi terminal bell.
"""

import time
import os


# ── Try importing pygame ───────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False
    print("[NxV Deterrent] pygame not available — using terminal bell fallback.")


# ── Audio file paths ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND_DIR    = os.path.join(BASE_DIR, "assets", "sounds")
WARNING_FILE = os.path.join(SOUND_DIR, "warning.mp3")
SIREN_FILE   = os.path.join(SOUND_DIR, "siren.mp3")

# ── Cooldown ──────────────────────────────────────────────────────────────────
DETERRENT_COOLDOWN = 30   # seconds between deterrent triggers


class Deterrent:
    """
    Plays audio deterrent based on escalation level.

    Usage:
        deterrent = Deterrent()
        deterrent.trigger("ALERT")      # plays voice warning
        deterrent.trigger("EMERGENCY")  # plays siren
    """

    def __init__(self):
        self._last_triggered = 0

    def trigger(self, escalation: str) -> bool:
        """
        Trigger the deterrent for the given escalation level.
        Returns True if played, False if on cooldown or silent mode.
        """
        # NOTIFY = silent — no deterrent
        if escalation == "NOTIFY":
            return False

        # Rate limit
        if time.time() - self._last_triggered < DETERRENT_COOLDOWN:
            return False

        self._last_triggered = time.time()

        if escalation == "EMERGENCY":
            return self._play_siren()
        elif escalation == "ALERT":
            return self._play_warning()

        return False

    def stop(self):
        """Stop any playing audio."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    # ── Audio playback ─────────────────────────────────────────────────────────

    def _play_warning(self) -> bool:
        """Play voice warning audio."""
        print("[NxV Deterrent] Playing voice warning...")

        if PYGAME_AVAILABLE and os.path.exists(WARNING_FILE):
            try:
                pygame.mixer.music.load(WARNING_FILE)
                pygame.mixer.music.play()
                return True
            except Exception as e:
                print(f"[NxV Deterrent] Audio error: {e}")

        # Fallback — terminal bell x3
        self._bell(3)
        return True

    def _play_siren(self) -> bool:
        """Play emergency siren audio."""
        print("[NxV Deterrent] Playing EMERGENCY siren...")

        if PYGAME_AVAILABLE and os.path.exists(SIREN_FILE):
            try:
                pygame.mixer.music.load(SIREN_FILE)
                pygame.mixer.music.play(loops=3)
                return True
            except Exception as e:
                print(f"[NxV Deterrent] Audio error: {e}")

        # Fallback — terminal bell x10
        self._bell(10)
        return True

    def _bell(self, count: int):
        """Terminal bell fallback."""
        for _ in range(count):
            print("\a", end="", flush=True)
            time.sleep(0.3)
