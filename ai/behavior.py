"""
NxV - Behavior Analyzer
ai/behavior.py

Takes Person objects from the tracker and scores their behavior.
Outputs a 0-100 behavior score per person.

Signals detected:
  - Loitering    : staying in one zone too long
  - Pacing       : back and forth movement
  - Scanning     : slow movement while changing direction frequently
  - Fast approach: moving quickly toward the camera
"""

import numpy as np


# ── Tunable thresholds ────────────────────────────────────────────────────────
LOITER_SECONDS      = 15.0   # dwell time before loitering flag
LOITER_DISPLACEMENT = 60.0   # max displacement to still count as loitering
FAST_APPROACH_SPEED = 120.0  # pixels/sec toward camera = fast approach
SCAN_MIN_DIRS       = 4      # direction changes needed to flag scanning


class BehaviorAnalyzer:
    """
    Analyzes behavior of tracked persons and returns a score per person.

    Usage:
        analyzer = BehaviorAnalyzer()
        scores   = analyzer.analyze(persons)

        # scores is a dict: { person_id: BehaviorResult }
    """

    def analyze(self, persons: list) -> dict:
        """
        persons — list of Person objects from PersonTracker.update()
        Returns dict of { person_id: BehaviorResult }
        """
        results = {}
        for person in persons:
            result = self._score_person(person)
            results[person.id] = result
        return results

    def _score_person(self, person) -> "BehaviorResult":
        score   = 0
        flags   = []

        # ── Loitering ─────────────────────────────────────────────────────────
        # Person has been in scene a long time but hasn't moved much
        if (person.dwell_time >= LOITER_SECONDS and
                person.displacement <= LOITER_DISPLACEMENT):
            loiter_score = min(40, int((person.dwell_time / LOITER_SECONDS) * 20))
            score += loiter_score
            flags.append(f"loitering({person.dwell_time:.0f}s)")

        # ── Pacing ────────────────────────────────────────────────────────────
        if person.is_pacing():
            score += 25
            flags.append("pacing")

        # ── Scanning ──────────────────────────────────────────────────────────
        # Slow overall speed but many direction changes = scanning behavior
        if self._is_scanning(person):
            score += 20
            flags.append("scanning")

        # ── Fast approach ─────────────────────────────────────────────────────
        # Moving quickly AND getting larger in frame (toward camera)
        if self._is_approaching(person):
            score += 30
            flags.append("fast_approach")

        score = min(100, score)
        return BehaviorResult(person.id, score, flags)

    def _is_scanning(self, person) -> bool:
        """
        Slow movement but frequent direction changes = scanning/casing.
        """
        if len(person.history) < 12:
            return False
        if person.speed > 60:
            return False   # moving too fast to be scanning

        direction_changes = 0
        for i in range(2, len(person.history)):
            dx1 = person.history[i-1][0] - person.history[i-2][0]
            dx2 = person.history[i][0]   - person.history[i-1][0]
            dy1 = person.history[i-1][1] - person.history[i-2][1]
            dy2 = person.history[i][1]   - person.history[i-1][1]

            # Count as a direction change if X or Y flips sign
            x_flip = (dx1 * dx2 < 0) and abs(dx1) > 2 and abs(dx2) > 2
            y_flip = (dy1 * dy2 < 0) and abs(dy1) > 2 and abs(dy2) > 2
            if x_flip or y_flip:
                direction_changes += 1

        return direction_changes >= SCAN_MIN_DIRS

    def _is_approaching(self, person) -> bool:
        """
        Person is moving fast AND bbox is growing (getting closer to camera).
        """
        if person.speed < FAST_APPROACH_SPEED:
            return False
        if len(person.history) < 5:
            return False

        # Check if moving toward camera (y increasing = moving down = closer)
        recent = person.history[-5:]
        dy = recent[-1][1] - recent[0][1]
        return dy > 10   # moving downward in frame = approaching


class BehaviorResult:
    """
    Result for a single person from the behavior analyzer.
    """
    def __init__(self, person_id: int, score: int, flags: list):
        self.person_id = person_id
        self.score     = score      # 0-100
        self.flags     = flags      # list of string flags

    @property
    def label(self) -> str:
        if not self.flags:
            return "normal"
        return ", ".join(self.flags)

    @property
    def risk_level(self) -> str:
        if self.score >= 60:
            return "HIGH"
        elif self.score >= 30:
            return "MEDIUM"
        return "LOW"

    def __repr__(self):
        return (f"BehaviorResult(id={self.person_id}, "
                f"score={self.score}, risk={self.risk_level}, "
                f"flags={self.flags})")
