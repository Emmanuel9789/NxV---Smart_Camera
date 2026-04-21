"""
NxV - Violence Detector
ai/violence.py

Analyzes interactions between tracked persons to detect violent behavior.
Lightweight — designed to run on Pi without killing CPU.

Signals detected:
  - Close contact    : two people very close together
  - Rapid movement   : sudden speed spike (lunge, strike)
  - Fall detection   : person bbox suddenly drops / expands horizontally
  - Drag detection   : one person moving fast while another barely moves
  - Crowd surge      : multiple people converging rapidly on one point
"""

import numpy as np


# ── Tunable thresholds ────────────────────────────────────────────────────────
CLOSE_CONTACT_DIST  = 80    # pixels between centroids = close contact
RAPID_MOVE_SPEED    = 200   # px/sec spike = rapid/aggressive movement
FALL_RATIO          = 1.8   # width/height ratio — person lying down
FALL_SPEED_DROP     = 80    # vertical speed downward before fall
DRAG_SPEED_DIFF     = 150   # speed difference between two close persons
SURGE_MIN_PERSONS   = 3     # min persons needed to detect crowd surge
SURGE_CONVERGE_DIST = 120   # all surging toward a point within this radius


class ViolenceDetector:
    """
    Detects violent interactions between tracked persons.

    Usage:
        detector = ViolenceDetector()
        result   = detector.analyze(persons)
        print(result.score, result.flags)
    """

    def analyze(self, persons: list) -> "ViolenceResult":
        """
        persons — list of Person objects from PersonTracker.update()
        Returns a single ViolenceResult for the whole scene.
        """
        score = 0
        flags = []

        if len(persons) == 0:
            return ViolenceResult(0, [])

        # ── Single-person signals ─────────────────────────────────────────────
        for person in persons:

            # Rapid movement spike
            if person.speed >= RAPID_MOVE_SPEED:
                score += 25
                flags.append(f"rapid_move(P{person.id})")

            # Fall detection — bbox becomes wider than tall
            x, y, w, h = person.bbox
            if h > 0 and (w / h) >= FALL_RATIO:
                score += 30
                flags.append(f"fall(P{person.id})")

            # Vertical drop speed (falling)
            if len(person.history) >= 3:
                dy = person.history[-1][1] - person.history[-3][1]
                if dy >= FALL_SPEED_DROP:
                    score += 20
                    flags.append(f"drop(P{person.id})")

        # ── Multi-person signals ──────────────────────────────────────────────
        if len(persons) >= 2:
            pairs = self._get_pairs(persons)

            for pa, pb in pairs:
                dist = self._centroid_dist(pa, pb)

                # Close contact
                if dist <= CLOSE_CONTACT_DIST:
                    score += 20
                    flags.append(f"close_contact(P{pa.id},P{pb.id})")

                    # Drag: one moving fast, one slow while close together
                    speed_diff = abs(pa.speed - pb.speed)
                    if speed_diff >= DRAG_SPEED_DIFF:
                        score += 25
                        faster = pa.id if pa.speed > pb.speed else pb.id
                        flags.append(f"drag(P{faster})")

            # Crowd surge — 3+ people converging on same point fast
            if len(persons) >= SURGE_MIN_PERSONS:
                if self._is_surging(persons):
                    score += 35
                    flags.append("crowd_surge")

        score = min(100, score)
        return ViolenceResult(score, flags)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_pairs(self, persons: list) -> list:
        """Return all unique pairs of persons."""
        pairs = []
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                pairs.append((persons[i], persons[j]))
        return pairs

    @staticmethod
    def _centroid_dist(pa, pb) -> float:
        ax, ay = pa.centroid
        bx, by = pb.centroid
        return float(np.sqrt((ax - bx)**2 + (ay - by)**2))

    def _is_surging(self, persons: list) -> bool:
        """
        Check if 3+ persons are all moving fast and converging
        toward a common point (average of their centroids).
        """
        fast_persons = [p for p in persons if p.speed >= 80]
        if len(fast_persons) < SURGE_MIN_PERSONS:
            return False

        # Find average centroid
        avg_x = np.mean([p.centroid[0] for p in fast_persons])
        avg_y = np.mean([p.centroid[1] for p in fast_persons])

        # Check all fast persons are within surge radius of that point
        all_close = all(
            np.sqrt((p.centroid[0] - avg_x)**2 + (p.centroid[1] - avg_y)**2)
            <= SURGE_CONVERGE_DIST
            for p in fast_persons
        )
        return all_close


class ViolenceResult:
    """
    Result for the whole scene from the violence detector.
    """
    def __init__(self, score: int, flags: list):
        self.score = score    # 0-100
        self.flags = flags    # list of string flags

    @property
    def label(self) -> str:
        if not self.flags:
            return "peaceful"
        return ", ".join(self.flags)

    @property
    def risk_level(self) -> str:
        if self.score >= 60:
            return "HIGH"
        elif self.score >= 30:
            return "MEDIUM"
        return "LOW"

    def __repr__(self):
        return (f"ViolenceResult(score={self.score}, "
                f"risk={self.risk_level}, flags={self.flags})")
