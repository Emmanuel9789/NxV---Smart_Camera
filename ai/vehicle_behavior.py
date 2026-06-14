"""
Analyzes tracked vehicles for suspicious behavior patterns.

Behaviors detected:
    SLOW_ROLL   — vehicle moving unusually slowly past the property
    PARKED      — vehicle stopped outside for an extended time
    CIRCLING    — same vehicle passes multiple times
    BLOCKING    — vehicle stopped in the driveway zone

Scoring philosophy:
    A car driving by normally → score 0, no alert
    A car parked 10+ minutes → score 30, NOTIFY
    A car doing slow roll at night → score 45, NOTIFY-ALERT
    Same car circles 3 times → score 60, ALERT
    Vehicle + persons loitering → handled by threat engine
"""

import time
from dataclasses import dataclass, field
from tracking.vehicle_tracker import TrackedVehicle


@dataclass
class VehicleBehaviorResult:
    """
    Result of analyzing one vehicle's behavior.

    vehicle_id  : which vehicle
    score       : 0-100 threat contribution
    flags       : human-readable list of what triggered
    behavior    : primary behavior label
    """
    vehicle_id : int
    score      : int
    flags      : list[str]
    behavior   : str = 'normal'


class VehicleBehaviorAnalyzer:
    """
    Analyzes vehicle behavior patterns over time.

    Uses:
        Hash map  : _pass_counts {vehicle_id: count}   O(1) lookup
        Hash map  : _last_direction {id: direction}    O(1) lookup
        Sliding window: path_history from tracker      O(1) amortized
    """

    # Thresholds — tune these based on your camera's pixel density
    # At 256x192 resolution:
    #   Slow roll: moving less than 2 pixels per frame on average
    #   Parked: dwell time > 120 seconds with minimal movement
    #   Circling: same vehicle seen 3+ separate times

    SLOW_ROLL_SPEED_THRESHOLD = 2.5    # pixels per frame
    PARKED_DWELL_THRESHOLD    = 120    # seconds before flagging
    CIRCLING_COUNT_THRESHOLD  = 3      # passes before flagging
    BLOCKING_DWELL_THRESHOLD  = 30     # seconds if in door zone

    def __init__(self):
        # Hash map: vehicle_id → how many times it has come and gone
        # A vehicle that stays continuously is count=1
        # A vehicle that leaves and comes back is count=2, etc.
        self._pass_counts   : dict[int, int]   = {}

        # Track whether vehicle was lost last frame
        # so we can increment pass count on reappearance
        self._was_lost      : dict[int, bool]  = {}

        # Score history for smoothing — prevents one-frame spikes
        self._score_history : dict[int, list]  = {}

        # Door zone — same as DOOR_ZONE in stream.py
        # If None, blocking detection is disabled
        self._door_zone = None

    def set_door_zone(self, zone):
        """Set the door zone for blocking detection. (x1,y1,x2,y2)"""
        self._door_zone = zone

    def analyze(self, vehicles: list[TrackedVehicle]) -> dict[int, VehicleBehaviorResult]:
        """
        Analyze all tracked vehicles and return behavior results.

        Returns dict: {vehicle_id: VehicleBehaviorResult}
        All lookups are O(1) via hash maps.
        """
        results = {}

        for vehicle in vehicles:
            if vehicle.lost:
                # Vehicle disappeared — may come back (circling)
                self._was_lost[vehicle.vehicle_id] = True
                continue

            vid = vehicle.vehicle_id

            # ── Update pass count ──────────────────────────────────
            # If vehicle was previously lost and is now back,
            # it's a new pass — increment counter
            if self._was_lost.get(vid, False):
                self._pass_counts[vid] = self._pass_counts.get(vid, 1) + 1
                self._was_lost[vid] = False
            elif vid not in self._pass_counts:
                self._pass_counts[vid] = 1

            # ── Analyze behaviors ──────────────────────────────────
            score = 0
            flags = []
            behavior = 'normal'

            # 1. SLOW ROLL — calculate movement speed from path history
            speed = self._calculate_speed(vehicle)
            is_moving = speed > 0.5    # more than half a pixel per frame

            if is_moving and speed < self.SLOW_ROLL_SPEED_THRESHOLD:
                if vehicle.dwell_time > 3.0:  # ignore first few frames
                    score    += 25
                    behavior  = 'slow_roll'
                    flags.append(f'slow_roll(speed={speed:.1f}px/f)')

            # 2. PARKED — stopped for a long time
            if not is_moving and vehicle.dwell_time > self.PARKED_DWELL_THRESHOLD:
                minutes = vehicle.dwell_time / 60
                score  += min(40, int(minutes * 3))   # +3 per minute, max 40
                behavior = 'parked'
                flags.append(f'parked({minutes:.1f}min)')

            # 3. CIRCLING — same vehicle passed multiple times
            pass_count = self._pass_counts.get(vid, 1)
            if pass_count >= self.CIRCLING_COUNT_THRESHOLD:
                circle_score = min(50, (pass_count - 2) * 15)
                score       += circle_score
                behavior     = 'circling'
                flags.append(f'circling(passes={pass_count})')

            # 4. BLOCKING — vehicle stopped in door zone
            if (self._door_zone and not is_moving
                    and vehicle.dwell_time > self.BLOCKING_DWELL_THRESHOLD):
                if self._in_door_zone(vehicle.center):
                    score    += 35
                    behavior  = 'blocking'
                    flags.append(f'blocking_driveway({vehicle.dwell_time:.0f}s)')

            # 5. Night multiplier — suspicious vehicle behavior
            # is more concerning at night
            # (actual time check happens in threat_score.py)

            # ── Smooth score over last 5 frames ───────────────────
            # Prevents single-frame detection spikes from causing alerts
            if vid not in self._score_history:
                self._score_history[vid] = []
            self._score_history[vid].append(score)
            if len(self._score_history[vid]) > 5:
                self._score_history[vid].pop(0)
            smoothed_score = int(sum(self._score_history[vid])
                                 / len(self._score_history[vid]))

            results[vid] = VehicleBehaviorResult(
                vehicle_id = vid,
                score      = min(100, smoothed_score),
                flags      = flags,
                behavior   = behavior,
            )

        return results

    def _calculate_speed(self, vehicle: TrackedVehicle) -> float:
        """
        Calculate average movement speed from path history.

        Uses the sliding window of (cx, cy) positions.
        Speed = average Euclidean distance between consecutive positions.

        Returns pixels per frame (float).
        Complexity: O(w) where w = window size = 90, effectively O(1).
        """
        history = vehicle.path_history
        if len(history) < 2:
            return 0.0

        # Use last 10 positions for speed calculation
        # Less noisy than using all 90
        recent = history[-10:]
        distances = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            distances.append((dx**2 + dy**2) ** 0.5)

        return sum(distances) / len(distances) if distances else 0.0

    def _in_door_zone(self, center: tuple) -> bool:
        """Check if vehicle center is inside the door zone."""
        if not self._door_zone:
            return False
        x1, y1, x2, y2 = self._door_zone
        cx, cy = center
        return x1 <= cx <= x2 and y1 <= cy <= y2
