"""
Detects delivery scenarios at the door:
  Delivery trucks (UPS, FedEx, Amazon, USPS, DHL)
  Person carrying a package/box
  Person at door who leaves quickly (< 90 seconds)
  Time-of-day context (daytime = likely delivery)

Outputs:
  DeliveryResult with type, confidence, and label
"""

import cv2
import numpy as np
import time
from datetime import datetime


# Delivery truck colors (BGR)
# Each carrier has a distinctive vehicle color
CARRIER_COLORS = {
    "UPS":    {"lower": np.array([0,  50,  80]),   "upper": np.array([20, 180, 180]),  "name": "UPS"},
    "FedEx":  {"lower": np.array([100, 50, 150]),  "upper": np.array([130, 255, 255]), "name": "FedEx"},
    "Amazon": {"lower": np.array([15,  50, 150]),  "upper": np.array([35, 255, 255]),  "name": "Amazon"},
    "USPS":   {"lower": np.array([100, 100, 100]), "upper": np.array([130, 255, 255]), "name": "USPS"},
    "DHL":    {"lower": np.array([5,   100, 150]), "upper": np.array([20, 255, 255]),  "name": "DHL"},
}

# Delivery hours — outside these hours delivery is suspicious
DELIVERY_HOURS = (7, 21)   # 7am to 9pm

# How long a delivery person typically stays (seconds)
MAX_DELIVERY_DWELL = 120    # 2 minutes max for a normal delivery


class DeliveryDetector:
    """
    Detects delivery scenarios using color analysis,
    person dwell time, and time-of-day context.
    """

    def __init__(self):
        self._door_visitors = {}   # person_id → first_seen timestamp
        print("[NxV Delivery] Detector ready")

    def analyze(self, frame, persons: list,
                motion_boxes: list) -> "DeliveryResult":
        """
        Analyze frame for delivery scenarios.

        Returns DeliveryResult with:
          is_delivery  — True if delivery detected
          carrier      — carrier name or "Unknown carrier"
          confidence   — 0.0 to 1.0
          label        — human readable description
          tier         — notification tier (0 = silent, 1 = friendly)
        """
        
        hour = datetime.now().hour
        is_daytime = DELIVERY_HOURS[0] <= hour <= DELIVERY_HOURS[1]

        #  Check for delivery truck colors 
        carrier, truck_conf = self._detect_truck(frame)

        #  Check person dwell time (quick visit = delivery) 
        quick_visit = False
        for person in persons:
            pid = person.id
            if pid not in self._door_visitors:
                self._door_visitors[pid] = time.time()

            dwell = time.time() - self._door_visitors[pid]

            # Person left quickly = delivery
            if person.dwell_time > 5 and dwell < MAX_DELIVERY_DWELL:
                quick_visit = True

        # Clean up old visitors
        now = time.time()
        self._door_visitors = {
            pid: t for pid, t in self._door_visitors.items()
            if now - t < 300
        }

        # Scoring 
        confidence = 0.0

        if carrier:
            confidence += 0.6   # strong signal — known carrier color
        if is_daytime:
            confidence += 0.2   # daytime = more likely delivery
        if quick_visit:
            confidence += 0.2   # quick visit = delivery pattern
        if persons:
            confidence += 0.1   # someone is actually there

        confidence = min(1.0, confidence)
        is_delivery = confidence >= 0.4

        # Build label 
        if is_delivery:
            if carrier:
                label = f"{carrier} delivery"
            elif persons:
                label = "Delivery person at door"
            else:
                label = "Possible delivery"
        else:
            label = "No delivery detected"

        return DeliveryResult(
            is_delivery = is_delivery,
            carrier     = carrier or ("delivery person" if is_delivery else None),
            confidence  = round(confidence, 2),
            label       = label,
            is_daytime  = is_daytime,
        )

    def _detect_truck(self, frame) -> tuple:
        """
        Detect delivery truck by color signature.
        Returns (carrier_name, confidence) or (None, 0.0)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Check lower 2/3 of frame (truck body area)
        roi = hsv[h//3:, :]

        best_carrier  = None
        best_coverage = 0.0

        for carrier, colors in CARRIER_COLORS.items():
            mask     = cv2.inRange(roi, colors["lower"], colors["upper"])
            coverage = np.sum(mask > 0) / mask.size

            if coverage > best_coverage and coverage > 0.08:
                best_coverage = coverage
                best_carrier  = carrier

        if best_carrier:
            return best_carrier, min(1.0, best_coverage * 5)
        return None, 0.0


class DeliveryResult:
    def __init__(self, is_delivery: bool, carrier: str,
                 confidence: float, label: str, is_daytime: bool):
        self.is_delivery = is_delivery
        self.carrier     = carrier
        self.confidence  = confidence
        self.label       = label
        self.is_daytime  = is_daytime

    @property
    def notification_label(self) -> str:
        if self.carrier and self.carrier not in ("delivery person",):
            return f"{self.carrier} delivery at your door"
        return "Delivery person at your door"

    def __repr__(self):
        return (f"DeliveryResult(delivery={self.is_delivery}, "
                f"carrier={self.carrier}, conf={self.confidence})")
