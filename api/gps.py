"""
NxV - GPS Location API
api/gps.py

Receives GPS coordinates from the owner's phone.
Calculates distance from home and automatically sets user_away flag.

How it works:
  1. Phone visits: http://<PI_IP>:5000/gps?lat=XX.XXX&lon=YY.YYY
  2. NxV calculates distance from home coordinates
  3. If distance > AWAY_THRESHOLD → user_away = True
  4. If distance < HOME_THRESHOLD → user_away = False
  5. Escalation engine adjusts automatically

Phone automation options (no app needed yet):
  - iPhone: use Shortcuts app to ping the URL every 30 mins
  - Android: use Tasker or MacroDroid to ping the URL automatically
  - Or: just bookmark the URL and tap it when leaving/arriving
"""

import math
from datetime import datetime


# ── Your home coordinates ─────────────────────────────────────────────────────
# Get these from Google Maps — right click your house → copy coordinates
HOME_LAT = 39.204417  # ← replace with your home latitude  e.g. 39.2034
HOME_LON = -76.779902   # ← replace with your home longitude e.g. -76.8621

# ── Distance thresholds ───────────────────────────────────────────────────────
AWAY_THRESHOLD_KM  = 0.5   # further than 500m = AWAY
HOME_THRESHOLD_KM  = 0.3   # closer than 300m  = HOME
# The gap between thresholds prevents flapping when you're on the boundary

# ── Escalation distance scaling ───────────────────────────────────────────────
# How far away changes escalation behavior
NEARBY_KM      = 2.0    # under 2km   → normal escalation
FAR_KM         = 20.0   # 2-20km      → bump escalation one level
VERY_FAR_KM    = 100.0  # 20-100km    → bump two levels + contact immediately
ACROSS_COUNTRY = 100.0  # 100km+      → max escalation, contact + police prompt


class GPSTracker:
    """
    Tracks owner's GPS location and computes distance from home.

    Usage:
        tracker = GPSTracker()
        result  = tracker.update(lat, lon)
        print(result['user_away'], result['distance_km'])
    """

    def __init__(self):
        self.last_lat      = None
        self.last_lon      = None
        self.last_update   = None
        self.distance_km   = None
        self.user_away     = False
        self.distance_zone = "unknown"   # nearby / far / very_far / across_country

        if HOME_LAT == 0.0 and HOME_LON == 0.0:
            print("[NxV GPS] WARNING: Home coordinates not set!")
            print("  Edit api/gps.py and set HOME_LAT and HOME_LON")
        else:
            print(f"[NxV GPS] Home set to ({HOME_LAT}, {HOME_LON})")

    def update(self, lat: float, lon: float) -> dict:
        """
        Receive new GPS coordinates from the phone.
        Returns a result dict with distance, away status and zone.
        """
        self.last_lat    = lat
        self.last_lon    = lon
        self.last_update = datetime.now()

        self.distance_km   = self._haversine(lat, lon, HOME_LAT, HOME_LON)
        self.distance_zone = self._get_zone(self.distance_km)

        # Update user_away with hysteresis to prevent flapping
        if self.distance_km > AWAY_THRESHOLD_KM:
            self.user_away = True
        elif self.distance_km < HOME_THRESHOLD_KM:
            self.user_away = False
        # Between thresholds — keep current state

        result = {
            "lat"          : lat,
            "lon"          : lon,
            "distance_km"  : round(self.distance_km, 3),
            "distance_m"   : round(self.distance_km * 1000),
            "user_away"    : self.user_away,
            "zone"         : self.distance_zone,
            "updated_at"   : self.last_update.strftime("%H:%M:%S"),
        }

        print(f"[NxV GPS] {result['distance_m']}m from home | "
              f"{'AWAY' if self.user_away else 'HOME'} | zone:{self.distance_zone}")

        return result

    def get_escalation_modifier(self) -> str:
        """
        Returns escalation modifier string based on distance zone.
        Used by threat score engine to adjust escalation.
        """
        if not self.user_away:
            return "home"
        return self.distance_zone

    def is_stale(self, max_age_minutes: int = 10) -> bool:
        """Returns True if GPS hasn't been updated recently."""
        if self.last_update is None:
            return True
        age = (datetime.now() - self.last_update).total_seconds() / 60
        return age > max_age_minutes

    # ── Math ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _haversine(lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates in kilometers.
        Uses the Haversine formula — accurate for short distances.
        """
        R = 6371   # Earth radius in km

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)

        c = 2 * math.asin(math.sqrt(a))
        return R * c

    @staticmethod
    def _get_zone(distance_km: float) -> str:
        if distance_km <= NEARBY_KM:
            return "nearby"
        elif distance_km <= FAR_KM:
            return "far"
        elif distance_km <= VERY_FAR_KM:
            return "very_far"
        else:
            return "across_country"
