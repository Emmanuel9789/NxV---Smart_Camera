"""
Manages all 5 notification tiers:

  TIER 0 — Silent log (always saved, never notified unless user opts in)
    → Motion history, cars passing, animals
    → Toggle: user can enable notifications for this

  TIER 1 — Friendly notification (user can toggle OFF)
    → Delivery detected
    → Known trusted face arrived
    → Visitor at door (non-threatening, quick visit)
    → Package dropped off

  TIER 2 — NOTIFY (always on, cannot disable)
    → Suspicious loitering
    → Unknown person at night
    → Masked person

  TIER 3 — ALERT (always on)
    → Weapon detected
    → Known dangerous person
    → Break-in attempt

  TIER 4 — EMERGENCY (always on, full call chain)
    → Weapon aimed at camera
    → Active break-in
    → Violence detected
"""

from datetime import datetime


# Tier definitions 
TIERS = { #Dictionary
    0: {
        "name"        : "SILENT",
        "label"       : "Motion log",
        "can_disable" : True,
        "can_enable"  : True,   # user CAN opt in to get notified
        "default_on"  : False,
        "color"       : "gray",
        "escalation"  : "NONE",
    },
    1: {
        "name"        : "FRIENDLY",
        "label"       : "Friendly alert",
        "can_disable" : True,
        "can_enable"  : True,
        "default_on"  : True,
        "color"       : "blue",
        "escalation"  : "NOTIFY",
    },
    2: {
        "name"        : "NOTIFY",
        "label"       : "Security alert",
        "can_disable" : False,   # CANNOT be turned off
        "can_enable"  : True,
        "default_on"  : True,
        "color"       : "yellow",
        "escalation"  : "NOTIFY",
    },
    3: {
        "name"        : "ALERT",
        "label"       : "Threat alert",
        "can_disable" : False,
        "can_enable"  : True,
        "default_on"  : True,
        "color"       : "orange",
        "escalation"  : "ALERT",
    },
    4: {
        "name"        : "EMERGENCY",
        "label"       : "Emergency",
        "can_disable" : False,
        "can_enable"  : True,
        "default_on"  : True,
        "color"       : "red",
        "escalation"  : "EMERGENCY",
    },
}


def get_tier_for_event(event_type: str, threat_score: int = 0,
                       flags: list = None) -> int:
    """
    Determine notification tier for a detected event.

    Returns tier number 0-4.
    """
    flags = flags or []

    # TIER 4 — EMERGENCY 
    if any(f in flags for f in [
        "AIMING_AT_CAMERA", "BREAK_IN_ATTEMPT",
    ]):
        return 4
    if threat_score >= 75:
        return 4

    # TIER 3 — ALERT 
    if any("weapon:" in f for f in flags):
        return 3
    if any("face:known_dangerous" in f for f in flags):
        return 3
    if threat_score >= 55:
        return 3

    # TIER 2 — NOTIFY (mandatory) 
    if any(f in flags for f in [
        "face:masked", "near_door_zone", "known_danger_floor",
    ]):
        return 2
    if any("behavior:loitering" in f for f in flags):
        return 2
    if threat_score >= 30:
        return 2

    # TIER 1 — FRIENDLY 
    if event_type == "delivery":
        return 1
    if event_type == "trusted_face":
        return 1
    if event_type == "visitor":
        return 1

    # ── TIER 0 — SILENT ───────────────────────────────────────────────────────
    return 0


class NotificationEvent:
    """Represents a single notification event with its tier."""

    def __init__(self, tier: int, event_type: str, title: str,
                 body: str, threat_score: int = 0,
                 flags: list = None, clip_id: str = None):
        self.tier        = tier
        self.event_type  = event_type
        self.title       = title
        self.body        = body
        self.threat_score= threat_score
        self.flags       = flags or []
        self.clip_id     = clip_id
        self.timestamp   = datetime.now().isoformat()
        self.tier_info   = TIERS[tier]

    @property
    def escalation(self) -> str:
        return self.tier_info["escalation"]

    @property
    def color(self) -> str:
        return self.tier_info["color"]

    @property
    def tier_name(self) -> str:
        return self.tier_info["name"]

    @property
    def can_disable(self) -> bool:
        return self.tier_info["can_disable"]

    def to_dict(self) -> dict:
        return {
            "tier"       : self.tier,
            "tier_name"  : self.tier_name,
            "event_type" : self.event_type,
            "title"      : self.title,
            "body"       : self.body,
            "score"      : self.threat_score,
            "flags"      : self.flags,
            "clip_id"    : self.clip_id,
            "timestamp"  : self.timestamp,
            "color"      : self.color,
            "escalation" : self.escalation,
            "can_disable": self.can_disable,
        }

    def __repr__(self):
        return (f"NotificationEvent(tier={self.tier}, "
                f"type={self.event_type}, title={self.title})")


class NotificationManager:
    """
    Decides what notifications to send based on tier and user settings.
    """

    def __init__(self, settings: dict = None):
        # Default settings — which tiers are enabled
        self.settings = {
            "tier_0_enabled": False,   # user opt-in for silent motion
            "tier_1_enabled": True,    # friendly (delivery etc) — can toggle
            "tier_2_enabled": True,    # notify — cannot toggle off
            "tier_3_enabled": True,    # alert — cannot toggle off
            "tier_4_enabled": True,    # emergency — cannot toggle off
        }
        if settings:
            self.settings.update(settings)

        self._history = []   # in-memory notification history

    def should_notify(self, tier: int) -> bool:
        """Returns True if this tier should send a notification."""
        # Tiers 2-4 are always on
        if tier >= 2:
            return True
        # Tiers 0-1 depend on user settings
        return self.settings.get(f"tier_{tier}_enabled", False)

    def process(self, event: NotificationEvent) -> bool:
        """
        Process a notification event.
        Returns True if notification should be sent.
        """
        self._history.insert(0, event.to_dict())
        if len(self._history) > 500:
            self._history.pop()

        return self.should_notify(event.tier)

    def update_settings(self, new_settings: dict):
        """
        Update notification settings.
        Tiers 2-4 cannot be disabled — we enforce this here.
        """
        for key, value in new_settings.items():
            # Enforce mandatory tiers
            if key in ("tier_2_enabled", "tier_3_enabled", "tier_4_enabled"):
                self.settings[key] = True   # always on, ignore user input
            else:
                self.settings[key] = value

    def get_history(self, limit: int = 50) -> list:
        return self._history[:limit]

    def get_settings(self) -> dict:
        return {**self.settings}
