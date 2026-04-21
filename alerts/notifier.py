"""

Sends SMS alerts to the owner and trusted contact via Twilio.
Handles:
  - Owner alert with threat summary
  - Trusted contact alert when owner unreachable
  - Police prompt message with pre-written evidence summary
  - Rate limiting so the same alert doesn't spam every frame

"""

import time
import os
from datetime import datetime


# ── Try importing Twilio ───────────────────────────────────────────────────────
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[NxV Notifier] Twilio not installed.")
    print("  Run: pip install twilio --break-system-packages")


# ── Config — fill these in or set as environment variables ────────────────────
# export NXV_TWILIO_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# export NXV_TWILIO_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# export NXV_TWILIO_FROM="+18446563069"
# export NXV_OWNER_PHONE="+14437641048"
# export NXV_CONTACT_PHONE="+16672006967"

TWILIO_SID      = os.environ.get("NXV_TWILIO_SID",    "")
TWILIO_TOKEN    = os.environ.get("NXV_TWILIO_TOKEN",   "")
TWILIO_FROM     = os.environ.get("NXV_TWILIO_FROM",    "")
OWNER_PHONE     = os.environ.get("NXV_OWNER_PHONE",    "")
CONTACT_PHONE   = os.environ.get("NXV_CONTACT_PHONE",  "")

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Minimum seconds between alerts of the same type to avoid spam
COOLDOWN = {
    "NOTIFY"    : 60,    # 1 min between NOTIFY SMS
    "ALERT"     : 30,    # 30s between ALERT SMS
    "EMERGENCY" : 10,    # 10s between EMERGENCY SMS
}


class Notifier:
    """
    Sends SMS alerts via Twilio.

    Usage:
        notifier = Notifier()
        notifier.notify_owner(threat_score, flags, clip_path)
        notifier.notify_contact(threat_score, flags, clip_path)
        notifier.notify_police_prompt(threat_score, flags, clip_path)
    """

    def __init__(self):
        self._last_sent = {}   # escalation_type → timestamp
        self._client    = None

        if TWILIO_AVAILABLE and TWILIO_SID and TWILIO_TOKEN:
            try:
                self._client = Client(TWILIO_SID, TWILIO_TOKEN)
                print("[NxV Notifier] Twilio client ready.")
            except Exception as e:
                print(f"[NxV Notifier] Twilio init failed: {e}")
        else:
            print("[NxV Notifier] Running in DRY RUN mode — SMS will be printed, not sent.")

    # ── Public alert methods ───────────────────────────────────────────────────

    def notify_owner(self,
                     threat_score : int,
                     escalation   : str,
                     flags        : list,
                     clip_path    : str = None) -> bool:
        """
        Send SMS to owner with threat summary.
        Returns True if sent, False if rate-limited or failed.
        """
        if not self._can_send("owner", escalation):
            return False

        time_str = datetime.now().strftime("%I:%M %p")
        flag_str = ", ".join(flags[:4]) or "no specific flags"

        msg = (
            f"NxV ALERT [{escalation}] {time_str}\n"
            f"Threat score: {threat_score}/100\n"
            f"Signals: {flag_str}\n"
            f"Check your camera feed immediately."
        )

        if clip_path:
            msg += f"\nClip saved: {clip_path}"

        return self._send(OWNER_PHONE, msg, tag="owner")

    def notify_contact(self,
                       threat_score : int,
                       escalation   : str,
                       flags        : list,
                       owner_name   : str = "The homeowner",
                       clip_path    : str = None) -> bool:
        """
        Send SMS to trusted contact when owner is unreachable.
        """
        if not self._can_send("contact", escalation):
            return False

        time_str = datetime.now().strftime("%I:%M %p")
        flag_str = ", ".join(flags[:4]) or "suspicious activity"

        msg = (
            f"NxV URGENT [{escalation}] {time_str}\n"
            f"{owner_name} is away and their security camera detected a threat.\n"
            f"Threat score: {threat_score}/100\n"
            f"Signals: {flag_str}\n"
            f"Please check on them or call 911 if needed."
        )

        return self._send(CONTACT_PHONE, msg, tag="contact")

    def notify_police_prompt(self,
                              threat_score : int,
                              flags        : list,
                              address      : str = "your registered address",
                              clip_path    : str = None) -> bool:
        """
        Send a pre-written police report prompt to owner.
        Owner just needs to forward this or read it when calling 911.
        """
        if not self._can_send("police", "EMERGENCY"):
            return False

        time_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        flag_str = ", ".join(flags[:6]) or "threat detected"

        msg = (
            f"NxV EMERGENCY — CALL 911 NOW\n"
            f"---\n"
            f"Say this: 'I want to report a security threat at {address}. "
            f"My security camera detected: {flag_str} "
            f"at {time_str}. Threat score: {threat_score}/100. "
            f"I have video evidence.'\n"
            f"---\n"
            f"Evidence clip: {clip_path or 'check NxV app'}"
        )

        return self._send(OWNER_PHONE, msg, tag="police")

    # ── Internals ──────────────────────────────────────────────────────────────

    def _can_send(self, tag: str, escalation: str) -> bool:
        """Check rate limit for this alert type."""
        cooldown = COOLDOWN.get(escalation, 60)
        last     = self._last_sent.get(tag, 0)
        return (time.time() - last) >= cooldown

    def _send(self, to: str, body: str, tag: str) -> bool:
        """Send SMS or print in dry-run mode."""
        self._last_sent[tag] = time.time()

        if self._client and to:
            try:
                msg = self._client.messages.create(
                    body = body,
                    from_= TWILIO_FROM,
                    to   = to
                )
                print(f"[NxV Notifier] SMS sent to {to} — SID: {msg.sid}")
                return True
            except Exception as e:
                print(f"[NxV Notifier] SMS failed: {e}")
                return False
        else:
            # Dry run — print what would be sent
            print(f"\n[NxV Notifier DRY RUN → {to or 'NO_NUMBER_SET'}]")
            print("─" * 50)
            print(body)
            print("─" * 50)
            return True   # count as sent in dry run

    def reset_cooldowns(self):
        """Reset all rate limits — useful for testing."""
        self._last_sent.clear()
