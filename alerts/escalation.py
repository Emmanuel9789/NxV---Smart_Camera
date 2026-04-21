"""

Receives ThreatScore objects and decides:
  - Who to alert (owner / trusted contact / police prompt)
  - When to escalate (owner unreachable timeout)
  - When to trigger active deterrent

Escalation flow:
  NOTIFY    → SMS owner only
  ALERT     → SMS owner + start owner-unreachable timer
  EMERGENCY → SMS owner + SMS contact immediately + police prompt
            → if owner_away: skip owner timer, go straight to contact + police
"""

import time
from alerts.notifier  import Notifier
from alerts.deterrent import Deterrent


# ── Config ────────────────────────────────────────────────────────────────────
OWNER_RESPONSE_TIMEOUT = 120   # seconds before assuming owner unreachable
OWNER_NAME             = "Emmanuel"
HOME_ADDRESS           = "your home address"   # update this


class EscalationEngine:
    """
    Orchestrates the full NxV alert and escalation pipeline.

    Usage:
        engine = EscalationEngine()

        # Call this every time a new ThreatScore is produced
        engine.handle(threat_score)
    """

    def __init__(self):
        self.notifier   = Notifier()
        self.deterrent  = Deterrent()

        # Track escalation state
        self._active_alerts     = {}   # person_id → { score, time, escalation }
        self._owner_alerted_at  = {}   # person_id → timestamp owner was first alerted
        self._contact_notified  = set()  # person_ids already sent to contact

    def handle(self, threat_score, user_away: bool = False) -> dict:
        """
        Main entry point. Pass in a ThreatScore and user_away flag.
        Returns a dict describing what actions were taken.
        """
        pid        = threat_score.person_id
        score      = threat_score.final_score
        escalation = threat_score.escalation
        flags      = threat_score.all_flags

        actions = {
            "person_id"       : pid,
            "score"           : score,
            "escalation"      : escalation,
            "sms_owner"       : False,
            "sms_contact"     : False,
            "police_prompt"   : False,
            "deterrent"       : False,
            "social_search"   : threat_score.trigger_social_search,
        }

        if escalation == "NONE":
            return actions

        # ── NOTIFY — SMS owner only ───────────────────────────────────────────
        if escalation == "NOTIFY":
            sent = self.notifier.notify_owner(score, escalation, flags)
            actions["sms_owner"] = sent
            if sent:
                self._owner_alerted_at.setdefault(pid, time.time())

        # ── ALERT — SMS owner + deterrent + start unreachable timer ──────────
        elif escalation == "ALERT":
            sent = self.notifier.notify_owner(score, escalation, flags)
            actions["sms_owner"] = sent
            if sent:
                self._owner_alerted_at.setdefault(pid, time.time())

            # Trigger deterrent (siren/voice warning)
            self.deterrent.trigger(escalation)
            actions["deterrent"] = True

            # If user is away — don't wait, notify contact immediately
            if user_away and pid not in self._contact_notified:
                sent_c = self.notifier.notify_contact(
                    score, escalation, flags, owner_name=OWNER_NAME
                )
                actions["sms_contact"] = sent_c
                if sent_c:
                    self._contact_notified.add(pid)

            # If user is home — check if owner hasn't responded in time
            elif not user_away:
                self._check_owner_timeout(pid, score, escalation, flags, actions)

        # ── EMERGENCY — all channels immediately ──────────────────────────────
        elif escalation == "EMERGENCY":
            # Owner
            sent_o = self.notifier.notify_owner(score, escalation, flags)
            actions["sms_owner"] = sent_o

            # Contact — immediately, no waiting
            if pid not in self._contact_notified:
                sent_c = self.notifier.notify_contact(
                    score, escalation, flags, owner_name=OWNER_NAME
                )
                actions["sms_contact"] = sent_c
                if sent_c:
                    self._contact_notified.add(pid)

            # Police prompt
            sent_p = self.notifier.notify_police_prompt(
                score, flags, address=HOME_ADDRESS
            )
            actions["police_prompt"] = sent_p

            # Max deterrent
            self.deterrent.trigger(escalation)
            actions["deterrent"] = True

        self._log(actions)
        return actions

    def _check_owner_timeout(self, pid, score, escalation, flags, actions):
        """
        If the owner was alerted but hasn't acknowledged within
        OWNER_RESPONSE_TIMEOUT seconds, escalate to trusted contact.
        """
        alerted_at = self._owner_alerted_at.get(pid)
        if not alerted_at:
            return
        if time.time() - alerted_at >= OWNER_RESPONSE_TIMEOUT:
            if pid not in self._contact_notified:
                sent_c = self.notifier.notify_contact(
                    score, escalation, flags, owner_name=OWNER_NAME
                )
                actions["sms_contact"] = sent_c
                if sent_c:
                    self._contact_notified.add(pid)

    def acknowledge(self, person_id: int):
        """
        Call this when owner acknowledges an alert (e.g. taps 'I see it' in app).
        Resets the unreachable timer for that person.
        """
        self._owner_alerted_at.pop(person_id, None)
        self._contact_notified.discard(person_id)
        print(f"[NxV Escalation] Alert acknowledged for P{person_id}")

    def clear(self, person_id: int):
        """
        Call when a threat is resolved (person left the scene).
        """
        self._active_alerts.pop(person_id, None)
        self._owner_alerted_at.pop(person_id, None)
        self._contact_notified.discard(person_id)
        print(f"[NxV Escalation] Threat cleared for P{person_id}")

    def _log(self, actions: dict):
        pid  = actions["person_id"]
        esc  = actions["escalation"]
        sent = [k for k, v in actions.items()
                if v is True and k not in ("deterrent",)]
        print(f"[NxV Escalation] P{pid} | {esc} | actions: {sent or 'none'}")
