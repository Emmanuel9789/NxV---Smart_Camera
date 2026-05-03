"""
NxV - Call Escalation Chain
alerts/call_chain.py

Calls owner first, then trusted contacts in sequence.
Final fallback sends 911 prompt to all if nobody answers.
"""

import os
import time
import threading
from datetime import datetime

# ── Contact list ──────────────────────────────────────────────────────────────
CONTACTS = [
    {"name": "Owner",             "phone": os.environ.get("NXV_OWNER_PHONE",    ""), "role": "owner"},
    {"name": "Trusted Contact 1", "phone": os.environ.get("NXV_CONTACT_1_PHONE",""), "role": "contact"},
    {"name": "Trusted Contact 2", "phone": os.environ.get("NXV_CONTACT_2_PHONE",""), "role": "contact"},
    {"name": "Trusted Contact 3", "phone": os.environ.get("NXV_CONTACT_3_PHONE",""), "role": "contact"},
]

# ── Config ────────────────────────────────────────────────────────────────────
ANSWER_WAIT_SECS = 30
CALL_TIMEOUT     = 25
TEST_MODE        = os.environ.get("NXV_TEST_MODE", "true").lower() == "true"

# ── Twilio ────────────────────────────────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_SID   = os.environ.get("NXV_TWILIO_SID",   "")
    TWILIO_TOKEN = os.environ.get("NXV_TWILIO_TOKEN",  "")
    TWILIO_FROM  = os.environ.get("NXV_TWILIO_FROM",   "")
    _twilio      = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID else None
    TWILIO_OK    = bool(_twilio)
except ImportError:
    _twilio   = None
    TWILIO_OK = False


class CallChain:
    """Full call escalation chain for NxV EMERGENCY alerts."""

    def __init__(self):
        self._active      = False
        self._thread      = None
        self._answered_by = None

    def start(self, threat_score: int, flags: list,
              clip_path: str = None, address: str = "your home"):
        if self._active:
            print("[NxV CallChain] Already running — skipping")
            return
        self._active      = True
        self._answered_by = None
        self._thread      = threading.Thread(
            target = self._run_chain,
            args   = (threat_score, flags, clip_path, address),
            daemon = True
        )
        self._thread.start()
        print("[NxV CallChain] Escalation chain started in background")

    def stop(self):
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def answered_by(self) -> str | None:
        return self._answered_by

    def _run_chain(self, threat_score, flags, clip_path, address):
        time_str = datetime.now().strftime("%I:%M %p")
        date_str = datetime.now().strftime("%B %d")
        answered = False

        print(f"\n[NxV CallChain] ═══ EMERGENCY ESCALATION STARTED ═══")
        print(f"[NxV CallChain] Score : {threat_score}")
        print(f"[NxV CallChain] Flags : {', '.join(flags[:4])}")
        print(f"[NxV CallChain] Time  : {time_str}")
        print(f"[NxV CallChain] Chain : {len(CONTACTS)} contacts\n")

        for i, contact in enumerate(CONTACTS):
            if not self._active:
                return

            if not contact["phone"]:
                print(f"[NxV CallChain] Step {i+1}: {contact['name']} — no number set, skipping")
                continue

            print(f"[NxV CallChain] Step {i+1}/{len(CONTACTS)}: Calling {contact['name']} ({contact['phone']})")

            message       = self._build_message(contact, threat_score, flags, time_str, date_str)
            call_answered = self._make_call(contact["phone"], message, contact["name"])

            if call_answered:
                answered          = True
                self._answered_by = contact["name"]
                print(f"[NxV CallChain] ✓ Answered by {contact['name']}")
                self._send_sms(
                    contact["phone"],
                    f"NxV EMERGENCY [{time_str}]\n"
                    f"Score: {threat_score}/100\n"
                    f"Detected: {', '.join(flags[:4])}\n"
                    f"Evidence: {clip_path or 'check NxV app'}"
                )
                break

            else:
                print(f"[NxV CallChain] ✗ No answer from {contact['name']}")
                self._send_sms(
                    contact["phone"],
                    f"NxV MISSED CALL [{time_str}]\n"
                    f"You missed an emergency alert.\n"
                    f"Score: {threat_score}/100\n"
                    f"Detected: {', '.join(flags[:4])}\n"
                    f"Evidence: {clip_path or 'check NxV app'}"
                )
                if i < len(CONTACTS) - 1:
                    next_c = next((c for c in CONTACTS[i+1:] if c["phone"]), None)
                    if next_c:
                        print(f"[NxV CallChain] Waiting {ANSWER_WAIT_SECS}s then trying {next_c['name']}...")
                        time.sleep(ANSWER_WAIT_SECS)

        if not answered:
            self._nobody_answered(threat_score, flags, clip_path, address, time_str)

        self._active = False
        print(f"[NxV CallChain] ═══ ESCALATION COMPLETE ═══\n")

    def _build_message(self, contact, threat_score, flags, time_str, date_str) -> str:
        detections   = []
        time_context = ""

        for flag in flags:
            if "AIMING_AT_CAMERA"   in flag: detections.append("a weapon is being aimed directly at the camera")
            elif "BREAK_IN_ATTEMPT" in flag: detections.append("a break-in attempt at the door")
            elif "weapon:gun"       in flag: detections.append("a gun was detected")
            elif "weapon:knife"     in flag: detections.append("a knife was detected")
            elif "weapon:"          in flag: detections.append("a weapon was detected")
            elif "behavior:loitering" in flag: detections.append("a person loitering suspiciously")
            elif "behavior:pacing"  in flag: detections.append("suspicious pacing behavior")
            elif "face:known_dangerous" in flag: detections.append("a known dangerous individual was recognized")
            elif "face:masked"      in flag: detections.append("a masked individual was detected")
            elif "time:late_night"  in flag: time_context = "in the middle of the night"
            elif "time:night"       in flag: time_context = "at night"
            elif "time:evening"     in flag: time_context = "in the evening"

        if not detections:
            detections = ["suspicious activity"]

        detection_str = " and ".join(detections)
        time_say      = f"{time_context}, " if time_context else ""

        if contact["role"] == "owner":
            return (
                f"NxV Security Emergency. {time_say}"
                f"At {time_str} on {date_str}, "
                f"your security camera detected {detection_str} at your home. "
                f"Threat score is {threat_score} out of 100. "
                f"Please check your camera feed immediately or call 911."
            )
        else:
            return (
                f"NxV Security Alert. This is an automated emergency call. "
                f"The homeowner is not responding. "
                f"{time_say}At {time_str} on {date_str}, "
                f"a security camera detected {detection_str}. "
                f"Threat score is {threat_score} out of 100. "
                f"Please check on them or call 911 immediately."
            )

    def _nobody_answered(self, threat_score, flags, clip_path, address, time_str):
        print(f"[NxV CallChain] ⚠ NOBODY ANSWERED — sending 911 prompt to all")
        flag_str   = ", ".join(flags[:4]) or "threat detected"
        police_msg = (
            f"NxV EMERGENCY — NOBODY ANSWERED\n"
            f"CALL 911 NOW for {address}\n"
            f"Say: 'Security camera detected {flag_str} "
            f"at {address} at {time_str}. "
            f"Threat score {threat_score}/100. I have video evidence.'\n"
            f"Evidence: {clip_path or 'check NxV app'}"
        )
        for contact in CONTACTS:
            if contact["phone"]:
                self._send_sms(contact["phone"], police_msg)

        if TEST_MODE:
            print("[NxV CallChain] TEST MODE — skipping real 911 call")
            print("[NxV CallChain] In production, 911 would be called here")
        else:
            print("[NxV CallChain] PRODUCTION MODE — 911 would be called here")

    def _make_call(self, phone: str, message: str, name: str) -> bool:
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">{message}</Say>
    <Pause length="2"/>
    <Say voice="alice">Repeating. {message}</Say>
</Response>"""

        if not TWILIO_OK or not TWILIO_FROM:
            print(f"  [DRY RUN] Would call {name} at {phone}")
            time.sleep(2)
            return False

        try:
            call = _twilio.calls.create(
                twiml   = twiml,
                to      = phone,
                from_   = TWILIO_FROM,
                timeout = CALL_TIMEOUT,
            )
            print(f"  Call SID: {call.sid} — waiting...")
            time.sleep(6)
            
            in_progress_count = 0
            start = time.time()
            while time.time() - start < ANSWER_WAIT_SECS:
                time.sleep(3)
                updated = _twilio.calls(call.sid).fetch()
                status  = updated.status
                print(f"  Status: {status}")

                if status == "in-progress":
                    in_progress_count += 1
                    # If in-progress for 3+ polls (9+ seconds) = human answered
                    if in_progress_count >= 3:
                        print(f"  Call connected — human answered")
                        time.sleep(20)  # let message play fully
                        return True
                    continue
                    
                if status == "busy":
                    print(f"  {name} is busy or rejected")
                    return False
                if status in ("completed", "no-answer", "failed", "canceled"):
                    return False

            try:
                _twilio.calls(call.sid).update(status="completed")
            except Exception:
                pass
            return False

        except Exception as e:
            print(f"  [NxV CallChain] Call error: {e}")
            return False

    def _send_sms(self, phone: str, body: str):
        if not TWILIO_OK or not TWILIO_FROM:
            print(f"  [DRY RUN SMS → {phone}]: {body[:80]}...")
            return
        try:
            _twilio.messages.create(body=body, from_=TWILIO_FROM, to=phone)
            print(f"  SMS sent → {phone}")
        except Exception as e:
            print(f"  SMS failed → {phone}: {e}")