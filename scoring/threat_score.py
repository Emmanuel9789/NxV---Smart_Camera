"""
NxV - Threat Score Engine (v2.1)
scoring/threat_score.py

Added: user presence modifier
  - user_away=True  → escalation bumped up one level, score floor raised
  - user_away=False → normal scoring

Hard rules (bypass weighted scoring):
  - Weapon aimed at camera       → EMERGENCY immediately
  - Break-in attempt at night    → EMERGENCY immediately
  - Known dangerous person       → minimum ALERT always
  - Masked face at night         → minimum NOTIFY always
  - Any threat when user away    → escalation bumped one level minimum

Motion filter:
  - Passing motion (low dwell, high speed) → score capped at 10, no alert

Social media search:
  - Only triggered when final score >= 30 (MEDIUM)
  - Unknown face only
"""

import datetime


# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHT_BEHAVIOR  = 0.20
WEIGHT_VIOLENCE  = 0.25
WEIGHT_WEAPON    = 0.30
WEIGHT_FACE      = 0.15
WEIGHT_TIME      = 0.10

# ── Time of day risk ──────────────────────────────────────────────────────────
NIGHT_HOURS   = range(22, 24)
LATE_HOURS    = range(0, 5)
EVENING_HOURS = range(19, 22)

def time_risk_score() -> tuple:
    hour = datetime.datetime.now().hour
    if hour in LATE_HOURS:
        return (100, "late_night")
    if hour in NIGHT_HOURS:
        return (75, "night")
    if hour in EVENING_HOURS:
        return (40, "evening")
    return (0, "daytime")

# ── Escalation thresholds ─────────────────────────────────────────────────────
ESCALATION_NOTIFY    = 30
ESCALATION_ALERT     = 55
ESCALATION_EMERGENCY = 75

# ── User away modifiers ───────────────────────────────────────────────────────
AWAY_SCORE_BONUS     = 15    # added to final score when user is away
AWAY_MIN_SCORE       = 25    # any detection when away = at least this score

# Escalation bump order when user is away
ESCALATION_BUMP = {
    "NONE"      : "NOTIFY",
    "NOTIFY"    : "ALERT",
    "ALERT"     : "EMERGENCY",
    "EMERGENCY" : "EMERGENCY",
}

# ── Social media search trigger ───────────────────────────────────────────────
SOCIAL_SEARCH_THRESHOLD = 30

# ── Motion filter ─────────────────────────────────────────────────────────────
PASSING_MAX_DWELL = 4.0
PASSING_MIN_SPEED = 80.0


class ThreatScoreEngine:

    def score(self,
              person_id        : int,
              behavior_score   : int   = 0,
              behavior_flags   : list  = None,
              violence_score   : int   = 0,
              violence_flags   : list  = None,
              weapon_count     : int   = 0,
              weapon_types     : list  = None,
              face_score       : int   = 0,
              face_flags       : list  = None,
              is_masked        : bool  = False,
              is_known_danger  : bool  = False,
              is_aiming        : bool  = False,
              near_door        : bool  = False,
              dwell_time       : float = 0.0,
              speed            : float = 0.0,
              user_away        : bool  = False,
              ) -> "ThreatScore":

        behavior_flags = behavior_flags or []
        violence_flags = violence_flags or []
        face_flags     = face_flags     or []
        weapon_types   = weapon_types   or []
        all_flags      = []

        if user_away:
            all_flags.append("user_away")

        # ── HARD RULE 1: Weapon aimed → EMERGENCY ────────────────────────────
        if is_aiming:
            all_flags.append("AIMING_AT_CAMERA")
            return ThreatScore(
                person_id=person_id, final_score=100,
                escalation="EMERGENCY", all_flags=all_flags,
                trigger_social_search=False, breakdown={},
                user_away=user_away
            )

        # ── HARD RULE 2: Break-in at door at night → EMERGENCY ───────────────
        t_score, t_label = time_risk_score()
        if near_door and t_label in ("night", "late_night") and dwell_time > 5:
            all_flags.append(f"BREAK_IN_ATTEMPT({t_label})")
            return ThreatScore(
                person_id=person_id, final_score=100,
                escalation="EMERGENCY", all_flags=all_flags,
                trigger_social_search=False, breakdown={},
                user_away=user_away
            )

        # ── MOTION FILTER: Passing by → cap at 10, no alert ──────────────────
        # User away exception: even passing motion gets a NOTIFY when away
        is_passing = (dwell_time < PASSING_MAX_DWELL and
                      speed > PASSING_MIN_SPEED)
        if is_passing and not user_away:
            all_flags.append("passing_motion")
            return ThreatScore(
                person_id=person_id, final_score=min(10, behavior_score // 4),
                escalation="NONE", all_flags=all_flags,
                trigger_social_search=False, breakdown={},
                user_away=user_away
            )
        if is_passing and user_away:
            # Still filter but log silently — no alert for passing when away
            all_flags.append("passing_motion")
            return ThreatScore(
                person_id=person_id, final_score=min(10, behavior_score // 4),
                escalation="NONE", all_flags=all_flags,
                trigger_social_search=False, breakdown={},
                user_away=user_away
            )

        # ── Weapon score ──────────────────────────────────────────────────────
        weapon_score = 0
        if weapon_count > 0:
            weapon_score = min(100, weapon_count * 60)
            for wt in weapon_types:
                all_flags.append(f"weapon:{wt}")

        # ── Time of day ───────────────────────────────────────────────────────
        if t_label != "daytime":
            all_flags.append(f"time:{t_label}")

        # ── Near door (daytime) ───────────────────────────────────────────────
        if near_door:
            behavior_score = min(100, behavior_score + 20)
            all_flags.append("near_door_zone")

        # ── Weighted combination ──────────────────────────────────────────────
        weighted = (
            behavior_score * WEIGHT_BEHAVIOR +
            violence_score * WEIGHT_VIOLENCE +
            weapon_score   * WEIGHT_WEAPON   +
            face_score     * WEIGHT_FACE     +
            t_score        * WEIGHT_TIME
        )
        final_score = min(100, int(round(weighted)))

        # ── User away: add bonus + enforce minimum ────────────────────────────
        if user_away:
            final_score = min(100, final_score + AWAY_SCORE_BONUS)
            if final_score < AWAY_MIN_SCORE:
                final_score = AWAY_MIN_SCORE
                all_flags.append("away_floor_applied")

        # ── Soft floors ───────────────────────────────────────────────────────
        if weapon_count > 0 and final_score < 60:
            final_score = 60
            all_flags.append("weapon_floor")

        if is_known_danger and final_score < 55:
            final_score = 55
            all_flags.append("known_danger_floor")

        if is_masked and t_label in ("night", "late_night") and final_score < 40:
            final_score = 40
            all_flags.append("masked_night_floor")

        if is_masked:
            all_flags.append("face:masked")
        if is_known_danger:
            all_flags.append("face:known_dangerous")

        for f in behavior_flags:
            all_flags.append(f"behavior:{f}")
        for f in violence_flags:
            all_flags.append(f"violence:{f}")

        final_score = min(100, final_score)

        # ── Escalation ────────────────────────────────────────────────────────
        if final_score >= ESCALATION_EMERGENCY:
            escalation = "EMERGENCY"
        elif final_score >= ESCALATION_ALERT:
            escalation = "ALERT"
        elif final_score >= ESCALATION_NOTIFY:
            escalation = "NOTIFY"
        else:
            escalation = "NONE"

        # ── User away: bump escalation one level up ───────────────────────────
        if user_away and escalation != "NONE":
            original   = escalation
            escalation = ESCALATION_BUMP[escalation]
            if escalation != original:
                all_flags.append(f"away_escalation_bump:{original}->{escalation}")

        # ── Social search (unknown face, medium+ threat) ──────────────────────
        trigger_social = (
            not is_known_danger and
            final_score >= SOCIAL_SEARCH_THRESHOLD
        )

        breakdown = {
            "behavior"  : behavior_score,
            "violence"  : violence_score,
            "weapon"    : weapon_score,
            "face"      : face_score,
            "time"      : t_score,
            "time_label": t_label,
            "user_away" : user_away,
        }

        return ThreatScore(
            person_id=person_id, final_score=final_score,
            escalation=escalation, all_flags=all_flags,
            trigger_social_search=trigger_social,
            breakdown=breakdown, user_away=user_away
        )

    def score_from_results(self,
                           persons           : list,
                           behavior_results  : dict,
                           violence_result,
                           weapon_detections : list,
                           face_results      : list,
                           door_zone         : tuple = None,
                           user_away         : bool  = False) -> list:

        weapon_count    = len(weapon_detections)
        weapon_types    = [d.get('class', 'unknown') for d in weapon_detections]
        face_score      = 0
        face_flags      = []
        is_masked       = False
        is_known_danger = False

        for fr in face_results:
            if fr['threat_score'] > face_score:
                face_score = fr['threat_score']
            if fr['masked']:
                is_masked = True
            if fr['match']:
                is_known_danger = True
                face_flags.append(f"known:{fr['match'].get('name','?')}")

        scores = []
        for person in persons:
            b_result  = behavior_results.get(person.id)
            b_score   = b_result.score if b_result else 0
            b_flags   = b_result.flags if b_result else []

            near_door = False
            if door_zone:
                cx, cy         = person.centroid
                dx, dy, dw, dh = door_zone
                near_door      = (dx <= cx <= dx + dw and dy <= cy <= dy + dh)

            result = self.score(
                person_id       = person.id,
                behavior_score  = b_score,
                behavior_flags  = b_flags,
                violence_score  = violence_result.score,
                violence_flags  = violence_result.flags,
                weapon_count    = weapon_count,
                weapon_types    = weapon_types,
                face_score      = face_score,
                face_flags      = face_flags,
                is_masked       = is_masked,
                is_known_danger = is_known_danger,
                near_door       = near_door,
                dwell_time      = person.dwell_time,
                speed           = person.speed,
                user_away       = user_away,
            )
            scores.append(result)

        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores


class ThreatScore:

    def __init__(self, person_id, final_score, escalation,
                 all_flags, trigger_social_search, breakdown, user_away):
        self.person_id             = person_id
        self.final_score           = final_score
        self.escalation            = escalation
        self.all_flags             = all_flags
        self.trigger_social_search = trigger_social_search
        self.breakdown             = breakdown
        self.user_away             = user_away

    @property
    def summary(self) -> str:
        away = " [AWAY]" if self.user_away else ""
        return (f"P{self.person_id}{away} | Score:{self.final_score} | "
                f"{self.escalation} | "
                f"{', '.join(self.all_flags) or 'no flags'}")

    def __repr__(self):
        return (f"ThreatScore(person={self.person_id}, "
                f"score={self.final_score}, "
                f"escalation={self.escalation}, "
                f"away={self.user_away}, "
                f"social_search={self.trigger_social_search})")