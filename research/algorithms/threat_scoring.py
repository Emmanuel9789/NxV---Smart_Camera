"""
DSA Honor Project — Interpretable Multi-Factor Threat Scoring Algorithm
research/algorithm/threat_scoring.py

Evolution from simple boolean scoring to weighted multi-factor system.
"""

import datetime

#VERSION original 

def threat_score_v1(weapon, loitering, aggression):
    score = 0
    if weapon:    score += 0.6
    if loitering: score += 0.3
    if aggression:score += 0.4
    return score


# VERSION DSA upgrade


WEIGHTS = {
    "behavior": 0.20,
    "violence": 0.25,
    "weapon"  : 0.30,
    "face"    : 0.15,
    "time"    : 0.10,
}

ESCALATION_MAP = {
    "EMERGENCY": 75,
    "ALERT"    : 55,
    "NOTIFY"   : 30,
    "NONE"     : 0,
}

ESCALATION_BUMP = {
    "NONE"     : "NOTIFY",
    "NOTIFY"   : "ALERT",
    "ALERT"    : "EMERGENCY",
    "EMERGENCY": "EMERGENCY",
}


def get_time_risk():
    """Time-of-day risk score — O(1)"""
    hour = datetime.datetime.now().hour
    if 0  <= hour < 5:  return (100, "late_night")
    if 22 <= hour < 24: return (75,  "night")
    if 19 <= hour < 22: return (40,  "evening")
    return (0, "daytime")


def threat_score_v2(person_id,
                    behavior_score  = 0,
                    behavior_flags  = None,
                    violence_score  = 0,
                    weapon_count    = 0,
                    weapon_types    = None,
                    face_score      = 0,
                    is_masked       = False,
                    is_known_danger = False,
                    is_aiming       = False,
                    near_door       = False,
                    dwell_time      = 0.0,
                    speed           = 0.0,
                    user_away       = False):
    """
    Multi-factor interpretable threat scoring.
    Returns dict with score, escalation, flags and breakdown.

    Time complexity:
      - Scoring      : O(k), k = number of signal types (constant)
      - Hard rules   : O(1) per rule
      - Motion filter: O(1)
      - Escalation   : O(1) hash map lookup
    """
    behavior_flags = behavior_flags or []
    weapon_types   = weapon_types   or []
    flags          = []

    # ── HARD RULE 1: Aiming → EMERGENCY immediately ───────────────
    if is_aiming:
        flags.append("AIMING_AT_CAMERA")
        return _result(person_id, 100, "EMERGENCY", flags,
                       {"rule": "hard_override_aiming"})

    # ── HARD RULE 2: Break-in at night → EMERGENCY ────────────────
    t_score, t_label = get_time_risk()
    if near_door and t_label in ("night","late_night") and dwell_time > 5:
        flags.append(f"BREAK_IN_ATTEMPT({t_label})")
        return _result(person_id, 100, "EMERGENCY", flags,
                       {"rule": "hard_override_breakin"})

    # ── MOTION FILTER: passing by → ignore ────────────────────────
    if dwell_time < 4.0 and speed > 80.0:
        flags.append("passing_motion_filtered")
        return _result(person_id,
                       min(10, behavior_score // 4),
                       "NONE", flags,
                       {"rule": "motion_filter"})

    # ── Compute sub-scores ────────────────────────────────────────
    weapon_score = min(100, weapon_count * 60)
    for wt in weapon_types:
        flags.append(f"weapon:{wt}")
    if t_label != "daytime":
        flags.append(f"time:{t_label}")
    if near_door:
        behavior_score = min(100, behavior_score + 20)
        flags.append("near_door")

    # ── WEIGHTED COMBINATION — core DSA algorithm ─────────────────
    # Score = Σ (signal_i × weight_i)
    # Time complexity: O(k) where k = number of signals
    final = min(100, int(round(
        behavior_score * WEIGHTS["behavior"] +
        violence_score * WEIGHTS["violence"] +
        weapon_score   * WEIGHTS["weapon"]   +
        face_score     * WEIGHTS["face"]     +
        t_score        * WEIGHTS["time"]
    )))

    # ── Soft floors ───────────────────────────────────────────────
    if weapon_count > 0    and final < 60: final = 60; flags.append("weapon_floor")
    if is_known_danger     and final < 55: final = 55; flags.append("known_danger_floor")
    if is_masked and t_label in ("night","late_night") and final < 40:
        final = 40; flags.append("masked_night_floor")

    if is_masked:       flags.append("face:masked")
    if is_known_danger: flags.append("face:known_dangerous")
    for f in behavior_flags:
        flags.append(f"behavior:{f}")

    # ── User away bonus ───────────────────────────────────────────
    if user_away:
        final = min(100, final + 15)
        if final < 25: final = 25

    final = min(100, final)

    # ── ESCALATION — O(1) hash map lookup ────────────────────────
    esc = ("EMERGENCY" if final >= 75 else
           "ALERT"     if final >= 55 else
           "NOTIFY"    if final >= 30 else "NONE")

    if user_away and esc != "NONE":
        esc = ESCALATION_BUMP[esc]
        flags.append(f"away_bump→{esc}")

    breakdown = {
        "behavior": behavior_score, "violence": violence_score,
        "weapon"  : weapon_score,   "face"    : face_score,
        "time"    : t_score,        "time_label": t_label,
    }

    return _result(person_id, final, esc, flags, breakdown)


def score_and_sort(persons_data: list) -> list:
    """
    Score multiple persons and sort by threat level.

    Time complexity: O(n log n)
      - Score each person: O(n × k) = O(n) since k is constant
      - Sort by score:     O(n log n) using Timsort
    """
    results = [threat_score_v2(**p) for p in persons_data]
    # SORTING — highest threat first
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _result(person_id, score, escalation, flags, breakdown):
    return {
        "person_id" : person_id,
        "score"     : score,
        "escalation": escalation,
        "flags"     : flags,
        "breakdown" : breakdown,
    }


# ── DEMONSTRATION ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("V1 — Simple boolean scoring:")
    print(f"  weapon+loitering = {threat_score_v1(True, True, False)}")
    print()

    print("V2 — Multi-factor weighted scoring:")
    persons = [
        dict(person_id=0, dwell_time=1.5,  speed=140, behavior_score=5),
        dict(person_id=1, dwell_time=25,   speed=8,   behavior_score=40,
             behavior_flags=["loitering"], near_door=True),
        dict(person_id=2, weapon_count=1,  weapon_types=["gun"],
             dwell_time=5, speed=20, behavior_score=20),
        dict(person_id=3, is_aiming=True,  dwell_time=3, speed=10),
    ]

    ranked = score_and_sort(persons)
    print(f"  {'ID':<5} {'Score':<8} {'Escalation':<12} Flags")
    print(f"  {'─'*50}")
    for r in ranked:
        print(f"  P{r['person_id']:<4} {r['score']:<8} "
              f"{r['escalation']:<12} {r['flags']}")