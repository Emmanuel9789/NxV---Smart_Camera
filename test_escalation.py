from scoring.threat_score import ThreatScoreEngine
from alerts.escalation import EscalationEngine

scorer     = ThreatScoreEngine()
escalation = EscalationEngine()

# Test 1 — NOTIFY (loitering at night, user home)
r1 = scorer.score(person_id=0, behavior_score=35,
                  behavior_flags=['loitering(20s)'],
                  dwell_time=20.0, speed=5.0)
print("Test 1 - NOTIFY:")
a1 = escalation.handle(r1, user_away=False)
print(f"  Actions: {a1}\n")

# Test 2 — EMERGENCY (weapon, user away)
escalation.notifier.reset_cooldowns()
r2 = scorer.score(person_id=1, weapon_count=1,
                  weapon_types=['gun'], is_aiming=True,
                  dwell_time=5.0, speed=10.0)
print("Test 2 - EMERGENCY (weapon + aiming, user away):")
a2 = escalation.handle(r2, user_away=True)
print(f"  Actions: {a2}\n")

print("Escalation OK")
