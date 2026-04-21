from scoring.threat_score import ThreatScoreEngine
engine = ThreatScoreEngine()

# Same scenario, user home vs away
base = dict(person_id=0, behavior_score=35,
            behavior_flags=['loitering(20s)'], dwell_time=20.0, speed=5.0)

r_home = engine.score(**base, user_away=False)
r_away = engine.score(**base, user_away=True)

print("User HOME:")
print(f"  {r_home.summary}")
print("User AWAY:")
print(f"  {r_away.summary}")
print("\nThreat scoring OK")