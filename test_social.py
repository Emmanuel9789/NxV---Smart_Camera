import numpy as np
from ai.social_search import SocialSearchEngine

engine = SocialSearchEngine()

# Fake face crop (blank image — just testing the flow)
fake_face = np.zeros((100, 80, 3), dtype=np.uint8)

# Should trigger — unknown face, score 45
if engine.should_search(person_id=0, threat_score=45, is_known=False):
    result = engine.search(fake_face, person_id=0)
    print(result)

# Should NOT trigger — known person
if not engine.should_search(person_id=1, threat_score=80, is_known=True):
    print("Correctly skipped known person")

print("Social search OK")
