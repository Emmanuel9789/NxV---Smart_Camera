import time
from tracking.tracker import PersonTracker
from ai.violence import ViolenceDetector

tracker  = PersonTracker()
detector = ViolenceDetector()
t = time.time()

# Simulate 2 people moving fast toward each other
boxes_start = [(50, 200, 70, 150), (400, 200, 70, 150)]
boxes_close = [(200, 200, 70, 150), (260, 200, 70, 150)]

persons = tracker.update(boxes_start, timestamp=t)
persons = tracker.update(boxes_close, timestamp=t + 0.1)

result = detector.analyze(persons)
print(result)
print(f"  Risk : {result.risk_level}")
print(f"  Flags: {result.flags}")
print(f"  Score: {result.score}")

print("\nViolence OK" if result.score > 0 else "PROBLEM - expected a score above 0")
