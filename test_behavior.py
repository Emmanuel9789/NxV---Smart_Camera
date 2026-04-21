import time
from tracking.tracker import PersonTracker
from ai.behavior import BehaviorAnalyzer

tracker  = PersonTracker()
analyzer = BehaviorAnalyzer()
t = time.time()

# Simulate a person loitering (barely moving for 20 seconds)
boxes = [(100, 100, 80, 160)]
for i in range(30):
    persons = tracker.update(boxes, timestamp=t + i * 0.7)
    # tiny wiggle so it's not perfectly static
    boxes = [(100 + (i % 3), 100, 80, 160)]

results = analyzer.analyze(persons)
for pid, result in results.items():
    print(result)
    print(f"  Risk : {result.risk_level}")
    print(f"  Flags: {result.flags}")
    print(f"  Score: {result.score}")

print("\nBehavior OK" if results else "PROBLEM - no results")
