import time
from tracking.tracker import PersonTracker

tracker = PersonTracker()
t = time.time()

frame1 = [(50, 50, 80, 160), (300, 60, 80, 160)]
frame2 = [(55, 52, 80, 160), (305, 58, 80, 160)]
frame3 = [(60, 54, 80, 160), (310, 56, 80, 160)]

for i, boxes in enumerate([frame1, frame2, frame3]):
    persons = tracker.update(boxes, timestamp=t + i * 0.1)
    print(f"Frame {i+1}: {[str(p) for p in persons]}")

print(f"\nActive persons: {tracker.active_count()}")
print("Tracker OK" if tracker.active_count() == 2 else "PROBLEM - expected 2 persons")
