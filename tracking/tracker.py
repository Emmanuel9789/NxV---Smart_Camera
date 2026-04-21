"""
Assigns a persistent ID to each detected person across frames.
Stores position history, computes velocity and dwell time.
Everything downstream (behavior, violence scoring) depends on this.

How it works:
  1. Each frame, we get bounding boxes from YOLO / motion detector
  2. We match new boxes to existing tracked persons using IoU overlap
  3. If a box matches an existing person → update their record
  4. If no match → register as a new person with a new ID
  5. If a tracked person disappears for too long → remove them
"""

import cv2
import numpy as np
from collections import OrderedDict
from datetime import datetime


class Person:
    """
    Represents a single tracked individual across frames.
    """

    def __init__(self, person_id: int, bbox: tuple, timestamp: float):
        """
        person_id  — unique integer ID for this person
        bbox       — (x, y, w, h) bounding box
        timestamp  — time.time() when first seen
        """
        self.id             = person_id
        self.bbox           = bbox          # current bounding box (x,y,w,h)
        self.centroid       = self._centroid(bbox)
        self.history        = [self.centroid]   # list of (cx, cy) positions
        self.timestamps     = [timestamp]        # time at each history point
        self.first_seen     = timestamp
        self.last_seen      = timestamp
        self.frames_missing = 0             # consecutive frames without detection
        self.dwell_time     = 0.0           # seconds in scene
        self.velocity       = (0.0, 0.0)   # (vx, vy) pixels per second
        self.disappeared    = False

    def update(self, bbox: tuple, timestamp: float):
        """Called each frame this person is matched to a detection."""
        prev_centroid       = self.centroid
        self.bbox           = bbox
        self.centroid       = self._centroid(bbox)
        self.last_seen      = timestamp
        self.frames_missing = 0
        self.dwell_time     = timestamp - self.first_seen
        self.disappeared    = False

        # Keep history capped at 60 points (~2 seconds at 30fps)
        self.history.append(self.centroid)
        self.timestamps.append(timestamp)
        if len(self.history) > 60:
            self.history.pop(0)
            self.timestamps.pop(0)

        # Velocity: pixels per second between last two positions
        dt = timestamp - self.timestamps[-2] if len(self.timestamps) >= 2 else 0.001
        if dt > 0:
            dx = self.centroid[0] - prev_centroid[0]
            dy = self.centroid[1] - prev_centroid[1]
            self.velocity = (dx / dt, dy / dt)

    @staticmethod
    def _centroid(bbox: tuple) -> tuple:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    @property
    def speed(self) -> float:
        """Scalar speed in pixels per second."""
        vx, vy = self.velocity
        return float(np.sqrt(vx**2 + vy**2))

    @property
    def displacement(self) -> float:
        """
        Total displacement from first seen position to now.
        Small displacement over long dwell = loitering candidate.
        """
        if len(self.history) < 2:
            return 0.0
        fx, fy = self.history[0]
        cx, cy = self.centroid
        return float(np.sqrt((cx - fx)**2 + (cy - fy)**2))

    @property
    def path_length(self) -> float:
        """Total distance travelled along the recorded path."""
        total = 0.0
        for i in range(1, len(self.history)):
            ax, ay = self.history[i - 1]
            bx, by = self.history[i]
            total += np.sqrt((bx - ax)**2 + (by - ay)**2)
        return total

    def is_pacing(self, min_reversals: int = 3, min_path: float = 80.0) -> bool:
        """
        Detect back-and-forth motion (pacing).
        Counts how many times the X direction reverses.
        """
        if len(self.history) < 10:
            return False
        if self.path_length < min_path:
            return False

        directions = []
        for i in range(1, len(self.history)):
            dx = self.history[i][0] - self.history[i - 1][0]
            if abs(dx) > 3:   # ignore tiny jitter
                directions.append(1 if dx > 0 else -1)

        reversals = sum(
            1 for i in range(1, len(directions))
            if directions[i] != directions[i - 1]
        )
        return reversals >= min_reversals

    def __repr__(self):
        return (f"Person(id={self.id}, dwell={self.dwell_time:.1f}s, "
                f"speed={self.speed:.1f}px/s, pacing={self.is_pacing()})")


class PersonTracker:
    """
    Tracks multiple people across frames using IoU-based matching.

    Usage:
        tracker = PersonTracker()

        # Each frame, pass in list of (x, y, w, h) bounding boxes:
        persons = tracker.update(boxes, timestamp=time.time())

        for person in persons:
            print(person.id, person.dwell_time, person.speed)
    """

    def __init__(self,
                 iou_threshold: float = 0.3,
                 max_missing_frames: int = 20):
        """
        iou_threshold      — minimum IoU to match a detection to a tracked person
        max_missing_frames — remove a person after this many frames without detection
        """
        self.iou_threshold      = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.persons            = OrderedDict()   # id → Person
        self._next_id           = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, boxes: list, timestamp: float = None) -> list:
        """
        Main update call. Pass in detected bounding boxes this frame.

        boxes     — list of (x, y, w, h) tuples from YOLO / motion detector
        timestamp — current time as float (defaults to now)

        Returns list of active Person objects.
        """
        import time
        if timestamp is None:
            timestamp = time.time()

        if len(boxes) == 0:
            # No detections — increment missing counter for everyone
            for person in self.persons.values():
                person.frames_missing += 1
            self._remove_lost()
            return list(self.persons.values())

        if len(self.persons) == 0:
            # No existing tracks — register all detections as new
            for box in boxes:
                self._register(box, timestamp)
            return list(self.persons.values())

        # ── Match detections to existing persons via IoU ───────────────────
        person_ids   = list(self.persons.keys())
        person_boxes = [self.persons[pid].bbox for pid in person_ids]

        iou_matrix = self._iou_matrix(person_boxes, boxes)

        matched_persons = set()
        matched_boxes   = set()

        # Greedy match: highest IoU pair first
        while True:
            if iou_matrix.size == 0:
                break
            idx = np.argmax(iou_matrix)
            r, c = np.unravel_index(idx, iou_matrix.shape)

            if iou_matrix[r, c] < self.iou_threshold:
                break   # remaining pairs are below threshold

            pid = person_ids[r]
            self.persons[pid].update(boxes[c], timestamp)
            matched_persons.add(r)
            matched_boxes.add(c)

            # Zero out this row and column so we don't re-match
            iou_matrix[r, :] = 0
            iou_matrix[:, c] = 0

        # Unmatched persons — increment missing counter
        for r, pid in enumerate(person_ids):
            if r not in matched_persons:
                self.persons[pid].frames_missing += 1

        # Unmatched detections — register as new persons
        for c, box in enumerate(boxes):
            if c not in matched_boxes:
                self._register(box, timestamp)

        self._remove_lost()
        return list(self.persons.values())

    def get_person(self, person_id: int):
        """Return a specific Person by ID, or None."""
        return self.persons.get(person_id)

    def active_count(self) -> int:
        return len(self.persons)

    def reset(self):
        """Clear all tracked persons (e.g. on camera restart)."""
        self.persons.clear()
        self._next_id = 0

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, persons: list = None) -> np.ndarray:
        """
        Draw tracking boxes, IDs, and motion trails on the frame.
        """
        if persons is None:
            persons = list(self.persons.values())

        for person in persons:
            x, y, w, h = person.bbox
            pid        = person.id

            # Box color: green normally, orange if pacing, red if long dwell
            if person.dwell_time > 30:
                color = (0, 0, 255)       # red — long dwell
            elif person.is_pacing():
                color = (0, 165, 255)     # orange — pacing
            else:
                color = (0, 200, 100)     # green — normal

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label: ID + dwell time
            label = f"P{pid} | {person.dwell_time:.0f}s"
            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Motion trail — last 15 positions
            trail = person.history[-15:]
            for i in range(1, len(trail)):
                alpha = int(80 + (i / len(trail)) * 175)
                pt1   = trail[i - 1]
                pt2   = trail[i]
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

        return frame

    # ── Internals ──────────────────────────────────────────────────────────────

    def _register(self, bbox: tuple, timestamp: float):
        person = Person(self._next_id, bbox, timestamp)
        self.persons[self._next_id] = person
        self._next_id += 1

    def _remove_lost(self):
        lost = [pid for pid, p in self.persons.items()
                if p.frames_missing > self.max_missing_frames]
        for pid in lost:
            del self.persons[pid]

    @staticmethod
    def _iou(boxA: tuple, boxB: tuple) -> float:
        """
        Compute Intersection over Union between two (x, y, w, h) boxes.
        """
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB

        # Convert to (x1, y1, x2, y2)
        ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
        bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def _iou_matrix(self, person_boxes: list, det_boxes: list) -> np.ndarray:
        """Build full IoU matrix: rows = existing persons, cols = new detections."""
        matrix = np.zeros((len(person_boxes), len(det_boxes)), dtype=float)
        for r, pb in enumerate(person_boxes):
            for c, db in enumerate(det_boxes):
                matrix[r, c] = self._iou(pb, db)
        return matrix