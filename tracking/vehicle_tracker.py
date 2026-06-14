"""
Tracks vehicles across frames using IoU bipartite matching —
the same algorithm as the person tracker

Key difference from person tracker:
    - Vehicles move slowly → IoU threshold is higher (0.4 vs 0.3)
    - Vehicles can disappear behind trees then reappear
      so we keep a lost vehicle in memory for 3 seconds
      before deleting it (person tracker uses 1 second)
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrackedVehicle:
    
    """
    One tracked vehicle holds everything we know about it.

    vehicle_id  : unique int assigned at first detection
    vehicle_type: 'car', 'truck', 'motorcycle', etc.
    bbox        : (x1, y1, x2, y2) current bounding box
    conf        : YOLO confidence score
    center      : (cx, cy) current center point
    area        : pixel area — proxy for distance from camera
    first_seen  : timestamp when first detected
    last_seen   : timestamp of most recent detection
    dwell_time  : seconds vehicle has been visible
    path_history: list of (cx, cy) positions — sliding window
    pass_count  : how many times this vehicle has passed
    lost        : True if not seen in last frame (might reappear)
    """
    vehicle_id   : int
    vehicle_type : str
    bbox         : tuple
    conf         : float
    center       : tuple
    area         : int
    first_seen   : float = field(default_factory=time.time)
    last_seen    : float = field(default_factory=time.time)
    dwell_time   : float = 0.0
    path_history : list  = field(default_factory=list)
    pass_count   : int   = 0
    lost         : bool  = False

    def update(self, detection: dict):
        #Update vehicle with new detection data
        self.bbox       = detection['bbox']
        self.conf       = detection['conf']
        self.center     = detection['center']
        self.area       = detection['area']
        self.last_seen  = time.time()
        self.dwell_time = self.last_seen - self.first_seen
        self.lost       = False

 
        self.path_history.append(self.center)
        if len(self.path_history) > 90:
            self.path_history.pop(0)


def _compute_iou(box1: tuple, box2: tuple) -> float:

    #Intersection over Union between two bounding boxes.
    #box format: (x1, y1, x2, y2)

    #Returns float 0.0 to 1.0.
    #1.0 = perfect overlap (same vehicle, didn't move)
    #0.0 = no overlap (completely different locations)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1        = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2        = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union        = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class VehicleTracker:
    # How long to keep a lost vehicle before deleting it
    # 3 seconds — vehicles can disappear behind trees briefly
    
    LOST_TIMEOUT = 3.0

    # Minimum IoU to consider a match valid
    # Higher than person tracker (0.4 vs 0.3) because
    # vehicles move slowly between frames
    
    IOU_THRESHOLD = 0.4

    def __init__(self):
        # Hash map: vehicle_id → TrackedVehicle   O(1) lookup
        self._vehicles : dict[int, TrackedVehicle] = {}
        self._next_id  : int = 0

    def update(self, detections: list[dict]) -> list[TrackedVehicle]:
        """
        Match new detections to existing tracks.
        Create new tracks for unmatched detections.
        Mark tracks as lost if not detected this frame.

        Returns list of all currently active vehicles.
        """
        now = time.time()

        # Step 1: Remove vehicles lost for too long 
        lost_ids = [
            vid for vid, v in self._vehicles.items()
            if v.lost and (now - v.last_seen) > self.LOST_TIMEOUT
        ]
        for vid in lost_ids:
            del self._vehicles[vid] 

        if not detections:
            # No detections, mark all vehicles as lost
            for v in self._vehicles.values():
                v.lost = True
            return list(self._vehicles.values())

        if not self._vehicles:
            # No existing tracks, create new track for every detection
            for det in detections:
                self._create_track(det)
            return list(self._vehicles.values())

        # Step 2: Build IoU matrix 
        # Rows = new detections, Cols = existing tracks
        track_ids  = list(self._vehicles.keys())
        tracks     = [self._vehicles[tid] for tid in track_ids]

        n = len(detections)
        m = len(tracks)
        iou_matrix = np.zeros((n, m))

        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou_matrix[i][j] = _compute_iou(det['bbox'], track.bbox)

        # Step 3: Greedy matching 
        # Sort all edges by weight descending, take best non-conflicting
        matched_dets   = set()
        matched_tracks = set()

        edges = []
        for i in range(n):
            for j in range(m):
                edges.append((iou_matrix[i][j], i, j))
        edges.sort(reverse=True)  

        for iou, det_idx, track_idx in edges:
            if iou < self.IOU_THRESHOLD:
                break   # remaining edges too weak
            if det_idx in matched_dets or track_idx in matched_tracks:
                continue
            # Match found — update existing track
            self._vehicles[track_ids[track_idx]].update(detections[det_idx])
            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)

        # Step 4: Unmatched detections → new tracks 
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det)

        # Step 5: Unmatched tracks → mark as lost
        for j, tid in enumerate(track_ids):
            if j not in matched_tracks:
                self._vehicles[tid].lost = True

        return list(self._vehicles.values())

    def _create_track(self, detection: dict):
        """Create a new TrackedVehicle from a detection."""
        vid = self._next_id
        self._next_id += 1
        self._vehicles[vid] = TrackedVehicle(
            vehicle_id   = vid,
            vehicle_type = detection['class'],
            bbox         = detection['bbox'],
            conf         = detection['conf'],
            center       = detection['center'],
            area         = detection['area'],
        )

    def active_count(self) -> int:
        """How many vehicles are currently being tracked."""
        return sum(1 for v in self._vehicles.values() if not v.lost)

    def get_vehicle(self, vehicle_id: int):
        """O(1) lookup by ID."""
        return self._vehicles.get(vehicle_id)

    def draw(self, frame, vehicles: list[TrackedVehicle]):
        """Draw vehicle bounding boxes and labels on the frame."""
        import cv2
        COLOR = {
            'car'       : (255, 200, 0),    # blue-ish
            'truck'     : (255, 100, 0),    # darker blue
            'motorcycle': (200, 255, 0),    # cyan
            'bus'       : (255, 0, 200),    # purple
            'bicycle'   : (0, 255, 200),    # green-cyan
        }
        for v in vehicles:
            if v.lost:
                continue
            x1, y1, x2, y2 = v.bbox
            color = COLOR.get(v.vehicle_type, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = (f"{v.vehicle_type} #{v.vehicle_id} "
                     f"{v.dwell_time:.0f}s")
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 1, cv2.LINE_AA)
        return frame
