"""
Handles:
  - Face detection in every frame
  - Masked / obscured face flagging
  - Matching detected faces against the flagged persons database
  - Returning enriched face records with threat attribution
"""

import cv2
import numpy as np
import os
import pickle
import json
from datetime import datetime


try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[NxV] face_recognition not installed — embedding match disabled. "
          "Run: pip install face_recognition --break-system-packages")


# Paths 
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH         = os.path.join(BASE_DIR, "datasets", "flagged_persons.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "datasets", "face_embeddings.pkl")

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    """
    Detects faces in a frame and optionally matches them against
    a local database of flagged individuals.

    Usage:
        detector = FaceDetector()
        results  = detector.detect(frame)

    Each result dict:
        {
          'bbox'          : (x, y, w, h),
          'confidence'    : float,          # Haar score (higher = more confident)
          'masked'        : bool,           # True if face appears covered
          'match'         : dict | None,    # Matched person record or None
          'threat_score'  : int,            # 0-100 inherited from DB, 0 if unknown
          'label'         : str,            # Display label for overlay
        }
    """

    def __init__(self,
                 scale_factor: float = 1.1,
                 min_neighbors: int  = 5,
                 min_face_size: tuple = (40, 40),
                 mask_threshold: float = 0.45):
        """
        scale_factor   — how much image is reduced at each scale (1.05–1.4)
        min_neighbors  — higher = fewer false positives but may miss faces
        min_face_size  — ignore faces smaller than this (filters far background)
        mask_threshold — ratio of lower-face skin pixels; below this = masked
        """
        self.cascade       = cv2.CascadeClassifier(CASCADE_PATH)
        self.scale_factor  = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.mask_threshold = mask_threshold

        # Load flagged persons DB
        self.flagged_db    = self._load_db()
        self.embeddings    = self._load_embeddings()

        print(f"[NxV FaceDetector] Loaded {len(self.flagged_db)} flagged persons.")
        print(f"[NxV FaceDetector] Embedding match: {'ON' if FACE_RECOGNITION_AVAILABLE else 'OFF'}")

    # DB helpers 

    def _load_db(self) -> list:
        """Load flagged persons JSON database."""
        if not os.path.exists(DB_PATH):
            self._create_empty_db()
        with open(DB_PATH, "r") as f:
            return json.load(f).get("persons", [])

    def _create_empty_db(self):
        """Create an empty DB file if none exists."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        empty = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "persons": []
        }
        with open(DB_PATH, "w") as f:
            json.dump(empty, f, indent=2)
        print(f"[NxV FaceDetector] Created empty DB at {DB_PATH}")

    def _load_embeddings(self) -> dict:
        """Load pre-computed face embeddings keyed by person ID."""
        if not os.path.exists(EMBEDDINGS_PATH):
            return {}
        with open(EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)

    def reload_db(self):
        """Hot-reload the DB without restarting (call after adding a new person)."""
        self.flagged_db  = self._load_db()
        self.embeddings  = self._load_embeddings()
        print(f"[NxV FaceDetector] DB reloaded — {len(self.flagged_db)} persons.")

    # Core detection

    def detect(self, frame: np.ndarray) -> list:
        """
        Run face detection on a single frame.
        Returns a list of enriched face record dicts.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improve contrast in dark / backlit scenes

        raw_faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor  = self.scale_factor,
            minNeighbors = self.min_neighbors,
            minSize      = self.min_face_size,
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        results = []

        if len(raw_faces) == 0:
            return results

        for (x, y, w, h) in raw_faces:
            face_roi = frame[y:y+h, x:x+w]

            masked       = self._is_masked(face_roi)
            match        = self._match_face(frame, x, y, w, h)
            threat_score = match.get("threat_score", 0) if match else 0

            # Bump threat score if masked — unknown + masked = higher suspicion
            if masked and threat_score == 0:
                threat_score = 30  # baseline suspicion for masked unknown

            label = self._build_label(masked, match, threat_score)

            results.append({
                'bbox'        : (x, y, w, h),
                'masked'      : masked,
                'match'       : match,
                'threat_score': threat_score,
                'label'       : label,
            })

        return results

    # Mask detection 

    def _is_masked(self, face_roi: np.ndarray) -> bool:
        """
        Simple lower-face skin-pixel check.
        If the bottom half of the face has very few skin-coloured pixels,
        it is likely covered by a mask, scarf, or balaclava.
        """
        if face_roi.size == 0:
            return False

        h, w = face_roi.shape[:2]
        lower_half = face_roi[h // 2:, :]          # bottom 50% of face

        # Convert to HSV for skin detection
        hsv  = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)

        # Skin colour range (works for a broad range of skin tones)
        lower_skin = np.array([0,  20,  70],  dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        skin_ratio = np.sum(mask > 0) / mask.size
        return skin_ratio < self.mask_threshold

    # Face matching

    def _match_face(self, frame: np.ndarray,
                    x: int, y: int, w: int, h: int) -> dict | None:
        """
        Try to match a detected face against the flagged persons database.
        Uses deep embeddings if face_recognition is available,
        otherwise skips matching (returns None).
        """
        if not FACE_RECOGNITION_AVAILABLE or not self.embeddings:
            return None

        # face_recognition expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use the bounding box we already have (faster than re-detecting)
        locations  = [(y, x + w, y + h, x)]   # (top, right, bottom, left)
        encodings  = face_recognition.face_encodings(rgb, known_face_locations=locations)

        if not encodings:
            return None

        query_enc = encodings[0]

        best_match    = None
        best_distance = 0.55  # tolerance — lower = stricter (0.4 strict, 0.6 lenient)

        for person_id, stored_enc in self.embeddings.items():
            distance = face_recognition.face_distance([stored_enc], query_enc)[0]
            if distance < best_distance:
                best_distance = distance
                best_match    = person_id

        if best_match:
            # Look up full record from DB
            for person in self.flagged_db:
                if person["id"] == best_match:
                    return {**person, "match_distance": round(float(best_distance), 3)}

        return None

    # Label builder

    def _build_label(self, masked: bool, match: dict | None, threat_score: int) -> str:
        parts = []
        if match:
            parts.append(match.get("name", "Unknown"))
        else:
            parts.append("Unknown")
        if masked:
            parts.append("MASKED")
        parts.append(f"T:{threat_score}")
        return " | ".join(parts)

    # Draw overlay

    def draw(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        Draw face bounding boxes and labels onto the frame.
        Color coding:
            Green  — unknown, unmasked, low threat
            Orange — masked or moderate threat (30–59)
            Red    — flagged match OR high threat (60+)
        """
        for r in results:
            x, y, w, h   = r['bbox']
            threat        = r['threat_score']
            masked        = r['masked']
            match         = r['match']

            if match or threat >= 60:
                color = (0, 0, 255)      # Red — known dangerous / high threat
            elif masked or threat >= 30:
                color = (0, 165, 255)    # Orange — suspicious
            else:
                color = (0, 255, 0)      # Green — no flags

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = r['label']
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return frame


# DB Management helpers (call from a separate admin script, not from main loop)

def add_person_to_db(name: str,
                     threat_score: int,
                     reason: str,
                     image_paths: list[str],
                     person_id: str | None = None):

    if not FACE_RECOGNITION_AVAILABLE:
        print("[NxV] face_recognition not installed — cannot compute embeddings.")
        return

    import uuid

    pid = person_id or str(uuid.uuid4())[:8]

    # Load DB
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            db = json.load(f)
    else:
        db = {"version": "1.0", "persons": []}

    # Compute average embedding from provided images
    all_encodings = []
    for img_path in image_paths:
        img = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            all_encodings.append(encs[0])
        else:
            print(f"[NxV] No face found in {img_path} — skipping.")

    if not all_encodings:
        print("[NxV] No valid face images found. Person NOT added.")
        return

    avg_encoding = np.mean(all_encodings, axis=0)

    # Save embedding
    embeddings = {}
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
    embeddings[pid] = avg_encoding
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    # Add to JSON DB
    record = {
        "id"          : pid,
        "name"        : name,
        "threat_score": threat_score,
        "reason"      : reason,
        "added"       : datetime.now().isoformat(),
        "sightings"   : 0,
        "last_seen"   : None,
    }
    db["persons"].append(record)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

    print(f"[NxV] Person added — ID: {pid} | Name: {name} | Threat: {threat_score}")