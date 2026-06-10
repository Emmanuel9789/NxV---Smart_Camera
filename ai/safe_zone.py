"""
Registers trusted faces so NxV never flags them.
Works from the app (live camera) or CLI (for testing).

Flow:
  1. App sends POST /register_face with name
  2. Pi captures 10 frames from camera
  3. Detects face in each frame
  4. Computes average face embedding
  5. Saves to DB + embeddings file
  6. Face detector reloads
  7. Returns success

"""

import cv2
import numpy as np
import os
import pickle
import time
import argparse
from datetime import datetime

try:
    import face_recognition
    FR_OK = True
except ImportError:
    FR_OK = False
    print("[NxV SafeZone] face_recognition not installed")

EMBEDDINGS_PATH = '/home/emmanuel/camera_project/datasets/face_embeddings.pkl'
FACES_DIR       = '/home/emmanuel/camera_project/datasets/trusted_faces'


class SafeZoneManager:
    """
    Manages trusted face registration and lookup.
    """

    def __init__(self):
        os.makedirs(FACES_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        self.embeddings = self._load_embeddings()
        print(f"[NxV SafeZone] Loaded {len(self.embeddings)} trusted faces")

    def _load_embeddings(self) -> dict:
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_embeddings(self):
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def register_from_frames(self, frames: list, person_id: str,
                              name: str) -> dict:
        """
        Register a trusted face from a list of frames.

        frames    — list of numpy arrays (BGR frames from camera)
        person_id — unique ID for this person
        name      — display name

        Returns result dict with success, frames_used, message
        """
        if not FR_OK:
            return {
                'success': False,
                'message': 'face_recognition not installed',
                'frames_used': 0,
            }

        all_encodings = []
        frames_used   = 0

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect face locations first (faster than full encoding)
            locations = face_recognition.face_locations(rgb, model='hog')
            if not locations:
                continue
            # Get encoding for largest face
            encodings = face_recognition.face_encodings(rgb, locations)
            if encodings:
                all_encodings.append(encodings[0])
                frames_used += 1

        if not all_encodings:
            return {
                'success': False,
                'message': 'No face detected in any frame. Please try again — face the camera directly in good light.',
                'frames_used': 0,
            }

        if len(all_encodings) < 3:
            return {
                'success': False,
                'message': f'Only {len(all_encodings)} clear frames captured. Need at least 3. Try better lighting.',
                'frames_used': frames_used,
            }

        # Average embedding for robustness
        avg_encoding = np.mean(all_encodings, axis=0)

        # Save embedding
        self.embeddings[person_id] = {
            'encoding': avg_encoding,
            'name'    : name,
            'added_at': datetime.now().isoformat(),
        }
        self._save_embeddings()

        # Save a sample face crop for reference
        self._save_face_crop(frames[0], name, person_id)

        print(f"[NxV SafeZone] Registered: {name} (ID: {person_id}) "
              f"— {frames_used} frames used")

        return {
            'success'    : True,
            'message'    : f'{name} registered successfully. NxV will no longer flag you.',
            'frames_used': frames_used,
            'person_id'  : person_id,
        }

    def register_from_camera(self, camera, name: str,
                              person_id: str, duration: float = 5.0) -> dict:
        """
        Register from live camera — captures frames for `duration` seconds.
        Used by CLI or direct Pi registration.
        """
        print(f"[NxV SafeZone] Capturing {name} for {duration}s...")
        frames  = []
        start   = time.time()
        fps     = 10
        delay   = 1.0 / fps

        while time.time() - start < duration:
            t     = time.time()
            frame = camera.get_frame()
            frames.append(frame)
            elapsed = time.time() - t
            if elapsed < delay:
                time.sleep(delay - elapsed)

        print(f"[NxV SafeZone] Captured {len(frames)} frames")
        return self.register_from_frames(frames, person_id, name)

    def is_trusted(self, face_encoding) -> tuple:
        """
        Check if a face encoding matches any trusted person.
        Returns (is_trusted, name, person_id) or (False, None, None)
        """
        if not self.embeddings:
            return False, None, None

        known_encodings = [v['encoding'] for v in self.embeddings.values()]
        known_ids       = list(self.embeddings.keys())

        distances = face_recognition.face_distance(
            known_encodings, face_encoding
        )

        best_idx  = int(np.argmin(distances))
        best_dist = distances[best_idx]

        if best_dist < 0.5:   # threshold — lower = stricter
            pid  = known_ids[best_idx]
            name = self.embeddings[pid]['name']
            return True, name, pid

        return False, None, None

    def remove_trusted(self, person_id: str) -> bool:
        if person_id in self.embeddings:
            del self.embeddings[person_id]
            self._save_embeddings()
            print(f"[NxV SafeZone] Removed: {person_id}")
            return True
        return False

    def get_trusted_list(self) -> list:
        return [
            {
                'id'      : pid,
                'name'    : v['name'],
                'added_at': v['added_at'],
            }
            for pid, v in self.embeddings.items()
        ]

    def reload(self):
        self.embeddings = self._load_embeddings()
        print(f"[NxV SafeZone] Reloaded — {len(self.embeddings)} faces")

    def _save_face_crop(self, frame, name: str, person_id: str):
        """Save a reference face crop image."""
        try:
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model='hog')
            if locations:
                top, right, bottom, left = locations[0]
                crop = frame[top:bottom, left:right]
                path = os.path.join(FACES_DIR, f'{person_id}_{name}.jpg')
                cv2.imwrite(path, crop)
        except Exception:
            pass


# CLI registration (for testing without app) 

def cli_register():
    parser = argparse.ArgumentParser(description='NxV Safe Zone Registration')
    parser.add_argument('--name',    required=True, help='Person name')
    parser.add_argument('--seconds', type=float, default=5.0,
                        help='Capture duration in seconds')
    parser.add_argument('--id',      default=None,
                        help='Person ID (auto-generated if not set)')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '/home/emmanuel/camera_project')

    from camera.input import Camera
    from utils.db import add_person
    import uuid

    person_id = args.id or str(uuid.uuid4())[:8]
    manager   = SafeZoneManager()
    camera    = Camera()

    print(f"\n[NxV SafeZone] Registering: {args.name}")
    print(f"  Face the camera and hold still...")
    print(f"  Capturing for {args.seconds} seconds...\n")

    time.sleep(1)   # give person time to get ready

    result = manager.register_from_camera(
        camera, args.name, person_id, duration=args.seconds
    )

    if result['success']:
        # Save to DB as trusted person
        add_person(
            id           = person_id,
            name         = args.name,
            threat_score = 0,
            reason       = 'Trusted — registered via safe zone',
            is_trusted   = True,
        )
        print(f"\n✓ {result['message']}")
        print(f"  Frames used: {result['frames_used']}")
        print(f"  Person ID  : {person_id}")
    else:
        print(f"\n✗ {result['message']}")

    camera.picam2.stop()


if __name__ == '__main__':
    cli_register()
