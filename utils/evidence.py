"""
NxV - Evidence Packager
utils/evidence.py

Saves timestamped video clips and threat log JSON whenever
an alert fires. Produces legal-ready evidence packages.

Each incident creates a folder:
  evidence/
    └── 2026-04-21_22-31-05_EMERGENCY/
          ├── clip.avi          ← video frames from incident
          ├── threat_log.json   ← full threat breakdown
          └── summary.txt       ← human-readable police report

Usage:
    packager = EvidencePackager()
    packager.start_recording(frame)   # call when threat detected
    packager.add_frame(frame)         # call each frame while threat active
    packager.save(threat_score)       # call when incident ends or escalates
"""

import cv2
import json
import os
import hashlib
import time
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_DIR   = os.path.join(BASE_DIR, "evidence")
MAX_CLIP_SECS  = 60       # maximum clip length in seconds
FPS            = 10       # frames per second for saved clip
FRAME_SIZE     = (320, 240)
HOME_ADDRESS   = "your home address"   # update this


class EvidencePackager:
    """
    Records video clips and threat logs for legal evidence.
    """

    def __init__(self):
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        self._frames        = []
        self._recording     = False
        self._start_time    = None
        self._incident_dir  = None
        self._max_frames    = MAX_CLIP_SECS * FPS

    # ── Public API ────────────────────────────────────────────────────────────

    def start_recording(self, frame):
        """
        Start a new incident recording.
        Call this the moment a non-NONE threat is detected.
        """
        if self._recording:
            return   # already recording

        self._recording  = True
        self._start_time = datetime.now()
        self._frames     = [frame.copy()]

        # Create incident folder
        ts = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self._incident_dir = os.path.join(EVIDENCE_DIR, f"{ts}_incident")
        os.makedirs(self._incident_dir, exist_ok=True)

        print(f"[NxV Evidence] Recording started → {self._incident_dir}")

    def add_frame(self, frame):
        """
        Add a frame to the current recording.
        Call this every frame while a threat is active.
        """
        if not self._recording:
            return
        if len(self._frames) >= self._max_frames:
            return   # clip length cap reached
        self._frames.append(frame.copy())

    def save(self, threat_score, force: bool = False) -> str | None:
        """
        Save the recorded clip and threat log to disk.
        Call this when an ALERT or EMERGENCY fires, or when the threat clears.

        Returns the incident directory path, or None if nothing was saved.
        """
        if not self._recording:
            return None
        if len(self._frames) < 3 and not force:
            return None   # too short to be meaningful

        clip_path    = self._save_clip()
        log_path     = self._save_log(threat_score, clip_path)
        summary_path = self._save_summary(threat_score, clip_path)

        self._recording = False
        self._frames    = []

        print(f"[NxV Evidence] Incident saved → {self._incident_dir}")
        print(f"  Clip   : {clip_path}")
        print(f"  Log    : {log_path}")
        print(f"  Summary: {summary_path}")

        return self._incident_dir

    def stop(self):
        """Stop recording without saving."""
        self._recording = False
        self._frames    = []
        print("[NxV Evidence] Recording stopped (not saved).")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def clip_path(self) -> str | None:
        if self._incident_dir:
            return os.path.join(self._incident_dir, "clip.avi")
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_clip(self) -> str:
        """Write frames to an AVI file and return the path."""
        path   = os.path.join(self._incident_dir, "clip.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(path, fourcc, FPS, FRAME_SIZE)

        for frame in self._frames:
            resized = cv2.resize(frame, FRAME_SIZE)
            writer.write(resized)

        writer.release()

        # Compute SHA256 hash for tamper-detection
        sha256 = self._hash_file(path)
        hash_path = path + ".sha256"
        with open(hash_path, "w") as f:
            f.write(sha256)

        return path

    def _save_log(self, threat_score, clip_path: str) -> str:
        """Write full threat breakdown to JSON."""
        path = os.path.join(self._incident_dir, "threat_log.json")

        log = {
            "nxv_version"   : "1.0",
            "incident_time" : self._start_time.isoformat(),
            "saved_at"      : datetime.now().isoformat(),
            "duration_secs" : round(len(self._frames) / FPS, 1),
            "frame_count"   : len(self._frames),
            "address"       : HOME_ADDRESS,
            "threat"        : {
                "person_id"   : threat_score.person_id,
                "final_score" : threat_score.final_score,
                "escalation"  : threat_score.escalation,
                "flags"       : threat_score.all_flags,
                "breakdown"   : threat_score.breakdown,
                "user_away"   : threat_score.user_away,
            },
            "clip_path"     : clip_path,
            "clip_hash_sha256": self._hash_file(clip_path),
        }

        with open(path, "w") as f:
            json.dump(log, f, indent=2)

        return path

    def _save_summary(self, threat_score, clip_path: str) -> str:
        """Write a human-readable police report summary."""
        path     = os.path.join(self._incident_dir, "summary.txt")
        dt       = self._start_time.strftime("%B %d, %Y at %I:%M %p")
        flags    = ", ".join(threat_score.all_flags) or "suspicious activity"
        duration = round(len(self._frames) / FPS, 1)

        summary = f"""NxV SECURITY INCIDENT REPORT
{"=" * 50}
Date/Time  : {dt}
Address    : {HOME_ADDRESS}
Duration   : {duration} seconds
{"=" * 50}

THREAT ASSESSMENT
-----------------
Final Score  : {threat_score.final_score} / 100
Escalation   : {threat_score.escalation}
Signals      : {flags}
User Away    : {"Yes" if threat_score.user_away else "No"}

SCORE BREAKDOWN
---------------
Behavior score : {threat_score.breakdown.get('behavior', 0)}
Violence score : {threat_score.breakdown.get('violence', 0)}
Weapon score   : {threat_score.breakdown.get('weapon', 0)}
Face score     : {threat_score.breakdown.get('face', 0)}
Time risk      : {threat_score.breakdown.get('time', 0)} ({threat_score.breakdown.get('time_label', 'unknown')})

EVIDENCE
--------
Video clip : {clip_path}
Hash (SHA256): {self._hash_file(clip_path)}

STATEMENT FOR LAW ENFORCEMENT
------------------------------
On {dt}, the NxV AI security system at {HOME_ADDRESS}
detected a threat with a score of {threat_score.final_score}/100.
The following signals were identified: {flags}.
A {duration}-second video clip has been preserved and
cryptographically hashed for evidentiary integrity.

{"=" * 50}
Generated by NxV — AI-Powered Intent-Aware Security System
"""

        with open(path, "w") as f:
            f.write(summary)

        return path

    @staticmethod
    def _hash_file(path: str) -> str:
        """Compute SHA256 hash of a file for tamper detection."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return "hash_unavailable"
