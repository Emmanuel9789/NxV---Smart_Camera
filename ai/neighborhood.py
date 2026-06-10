"""

Three-layer intelligence sharing:

  Layer 1 — Global network (all NxV cameras worldwide)
    → Dangerous individuals shared across all cameras
    → Score boost on sight, NO escalation unless they act
    → Exception: grave crimes = escalate on sight

  Layer 2 — Street trust circle (nearby cameras, opt-in)
    → Neighbors share trusted faces with each other
    → Known neighbor = never flagged
    → Neighbor's family/friends = lower suspicion score

  Layer 3 — Social/web intelligence
    → Google Vision results feed into persistent DB record
    → Criminal record found = score boost on every future sighting

Crime severity levels:
  MINOR    → +15 score boost, no escalation
  MODERATE → +25 score boost, no escalation
  SEVERE   → +35 score boost, NOTIFY on sight
  GRAVE    → +60 score boost, ALERT on sight
  CRITICAL → +75 score boost, EMERGENCY on sight
             (murder, terrorism, active warrant etc)
"""

import os
import json
import time
import uuid
import pickle
import hashlib
import threading
import numpy as np
from datetime import datetime

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import face_recognition
    FR_OK = True
except ImportError:
    FR_OK = False


# Config 
RELAY_URL    = os.environ.get("NXV_RELAY_URL", "")
CAMERA_ID    = os.environ.get("NXV_CAMERA_ID", str(uuid.uuid4())[:8])
NETWORK_KEY  = os.environ.get("NXV_NETWORK_KEY", "")
SYNC_INTERVAL= 60
MATCH_THRESH = 0.5


# Crime severity → score boost + escalation
SEVERITY_MAP = { #Hash map
    "MINOR"   : {"boost": 15,  "escalate": False, "min_escalation": "NONE"},
    "MODERATE": {"boost": 25,  "escalate": False, "min_escalation": "NONE"},
    "SEVERE"  : {"boost": 35,  "escalate": True,  "min_escalation": "NOTIFY"},
    "GRAVE"   : {"boost": 60,  "escalate": True,  "min_escalation": "ALERT"},
    "CRITICAL": {"boost": 75,  "escalate": True,  "min_escalation": "EMERGENCY"},
}

# Network source types
SOURCE_LOCAL         = "local"
SOURCE_NETWORK       = "network"
SOURCE_SOCIAL_SEARCH = "social_search"
SOURCE_STREET_CIRCLE = "street_circle"


class NetworkThreatRecord:
    """
    Represents a person flagged in the network with full context.
    """
    def __init__(self, person_id: str, name: str,
                 crime_severity: str = "MINOR",
                 source: str = SOURCE_NETWORK,
                 active_warrant: bool = False,
                 reason: str = "",
                 threat_score: int = 50,
                 embedding=None,
                 camera_id: str = ""):
        self.person_id      = person_id
        self.name           = name
        self.crime_severity = crime_severity.upper()
        self.source         = source
        self.active_warrant = active_warrant
        self.reason         = reason
        self.base_score     = threat_score
        self.embedding      = embedding
        self.source_camera  = camera_id
        self.added_at       = datetime.now().isoformat()
        self.last_seen      = None
        self.sightings      = 0

    @property
    def score_boost(self) -> int:
        """Score to add when this person is detected."""
        base = SEVERITY_MAP.get(self.crime_severity, SEVERITY_MAP["MINOR"])["boost"]
        if self.active_warrant:
            base += 20
        return base

    @property
    def should_escalate_on_sight(self) -> bool:
        """Whether detection alone triggers escalation (before any behavior)."""
        return SEVERITY_MAP.get(
            self.crime_severity, SEVERITY_MAP["MINOR"]
        )["escalate"]

    @property
    def min_escalation(self) -> str:
        """Minimum escalation level when this person is seen."""
        return SEVERITY_MAP.get(
            self.crime_severity, SEVERITY_MAP["MINOR"]
        )["min_escalation"]

    def to_dict(self) -> dict:
        return {
            "person_id"     : self.person_id,
            "name"          : self.name,
            "crime_severity": self.crime_severity,
            "source"        : self.source,
            "active_warrant": self.active_warrant,
            "reason"        : self.reason,
            "base_score"    : self.base_score,
            "score_boost"   : self.score_boost,
            "source_camera" : self.source_camera,
            "added_at"      : self.added_at,
            "last_seen"     : self.last_seen,
            "sightings"     : self.sightings,
            "escalate_sight": self.should_escalate_on_sight,
            "min_escalation": self.min_escalation,
        }


class TrustedCircleMember:
    """
    A trusted person shared in the street circle.
    Seeing this person at a neighbor's camera = never flag.
    """
    def __init__(self, person_id: str, name: str,
                 relationship: str = "neighbor",
                 shared_by: str = "",
                 embedding=None):
        self.person_id   = person_id
        self.name        = name
        self.relationship= relationship
        self.shared_by   = shared_by   # camera_id of the sharer
        self.embedding   = embedding
        self.added_at    = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "person_id"   : self.person_id,
            "name"        : self.name,
            "relationship": self.relationship,
            "shared_by"   : self.shared_by,
            "added_at"    : self.added_at,
        }


class NeighborhoodNetwork:
    """
    Full tiered neighborhood threat network.

    Usage:
        net = NeighborhoodNetwork()
        net.start()

        # Check face against all layers
        result = net.check_face(encoding, behavior_score=0, has_weapon=False)
        if result:
            print(result['score_boost'], result['should_escalate'])

        # Share a dangerous person
        net.share_threat(record)

        # Share a trusted person with street circle
        net.share_trusted(member)
    """

    def __init__(self):
        self._threats       = {}   # person_id → NetworkThreatRecord
        self._trusted       = {}   # person_id → TrustedCircleMember
        self._lock          = threading.Lock()
        self._enabled       = bool(RELAY_URL and NETWORK_KEY)
        self._sync_thread   = None

        if self._enabled:
            print(f"[NxV Network] Online → {RELAY_URL}")
            print(f"[NxV Network] Camera: {CAMERA_ID}")
        else:
            print("[NxV Network] Offline (set NXV_RELAY_URL + NXV_NETWORK_KEY)")

    def start(self):
        if not self._enabled:
            return
        self._sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True
        )
        self._sync_thread.start()

    # Core face check 

    def check_face(self, encoding,
                   behavior_score: int = 0,
                   has_weapon: bool = False,
                   is_at_door: bool = False) -> dict | None:
        """
        Check a face against ALL network layers.

        Returns result dict or None if not found.

        Result contains:
          matched         → True
          layer           → which layer matched (threat/trusted/street)
          name            → person name
          score_boost     → points to add to threat score
          should_escalate → whether to escalate on sight
          min_escalation  → minimum escalation level
          reason          → why this person is flagged
          override        → True if behavior should be ignored
        """
        if not FR_OK:
            return None

        # Layer 2 check: Street trust circle 
        # Check trusted first — trusted person = never flag
        trusted_match = self._check_trusted(encoding)
        if trusted_match:
            return {
                "matched"        : True,
                "layer"          : SOURCE_STREET_CIRCLE,
                "name"           : trusted_match.name,
                "score_boost"    : -50,    # NEGATIVE boost = reduces threat score
                "should_escalate": False,
                "min_escalation" : "NONE",
                "reason"         : f"Trusted — shared by {trusted_match.shared_by}",
                "override"       : False,
                "relationship"   : trusted_match.relationship,
            }

        # Layer 1 check: Global threat network 
        threat_match = self._check_threats(encoding)
        if threat_match:
            record = threat_match

            # Behavioral context — even flagged person needs to DO something
            # unless crime is GRAVE or CRITICAL
            should_escalate = record.should_escalate_on_sight

            # Gravity override — weapon or break-in = always escalate
            if has_weapon or is_at_door:
                should_escalate = True

            # Pure passing motion — don't escalate even for grave crimes
            # behavior_score < 10 means they're just passing by
            if behavior_score < 10 and not has_weapon and not is_at_door:
                should_escalate = False

            record.sightings += 1
            record.last_seen = datetime.now().isoformat()

            return {
                "matched"        : True,
                "layer"          : record.source,
                "name"           : record.name,
                "score_boost"    : record.score_boost,
                "should_escalate": should_escalate,
                "min_escalation" : record.min_escalation if should_escalate else "NONE",
                "reason"         : record.reason,
                "crime_severity" : record.crime_severity,
                "active_warrant" : record.active_warrant,
                "override"       : should_escalate,
                "sightings"      : record.sightings,
            }

        return None

    #  Share threat 

    def share_threat(self, record: NetworkThreatRecord) -> bool:
        """Share a dangerous person to the global network."""
        # Add locally first
        with self._lock:
            self._threats[record.person_id] = record

        if not self._enabled or not REQUESTS_OK:
            print(f"[NxV Network] Stored locally: {record.name}")
            return True

        try:
            enc_list = record.embedding.tolist() \
                if record.embedding is not None else []

            payload = {
                "camera_id"     : CAMERA_ID,
                "person_id"     : record.person_id,
                "name"          : record.name,
                "crime_severity": record.crime_severity,
                "active_warrant": record.active_warrant,
                "reason"        : record.reason,
                "threat_score"  : record.base_score,
                "embedding"     : enc_list,
                "source"        : record.source,
                "shared_at"     : datetime.now().isoformat(),
                "sig"           : self._sign(record.person_id),
            }

            r = requests.post(
                f"{RELAY_URL}/share",
                json    = payload,
                headers = {"X-Network-Key": NETWORK_KEY},
                timeout = 10,
            )
            success = r.status_code == 200
            if success:
                print(f"[NxV Network] Shared: {record.name} "
                      f"({record.crime_severity})")
            return success

        except Exception as e:
            print(f"[NxV Network] Share error: {e}")
            return False

    def share_trusted(self, member: TrustedCircleMember) -> bool:
        """Share a trusted person to the street circle."""
        with self._lock:
            self._trusted[member.person_id] = member

        if not self._enabled or not REQUESTS_OK:
            return True

        try:
            enc_list = member.embedding.tolist() \
                if member.embedding is not None else []

            payload = {
                "camera_id"   : CAMERA_ID,
                "person_id"   : member.person_id,
                "name"        : member.name,
                "relationship": member.relationship,
                "embedding"   : enc_list,
                "type"        : "trusted",
                "sig"         : self._sign(member.person_id),
            }

            r = requests.post(
                f"{RELAY_URL}/share_trusted",
                json    = payload,
                headers = {"X-Network-Key": NETWORK_KEY},
                timeout = 10,
            )
            return r.status_code == 200

        except Exception as e:
            print(f"[NxV Network] Share trusted error: {e}")
            return False

    def add_from_social_search(self, result: dict,
                               encoding=None) -> NetworkThreatRecord | None:
        """
        Add a person to network DB based on social media search results.

        result — output from SocialSearchEngine.search()
        """
        if not result or not result.get("danger_keywords_found"):
            return None

        keywords  = result["danger_keywords_found"]
        name      = result.get("name_guess") or "Unknown"

        # Determine severity from keywords
        severity = "MINOR"
        if any(k in keywords for k in ["murder", "terrorism", "gang", "warrant"]):
            severity = "CRITICAL"
        elif any(k in keywords for k in ["assault", "robbery", "convicted", "prison"]):
            severity = "GRAVE"
        elif any(k in keywords for k in ["arrested", "criminal", "felony"]):
            severity = "SEVERE"
        elif any(k in keywords for k in ["drug", "restraining"]):
            severity = "MODERATE"

        record = NetworkThreatRecord(
            person_id      = str(uuid.uuid4())[:8],
            name           = name,
            crime_severity = severity,
            source         = SOURCE_SOCIAL_SEARCH,
            active_warrant = "warrant" in keywords,
            reason         = f"Social search: {', '.join(keywords[:3])}",
            threat_score   = result.get("suggested_threat_boost", 20),
            embedding      = encoding,
            camera_id      = CAMERA_ID,
        )

        with self._lock:
            self._threats[record.person_id] = record

        print(f"[NxV Network] Social search added: {name} "
              f"({severity}) — {keywords}")
        return record

    # Internals 

    def _check_threats(self, encoding) -> NetworkThreatRecord | None:
        with self._lock:
            records = list(self._threats.values())

        if not records:
            return None

        known_encs = []
        valid      = []
        for r in records:
            if r.embedding is not None:
                try:
                    known_encs.append(np.array(r.embedding))
                    valid.append(r)
                except Exception:
                    continue

        if not known_encs:
            return None

        try:
            distances = face_recognition.face_distance(known_encs, encoding)
            best_idx  = int(np.argmin(distances))
            if distances[best_idx] < MATCH_THRESH:
                return valid[best_idx]
        except Exception:
            pass
        return None

    def _check_trusted(self, encoding) -> TrustedCircleMember | None:
        with self._lock:
            members = list(self._trusted.values())

        if not members:
            return None

        known_encs = []
        valid      = []
        for m in members:
            if m.embedding is not None:
                try:
                    known_encs.append(np.array(m.embedding))
                    valid.append(m)
                except Exception:
                    continue

        if not known_encs:
            return None

        try:
            distances = face_recognition.face_distance(known_encs, encoding)
            best_idx  = int(np.argmin(distances))
            if distances[best_idx] < MATCH_THRESH:
                return valid[best_idx]
        except Exception:
            pass
        return None

    def _sync_loop(self):
        while True:
            time.sleep(SYNC_INTERVAL)
            self._pull_from_relay()

    def _pull_from_relay(self):
        if not self._enabled or not REQUESTS_OK:
            return
        try:
            # Pull threats
            r = requests.get(
                f"{RELAY_URL}/threats",
                headers = {"X-Network-Key": NETWORK_KEY,
                           "X-Camera-ID"  : CAMERA_ID},
                timeout = 10,
            )
            if r.status_code == 200:
                data = r.json()
                with self._lock:
                    for t in data.get("threats", []):
                        pid = t.get("person_id")
                        if pid and pid not in self._threats:
                            enc = t.pop("embedding", None)
                            record = NetworkThreatRecord(
                                person_id      = pid,
                                name           = t.get("name", "Unknown"),
                                crime_severity = t.get("crime_severity", "MINOR"),
                                source         = SOURCE_NETWORK,
                                active_warrant = t.get("active_warrant", False),
                                reason         = t.get("reason", ""),
                                threat_score   = t.get("threat_score", 50),
                                embedding      = np.array(enc) if enc else None,
                                camera_id      = t.get("camera_id", ""),
                            )
                            self._threats[pid] = record

            # Pull trusted circle
            rt = requests.get(
                f"{RELAY_URL}/trusted",
                headers = {"X-Network-Key": NETWORK_KEY,
                           "X-Camera-ID"  : CAMERA_ID},
                timeout = 10,
            )
            if rt.status_code == 200:
                data = rt.json()
                with self._lock:
                    for m in data.get("trusted", []):
                        pid = m.get("person_id")
                        if pid and pid not in self._trusted:
                            enc    = m.pop("embedding", None)
                            member = TrustedCircleMember(
                                person_id    = pid,
                                name         = m.get("name", "Unknown"),
                                relationship = m.get("relationship","neighbor"),
                                shared_by    = m.get("camera_id", ""),
                                embedding    = np.array(enc) if enc else None,
                            )
                            self._trusted[pid] = member

        except Exception as e:
            print(f"[NxV Network] Sync error: {e}")

    def _sign(self, data: str) -> str:
        return hashlib.sha256(
            (data + NETWORK_KEY).encode()
        ).hexdigest()[:16]

    # Public getters 

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def threat_count(self) -> int:
        return len(self._threats)

    @property
    def trusted_count(self) -> int:
        return len(self._trusted)

    def get_all_threats(self) -> list:
        with self._lock:
            return [r.to_dict() for r in self._threats.values()]

    def get_all_trusted(self) -> list:
        with self._lock:
            return [m.to_dict() for m in self._trusted.values()]

    def remove_threat(self, person_id: str):
        with self._lock:
            self._threats.pop(person_id, None)

    def remove_trusted(self, person_id: str):
        with self._lock:
            self._trusted.pop(person_id, None)
