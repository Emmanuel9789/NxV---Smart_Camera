"""
NxV - Social Media Face Search
ai/social_search.py

When an unknown face reaches medium/high threat score,
this module reverse-searches the face image using Google Vision API
to find any public social media profiles, news articles, or
criminal records associated with that face.

Only triggers when:
  - Face is UNKNOWN (not already in flagged DB)
  - Threat score >= 30 (NOTIFY or above)
  - Not searched this person in last 10 minutes

Setup:
  1. Go to https://console.cloud.google.com
  2. Create a project → Enable "Cloud Vision API"
  3. Create an API key → copy it below
  4. Free tier: 1000 requests/month

Install:
  pip install google-cloud-vision requests --break-system-packages
"""

import os
import cv2
import time
import json
import base64
import requests
import tempfile
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
# Set as environment variable: export NXV_GOOGLE_API_KEY="AIzaXXXXX"
GOOGLE_API_KEY = os.environ.get("NXV_GOOGLE_API_KEY", "")

VISION_API_URL = (
    "https://vision.googleapis.com/v1/images:annotate?key={key}"
)

# How long to wait before re-searching the same person (seconds)
SEARCH_COOLDOWN = 600   # 10 minutes

# Minimum confidence score from Vision API to count as a match
MIN_MATCH_SCORE = 0.7

# ── Risk keywords to scan results for ────────────────────────────────────────
# If any of these appear in search results → threat score boosted
DANGER_KEYWORDS = [
    "arrested", "criminal", "convicted", "wanted", "felony",
    "assault", "robbery", "murder", "warrant", "prison",
    "gang", "weapon", "drug", "sex offender", "restraining order",
    "domestic violence", "breaking", "burglar", "suspect"
]


class SocialSearchEngine:
    """
    Reverse-searches unknown faces against the web using Google Vision API.

    Usage:
        engine = SocialSearchEngine()

        # Pass face crop (numpy array) and person ID
        result = engine.search(face_crop, person_id=3)

        if result:
            print(result['matches'])
            print(result['danger_keywords_found'])
            print(result['suggested_threat_boost'])
    """

    def __init__(self):
        self._last_searched = {}   # person_id → timestamp

        if not GOOGLE_API_KEY:
            print("[NxV SocialSearch] WARNING: No Google API key set.")
            print("  Set: export NXV_GOOGLE_API_KEY='your_key'")
            print("  Running in DEMO MODE — will simulate results.")
        else:
            print("[NxV SocialSearch] Google Vision API ready.")

    def should_search(self, person_id: int, threat_score: int,
                      is_known: bool) -> bool:
        """
        Returns True if this person should be searched now.
        """
        if is_known:
            return False   # already in DB, no need to search
        if threat_score < 30:
            return False   # below threshold
        last = self._last_searched.get(person_id, 0)
        return (time.time() - last) >= SEARCH_COOLDOWN

    def search(self, face_crop, person_id: int) -> dict | None:
        """
        Reverse-search a face crop against the web.

        face_crop — numpy array (BGR image of just the face region)
        person_id — tracker person ID

        Returns result dict or None if search failed/skipped.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        self._last_searched[person_id] = time.time()
        print(f"[NxV SocialSearch] Searching P{person_id}...")

        # Encode face crop as base64 JPEG
        encoded = self._encode_face(face_crop)
        if not encoded:
            return None

        # Run Vision API or demo mode
        if GOOGLE_API_KEY:
            raw = self._call_vision_api(encoded)
        else:
            raw = self._demo_result()

        if not raw:
            return None

        # Parse results
        result = self._parse_results(raw, person_id)
        self._log(result)
        return result

    def search_from_frame(self, frame, face_bbox: tuple,
                          person_id: int) -> dict | None:
        """
        Convenience method — crops face from full frame automatically.

        face_bbox — (x, y, w, h) bounding box of the face in the frame
        """
        x, y, w, h = face_bbox

        # Add padding around face for better recognition
        pad   = 20
        x1    = max(0, x - pad)
        y1    = max(0, y - pad)
        x2    = min(frame.shape[1], x + w + pad)
        y2    = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2]
        return self.search(face_crop, person_id)

    # ── Google Vision API ─────────────────────────────────────────────────────

    def _encode_face(self, face_crop) -> str | None:
        """Convert face numpy array to base64 JPEG string."""
        try:
            _, buffer = cv2.imencode('.jpg', face_crop,
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[NxV SocialSearch] Encode error: {e}")
            return None

    def _call_vision_api(self, image_b64: str) -> dict | None:
        """
        Call Google Vision API with WEB_DETECTION feature.
        This finds matching faces, pages, and similar images on the web.
        """
        url     = VISION_API_URL.format(key=GOOGLE_API_KEY)
        payload = {
            "requests": [{
                "image": {"content": image_b64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 10},
                    {"type": "SAFE_SEARCH_DETECTION"},
                ]
            }]
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("responses", [{}])[0]
        except requests.exceptions.Timeout:
            print("[NxV SocialSearch] API timeout.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[NxV SocialSearch] API error: {e}")
            return None

    def _demo_result(self) -> dict:
        """Returns a simulated result for testing without API key."""
        return {
            "webDetection": {
                "webEntities": [
                    {"description": "Unknown person", "score": 0.6},
                ],
                "pagesWithMatchingImages": [],
                "visuallySimilarImages": [],
                "bestGuessLabels": [
                    {"label": "person"}
                ]
            }
        }

    # ── Result parsing ────────────────────────────────────────────────────────

    def _parse_results(self, raw: dict, person_id: int) -> dict:
        """
        Parse Vision API response into a clean NxV result dict.
        """
        web = raw.get("webDetection", {})

        # Web entities (labels Google assigns to the face/person)
        entities = [
            e.get("description", "")
            for e in web.get("webEntities", [])
            if e.get("score", 0) >= MIN_MATCH_SCORE
        ]

        # Pages where this face appears
        pages = [
            {
                "url"  : p.get("url", ""),
                "title": p.get("pageTitle", ""),
            }
            for p in web.get("pagesWithMatchingImages", [])
        ]

        # Best guess labels
        labels = [
            l.get("label", "")
            for l in web.get("bestGuessLabels", [])
        ]

        # Combine all text for keyword scanning
        all_text = " ".join(
            entities + labels +
            [p["title"] for p in pages] +
            [p["url"] for p in pages]
        ).lower()

        # Scan for danger keywords
        found_keywords = [
            kw for kw in DANGER_KEYWORDS
            if kw in all_text
        ]

        # Calculate threat boost based on findings
        threat_boost = 0
        if found_keywords:
            threat_boost = min(40, len(found_keywords) * 10)
        if pages:
            threat_boost += min(10, len(pages) * 2)

        # Build name guess from entities
        name_guess = None
        for e in entities:
            # Heuristic: if entity has 2 words and starts with capital → likely a name
            words = e.strip().split()
            if (len(words) == 2 and
                    words[0][0].isupper() and words[1][0].isupper()):
                name_guess = e
                break

        return {
            "person_id"             : person_id,
            "searched_at"           : datetime.now().isoformat(),
            "name_guess"            : name_guess,
            "entities"              : entities,
            "pages_found"           : pages[:5],   # top 5 only
            "labels"                : labels,
            "danger_keywords_found" : found_keywords,
            "suggested_threat_boost": threat_boost,
            "has_web_presence"      : bool(pages or entities),
            "demo_mode"             : not bool(GOOGLE_API_KEY),
        }

    def _log(self, result: dict):
        pid      = result["person_id"]
        keywords = result["danger_keywords_found"]
        boost    = result["suggested_threat_boost"]
        name     = result["name_guess"] or "unknown"
        pages    = len(result["pages_found"])

        print(f"[NxV SocialSearch] P{pid} results:")
        print(f"  Name guess : {name}")
        print(f"  Pages found: {pages}")
        print(f"  Danger kws : {keywords or 'none'}")
        print(f"  Score boost: +{boost}")


def save_search_result(result: dict, base_dir: str = "evidence"):
    """
    Save social search result to the evidence folder for this incident.
    """
    if not result:
        return
    pid      = result["person_id"]
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(base_dir, f"social_search_P{pid}_{ts}.json")
    os.makedirs(base_dir, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[NxV SocialSearch] Result saved → {filename}")
