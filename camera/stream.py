"""
NxV - Live Stream (fully wired)
camera/stream.py

Full pipeline per frame:
  1. Grab frame from PiCamera2
  2. Motion detection
  3. YOLO weapon detection
  4. Face detection
  5. Person tracking
  6. Behavior analysis
  7. Violence detection
  8. Threat scoring
  9. Escalation (SMS / deterrent)
  10. Draw all overlays onto frame
  11. Stream to browser via Flask MJPEG
"""

import time
import cv2
from flask import Flask, Response

from detection.person     import detect_weapons
from detection.face       import FaceDetector
from tracking.tracker     import PersonTracker
from ai.behavior          import BehaviorAnalyzer
from ai.violence          import ViolenceDetector, ViolenceResult
from scoring.threat_score import ThreatScoreEngine
from alerts.escalation    import EscalationEngine

app = Flask(__name__)

# ── Module instances ──────────────────────────────────────────────────────────
face_detector     = FaceDetector()
person_tracker    = PersonTracker()
behavior_analyzer = BehaviorAnalyzer()
violence_detector = ViolenceDetector()
threat_engine     = ThreatScoreEngine()
escalation_engine = EscalationEngine()

# ── Config ────────────────────────────────────────────────────────────────────
# Door zone (x, y, w, h) in pixels — adjust to where your door appears
DOOR_ZONE = (100, 60, 120, 180)

FACE_EVERY_N_FRAMES     = 3
BEHAVIOR_EVERY_N_FRAMES = 5
ESCALATION_COOLDOWN     = 10

# ── Runtime state ─────────────────────────────────────────────────────────────
_frame_count        = 0
_last_escalation    = 0
_last_face_results  = []
_last_threat_scores = []
_user_away          = False


def generate_frames():
    global _frame_count, _last_escalation
    global _last_face_results, _last_threat_scores, _user_away

    while True:
        t_start       = time.time()
        _frame_count += 1

        # 1 ── Grab frame ──────────────────────────────────────────────────────
        frame = app.camera.get_frame()

        # 2 ── Motion detection ────────────────────────────────────────────────
        motion_detected, motion_boxes = app.motion_detector.detect(frame)

        # 3 ── Weapon detection ────────────────────────────────────────────────
        weapon_detections = []
        if motion_detected:
            weapon_detections = detect_weapons(frame, conf_threshold=0.25)

        # 4 ── Face detection (every N frames) ─────────────────────────────────
        if motion_detected and _frame_count % FACE_EVERY_N_FRAMES == 0:
            _last_face_results = face_detector.detect(frame)
        face_results = _last_face_results

        # 5 ── Person tracking ─────────────────────────────────────────────────
        tracking_boxes = list(motion_boxes)
        for det in weapon_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            tracking_boxes.append((x1, y1, x2 - x1, y2 - y1))

        persons = person_tracker.update(tracking_boxes, timestamp=time.time())

        # 6 + 7 ── Behavior + violence (every N frames) ────────────────────────
        behavior_results = {}
        violence_result  = ViolenceResult(0, [])

        if persons and _frame_count % BEHAVIOR_EVERY_N_FRAMES == 0:
            behavior_results = behavior_analyzer.analyze(persons)
            violence_result  = violence_detector.analyze(persons)

        # 8 ── Threat scoring ──────────────────────────────────────────────────
        if persons:
            _last_threat_scores = threat_engine.score_from_results(
                persons           = persons,
                behavior_results  = behavior_results,
                violence_result   = violence_result,
                weapon_detections = weapon_detections,
                face_results      = face_results,
                door_zone         = DOOR_ZONE,
                user_away         = _user_away,
            )
        threat_scores = _last_threat_scores

        # 9 ── Escalation (rate limited) ───────────────────────────────────────
        now = time.time()
        if (threat_scores and
                now - _last_escalation >= ESCALATION_COOLDOWN):
            top = threat_scores[0]
            if top.escalation != "NONE":
                escalation_engine.handle(top, user_away=_user_away)
                _last_escalation = now

        # 10 ── Draw overlays ──────────────────────────────────────────────────

        # Motion boxes
        for (x, y, bw, bh) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                          (0, 200, 80), 1)

        # Weapon boxes
        for det in weapon_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame,
                        f"WEAPON {float(det['conf']):.2f}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                        cv2.LINE_AA)

        # Face overlays
        frame = face_detector.draw(frame, face_results)

        # Person tracker trails + ID labels
        frame = person_tracker.draw(frame, persons)

        # Threat score badge per person
        ESCALATION_COLORS = {
            "EMERGENCY": (0, 0, 255),
            "ALERT"    : (0, 100, 255),
            "NOTIFY"   : (0, 200, 255),
            "NONE"     : (0, 200, 80),
        }
        for ts in threat_scores:
            person = person_tracker.get_person(ts.person_id)
            if person is None:
                continue
            x, y, bw, bh = person.bbox
            color = ESCALATION_COLORS.get(ts.escalation, (200, 200, 200))
            badge = f"T:{ts.final_score} {ts.escalation}"
            cv2.putText(frame, badge,
                        (x, max(y - 22, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                        cv2.LINE_AA)

        # Door zone outline
        dx, dy, dw, dh = DOOR_ZONE
        cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh),
                      (255, 150, 0), 1)
        cv2.putText(frame, "DOOR", (dx + 2, dy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 150, 0), 1)

        # HUD
        fps       = 1.0 / max(time.time() - t_start, 0.001)
        top_score = threat_scores[0].final_score if threat_scores else 0
        top_esc   = threat_scores[0].escalation  if threat_scores else "NONE"
        hud_color = ESCALATION_COLORS.get(top_esc, (200, 200, 200))

        hud_lines = [
            f"NxV  {'[AWAY]' if _user_away else '[HOME]'}",
            f"Motion : {'YES' if motion_detected else 'no'}",
            f"Persons: {len(persons)}",
            f"Weapons: {len(weapon_detections)}",
            f"Faces  : {len(face_results)}",
            f"Threat : {top_score}  {top_esc}",
            f"FPS    : {fps:.1f}",
        ]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (142, 10 + len(hud_lines) * 15), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for i, line in enumerate(hud_lines):
            c = hud_color if i >= 5 else (220, 220, 220)
            cv2.putText(frame, line, (4, 13 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1, cv2.LINE_AA)

        # 11 ── Encode and stream ──────────────────────────────────────────────
        _, buffer = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    status = "AWAY" if _user_away else "HOME"
    return f'''<html><body style="background:#111;color:#eee;
    font-family:monospace;padding:20px">
    <h2>NxV — {status}</h2>
    <img src="/video_feed" width="640"
         style="border:2px solid #333;display:block;margin-bottom:16px"/>
    <a href="/set_away/1"
       style="color:#f90;margin-right:20px;font-size:16px">Set AWAY</a>
    <a href="/set_away/0"
       style="color:#0f9;font-size:16px">Set HOME</a>
    </body></html>'''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_away/<int:flag>')
def set_away(flag):
    global _user_away
    _user_away = bool(flag)
    print(f"[NxV] User set to: {'AWAY' if _user_away else 'HOME'}")
    return f"{'AWAY' if _user_away else 'HOME'}", 200

@app.route('/acknowledge/<int:person_id>')
def acknowledge(person_id):
    escalation_engine.acknowledge(person_id)
    return f"Acknowledged P{person_id}", 200

@app.route('/status')
def status():
    top = _last_threat_scores[0] if _last_threat_scores else None
    return {
        "user_away" : _user_away,
        "persons"   : person_tracker.active_count(),
        "top_threat": top.final_score if top else 0,
        "escalation": top.escalation  if top else "NONE",
        "flags"     : top.all_flags   if top else [],
    }