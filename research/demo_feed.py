"""
DSA Honor Project — Live Detection Demo Feed
dsa_simulation/demo_feed.py

Plays a video file through the full detection pipeline
and streams it to a browser with DSA-focused overlay.

"""




"""
    Hash Maps

scoring/threat_score.py
 

    Dictionnaries

ai/behavior.py         
ai/predictive.py       
utils/db.py            
alerts/notification_tiers.py
camera/stream.py       

    Queue
    
camera/stream.py       
alerts/call_chain.py   
utils/clip_recorder.py 


    Sorting
    
scoring/threat_score.py   
utils/db.py             
ai/predictive.py          

    Graphs
    
tracking/tracker.py     
ai/neighborhood.py      

Trees

 utils/db.py              
 ai/safe_zone.py          
 scoring/threat_score.py  
"""


import cv2
import time
import sys
import os
import threading
from flask import Flask, Response


sys.path.insert(0, '/home/emmanuel/camera_project')

from detection.person     import detect_weapons
from detection.motion     import MotionDetector
from tracking.tracker     import PersonTracker
from ai.behavior          import BehaviorAnalyzer
from scoring.threat_score import ThreatScoreEngine
from ai.violence          import ViolenceResult

app = Flask(__name__)

# ── Config 
VIDEO_PATH = "/home/emmanuel/camera_project/research/test_video.mp4"
LOOP_VIDEO = True   
DSA_PORT   = 5001   

# ── Module instances 
motion_detector  = MotionDetector(min_area=1500)
person_tracker   = PersonTracker()
behavior_analyzer= BehaviorAnalyzer()
threat_engine    = ThreatScoreEngine()

# ── Async weapon detection 
_weapon_detections = []
_weapon_lock       = threading.Lock()
_weapon_thread     = None

def _run_weapon_detection(frame):
    global _weapon_detections
    result = detect_weapons(frame, conf_threshold=0.25)
    with _weapon_lock:
        _weapon_detections = result

# ── State ─────────────────────────────────────────────────────────────────────
_frame_count      = 0
_last_scores      = []
_last_behaviors   = {}

ESCALATION_COLORS_BGR = {
    "EMERGENCY": (0, 34, 255),
    "ALERT"    : (0, 100, 255),
    "NOTIFY"   : (0, 200, 255),
    "NONE"     : (0, 200, 80),
}

def generate_frames():
    global _frame_count, _weapon_thread, _last_scores, _last_behaviors

    if not os.path.exists(VIDEO_PATH):
        print(f"[DSA Demo] Video not found: {VIDEO_PATH}")
        print("  Run transfer_server.py to upload your video first.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[DSA Demo] Cannot open video: {VIDEO_PATH}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    delay     = 1.0 / min(fps_video, 15)

    print(f"[DSA Demo] Playing: {VIDEO_PATH}")
    print(f"[DSA Demo] FPS: {fps_video}")

    while True:
        t_start       = time.time()
        _frame_count += 1

        ret, frame = cap.read()

        if not ret:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                motion_detector.reference_frame = None
                person_tracker.reset()
                _last_scores    = []
                _last_behaviors = {}
                continue
            else:
                break

        # Resize to consistent size
        frame = cv2.resize(frame, (480, 360))

        # ── Motion detection ──────────────────────────────────────────────────
        motion_detected, motion_boxes = motion_detector.detect(frame)

        # ── Weapon detection (threaded) ───────────────────────────────────────
        if motion_detected and _frame_count % 3 == 0:
            if _weapon_thread is None or not _weapon_thread.is_alive():
                _weapon_thread = threading.Thread(
                    target=_run_weapon_detection,
                    args=(frame.copy(),), daemon=True)
                _weapon_thread.start()

        with _weapon_lock:
            weapon_detections = list(_weapon_detections)

        # ── Person tracking ───────────────────────────────────────────────────
        tracking_boxes = list(motion_boxes)
        for det in weapon_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            tracking_boxes.append((x1, y1, x2-x1, y2-y1))

        persons = person_tracker.update(tracking_boxes, timestamp=time.time())

        # ── Behavior analysis ─────────────────────────────────────────────────
        if persons and _frame_count % 5 == 0:
            _last_behaviors = behavior_analyzer.analyze(persons)

        # ── Threat scoring ────────────────────────────────────────────────────
        if persons:
            _last_scores = threat_engine.score_from_results(
                persons           = persons,
                behavior_results  = _last_behaviors,
                violence_result   = ViolenceResult(0, []),
                weapon_detections = weapon_detections,
                face_results      = [],
                door_zone         = None,
                user_away         = False,
            )

        # ── Draw overlays ─────────────────────────────────────────────────────

        # Motion boxes (thin green)
        for (x, y, bw, bh) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 200, 80), 1)

        # Weapon boxes (red, thick)
        for det in weapon_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame,
                        f"WEAPON {float(det['conf']):.0%}",
                        (x1, max(y1-8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Person tracker trails
        frame = person_tracker.draw(frame, persons)

        # Threat score badges
        top_score = 0
        top_esc   = "NONE"
        for ts in _last_scores:
            person = person_tracker.get_person(ts.person_id)
            if person is None:
                continue
            x, y, bw, bh = person.bbox
            color = ESCALATION_COLORS_BGR.get(ts.escalation, (200,200,200))
            cv2.putText(frame,
                        f"THREAT: {ts.final_score}  [{ts.escalation}]",
                        (x, max(y-22, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if ts.final_score > top_score:
                top_score = ts.final_score
                top_esc   = ts.escalation

        # ── DSA HUD ───────────────────────────────────────────────────────────
        hud_color = ESCALATION_COLORS_BGR.get(top_esc, (200,200,200))
        fps_actual = 1.0 / max(time.time() - t_start, 0.001)

        # Dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (310, 200), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Algorithm info
        hud = [
            ("DSA THREAT SCORING SIM", (0,255,136)),
            ("", None),
            (f"Algorithm : Weighted multi-factor", (150,150,150)),
            (f"Complexity : O(n log n)", (150,150,150)),
            (f"Persons   : {len(persons)}", (200,200,200)),
            (f"Weapons   : {len(weapon_detections)}", (200,200,200)),
            ("", None),
            (f"SCORE     : {top_score}/100", hud_color),
            (f"LEVEL     : {top_esc}", hud_color),
            (f"FPS       : {fps_actual:.1f}", (100,100,100)),
        ]

        for i, (text, color) in enumerate(hud):
            if not text or color is None:
                continue
            cv2.putText(frame, text, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        color, 1, cv2.LINE_AA)

        # Escalation bar at bottom
        bar_w = int((top_score / 100) * 640)
        cv2.rectangle(frame, (0, 472), (640, 480), (20,20,20), -1)
        cv2.rectangle(frame, (0, 472), (bar_w, 480), hud_color, -1)
        cv2.putText(frame, "THREAT LEVEL",
                    (4, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (100,100,100), 1)

        # Algorithm equation (bottom right)
        eq_lines = [
            "Score = (behavior x 0.20)",
            "      + (weapon   x 0.30)",
            "      + (face     x 0.15)",
            "      + (time     x 0.10)",
        ]
        for i, line in enumerate(eq_lines):
            cv2.putText(frame, line,
                        (340, 430 + i * 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (80, 80, 80), 1, cv2.LINE_AA)

        # ── Encode and stream ─────────────────────────────────────────────────
        _, buffer = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 50])

        # Sync to video FPS
        elapsed = time.time() - t_start
        if elapsed < delay:
            time.sleep(delay - elapsed)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return '''
    <html>
    <head>
      <style>
        body {
          background:#080c10; color:#c9d1d9;
          font-family:monospace; padding:20px;
          display:flex; flex-direction:column; align-items:center;
        }
        h2 { color:#00ff88; letter-spacing:4px; margin-bottom:6px; }
        p  { color:#4a5568; font-size:12px; margin-bottom:20px; }
        img { border: 1px solid #1a2332; display:block; }
        .badge {
          display:inline-block; margin-top:14px;
          padding:4px 14px; border:1px solid #00ff88;
          color:#00ff88; font-size:11px; letter-spacing:2px;
        }
      </style>
    </head>
    <body>
      <h2>DSA — THREAT SCORING SIMULATION</h2>
      <p>Interpretable Multi-Factor Algorithm | O(n log n) complexity</p>
      <img src="/video_feed" width="640"/>
      <div class="badge">LIVE DETECTION FEED</div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "10.71.53.113"

    print(f"""
[DSA Demo] Starting...
─────────────────────────────────
  Open in browser:
  http://{ip}:{DSA_PORT}
─────────────────────────────────
  Video: {VIDEO_PATH}
  Press Ctrl+C to stop
""")
    app.run(host='0.0.0.0', port=DSA_PORT, debug=False,
            use_reloader=False, threaded=True)
