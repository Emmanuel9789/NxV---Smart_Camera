"""

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

import cv2
from flask import Flask, Response
from flask import send_file, request, jsonify
import json, os, glob, time, uuid
from datetime import datetime


_fps_current         = 0.0
_frame_times         = []
_motion_history      = []
_last_alert_contacts = []
_system_settings     = {
    "gps"      : True,
    "deterrent": True,
    "audio"    : False,
    "social"   : True,
    "history"  : True,
}

from ai.social_search import SocialSearchEngine, save_search_result
_social_engine = SocialSearchEngine()
from api.gps import GPSTracker
_gps_tracker = GPSTracker()
from detection.person     import detect_weapons
from detection.face       import FaceDetector
from tracking.tracker     import PersonTracker
from ai.behavior          import BehaviorAnalyzer
from ai.violence          import ViolenceDetector, ViolenceResult
from scoring.threat_score import ThreatScoreEngine
from alerts.escalation    import EscalationEngine

app = Flask(__name__)


import threading

_weapon_detections_async = []
_weapon_lock = threading.Lock()

def _run_weapon_detection(frame):
    global _weapon_detections_async
    result = detect_weapons(frame, conf_threshold=0.25)
    with _weapon_lock:
        _weapon_detections_async = result

_weapon_thread = None

# ── Module instances ──────────────────────────────────────────────────────────
face_detector     = FaceDetector()
person_tracker    = PersonTracker()
behavior_analyzer = BehaviorAnalyzer()
violence_detector = ViolenceDetector()
threat_engine     = ThreatScoreEngine()
escalation_engine = EscalationEngine()

# ── Config ────────────────────────────────────────────────────────────────────
# Door zone (x, y, w, h) in pixels — adjust to where your door appears
DOOR_ZONE = None

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
        _frame_times.append(time.time())
        
        if len(_frame_times) > 60:
             _frame_times.pop(0)
             
        _frame_count += 1


        # 1 ── Grab frame ──────────────────────────────────────────────────────
        frame = app.camera.get_frame()

        # 2 ── Motion detection ────────────────────────────────────────────────
        motion_detected, motion_boxes = app.motion_detector.detect(frame)

        # 3 ── Weapon detection ────────────────────────────────────────────────
        global _weapon_thread, _weapon_detections_async
        if motion_detected:
            if _weapon_thread is None or not _weapon_thread.is_alive():
                f = frame.copy()
                _weapon_thread = threading.Thread(
                    target=_run_weapon_detection, args=(f,), daemon=True)
                _weapon_thread.start()
        with _weapon_lock:
            weapon_detections = list(_weapon_detections_async)

        # 4 ── Face detection (every N frames) ─────────────────────────────────
        if motion_detected and _frame_count % 5 == 0:
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
        
        
        # Log motion history
        if persons and _system_settings.get('history', True):
            top_ts = _last_threat_scores[0] if _last_threat_scores else None
            _motion_history.insert(0, {
                'title': _flags_to_desc(top_ts.all_flags) if top_ts and top_ts.all_flags else 'Motion detected',
                'sub'  : f'{len(persons)} person(s) · Score {top_ts.final_score if top_ts else 0}',
                'time' : datetime.now().strftime('%I:%M %p'),
                'threat': top_ts.escalation != 'NONE' if top_ts else False,
                'score' : top_ts.final_score if top_ts else 0,
                'ts'   : time.time(),
            })
            if len(_motion_history) > 500:
                _motion_history.pop()

        # 9 ── Escalation (rate limited) ───────────────────────────────────────
        now = time.time()
        if (threat_scores and
                now - _last_escalation >= ESCALATION_COOLDOWN):
            top = threat_scores[0]
            if top.escalation != "NONE":
                escalation_engine.handle(top, user_away=_user_away)
                _last_escalation = now

        # ── Social media search (unknown faces at medium+ threat) ─────────────────
        for ts in threat_scores:
            if ts.trigger_social_search:
                person = person_tracker.get_person(ts.person_id)
                if person and face_results:
                    for fr in face_results:
                        x, y, w, h = fr['bbox']
                        result = _social_engine.search_from_frame(
                            frame, (x, y, w, h), ts.person_id
                        )
                        if result:
                            save_search_result(result)
                            # Boost threat score if danger keywords found
                            if result['suggested_threat_boost'] > 0:
                                print(f"[NxV] Social search boosted P{ts.person_id} "
                                    f"by +{result['suggested_threat_boost']}")
        
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

        #Door zone outline
        if DOOR_ZONE is not None:
            dx, dy, dw, dh = DOOR_ZONE
            cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh),
                  (255, 150, 0), 1)
            cv2.putText(frame, "DOOR", (dx + 2, dy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 150, 0), 1)

        #HUD
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
                                 [cv2.IMWRITE_JPEG_QUALITY, 60])
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

@app.route('/gps')
def gps_update():
    """
    Phone calls this URL with its GPS coordinates.

    URL format:
      http://<PI_IP>:5000/gps?lat=39.2034&lon=-76.8621

    iPhone Shortcut:
      Action: Get Contents of URL
      URL: http://<PI_IP>:5000/gps?lat=[Latitude]&lon=[Longitude]
      Run every 30 minutes or on location change

    Android (MacroDroid or Tasker):
      Trigger: Location change
      Action: HTTP GET http://<PI_IP>:5000/gps?lat=%loc_lat&lon=%loc_long
    """
    from flask import request
    global _user_away

    try:
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
    except (TypeError, ValueError):
        return {"error": "Invalid lat/lon"}, 400

    result = _gps_tracker.update(lat, lon)

    # Auto-update user_away based on GPS distance
    _user_away = result['user_away']

    return result, 200


@app.route('/gps/status')
def gps_status():
    """Returns current GPS state."""
    stale = _gps_tracker.is_stale()
    return {
        "last_lat"    : _gps_tracker.last_lat,
        "last_lon"    : _gps_tracker.last_lon,
        "distance_km" : _gps_tracker.distance_km,
        "user_away"   : _gps_tracker.user_away,
        "zone"        : _gps_tracker.distance_zone,
        "stale"       : stale,
        "updated_at"  : (_gps_tracker.last_update.strftime("%H:%M:%S")
                         if _gps_tracker.last_update else "never"),
    }, 200



@app.route('/app')
def pwa_app():
    return send_file('/home/emmanuel/camera_project/camera/app.html')

@app.route('/manifest.json')
def manifest():
    return send_file('/home/emmanuel/camera_project/camera/manifest.json',
                     mimetype='application/json')

@app.route('/status')
def status():
    top = _last_threat_scores[0] if _last_threat_scores else None
    now = time.time()
    recent = [t for t in _frame_times if now - t < 2.0]
    fps = round(len(recent) / 2.0, 1) if len(recent) > 1 else 0.0
    return jsonify({
        "user_away"    : _user_away,
        "persons"      : person_tracker.active_count(),
        "top_threat"   : top.final_score if top else 0,
        "escalation"   : top.escalation  if top else "NONE",
        "flags"        : top.all_flags   if top else [],
        "fps"          : fps,
        "settings"     : _system_settings,
        "motion_count" : len(_motion_history),
    })

@app.route('/snapshot')
def snapshot():
    try:
        frame    = app.camera.get_frame()
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = '/home/emmanuel/camera_project/evidence/snapshots'
        os.makedirs(snap_dir, exist_ok=True)
        path     = f'{snap_dir}/snapshot_{ts}.jpg'
        import cv2
        cv2.imwrite(path, frame)
        return send_file(path, mimetype='image/jpeg',
                         as_attachment=True,
                         download_name=f'nxv_snapshot_{ts}.jpg')
    except Exception as e:
        return str(e), 500

@app.route('/evidence_list')
def evidence_list():
    evidence_dir = '/home/emmanuel/camera_project/evidence'
    incidents    = []
    if not os.path.exists(evidence_dir):
        return jsonify([])
    folders = sorted(glob.glob(f'{evidence_dir}/*_incident'), reverse=True)
    for folder in folders[:30]:
        log_path = os.path.join(folder, 'threat_log.json')
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path) as f:
                log = json.load(f)
            incidents.append({
                'id'         : os.path.basename(folder),
                'time'       : log.get('incident_time','')[:16].replace('T',' '),
                'duration'   : log.get('duration_secs', 0),
                'escalation' : log.get('threat',{}).get('escalation','NONE'),
                'score'      : log.get('threat',{}).get('final_score', 0),
                'description': _flags_to_desc(log.get('threat',{}).get('flags',[])),
                'flags'      : log.get('threat',{}).get('flags',[]),
            })
        except Exception:
            continue
    return jsonify(incidents)

@app.route('/motion_history')
def motion_history_route():
    return jsonify(_motion_history[-200:])

@app.route('/evidence_report/<incident_id>')
def evidence_report(incident_id):
    safe_id = os.path.basename(incident_id)
    summary = f'/home/emmanuel/camera_project/evidence/{safe_id}/summary.txt'
    if os.path.exists(summary):
        with open(summary) as f:
            content = f.read()
        return f'<pre style="font-family:monospace;padding:20px;white-space:pre-wrap;background:#0a0a0a;color:#f0f0f0;min-height:100vh;margin:0">{content}</pre>'
    return 'Report not found', 404

@app.route('/settings', methods=['GET', 'POST'])
def settings_route():
    global _system_settings
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        _system_settings.update({
            k: v for k, v in data.items()
            if k in _system_settings
        })
        return jsonify(_system_settings)
    return jsonify(_system_settings)

@app.route('/contacts', methods=['GET', 'POST'])
def contacts_route():
    env_path = '/home/emmanuel/camera_project/.env'
    if request.method == 'GET':
        return jsonify({
            'owner'    : os.environ.get('NXV_OWNER_PHONE',    ''),
            'contact_1': os.environ.get('NXV_CONTACT_1_PHONE',''),
            'contact_2': os.environ.get('NXV_CONTACT_2_PHONE',''),
            'contact_3': os.environ.get('NXV_CONTACT_3_PHONE',''),
        })
    data = request.get_json(silent=True) or {}
    mapping = {
        'contact_1': 'NXV_CONTACT_1_PHONE',
        'contact_2': 'NXV_CONTACT_2_PHONE',
        'contact_3': 'NXV_CONTACT_3_PHONE',
    }
    for key, env_var in mapping.items():
        if key in data and data[key]:
            os.environ[env_var] = data[key]
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
            updated = {v: False for v in mapping.values()}
            new_lines = []
            for line in lines:
                replaced = False
                for key, env_var in mapping.items():
                    if line.startswith(f'export {env_var}='):
                        if key in data and data[key]:
                            new_lines.append(f'export {env_var}="{data[key]}"\n')
                            updated[env_var] = True
                            replaced = True
                            break
                if not replaced:
                    new_lines.append(line)
            for key, env_var in mapping.items():
                if not updated[env_var] and key in data and data[key]:
                    new_lines.append(f'export {env_var}="{data[key]}"\n')
            with open(env_path, 'w') as f:
                f.writelines(new_lines)
    except Exception as e:
        print(f"[NxV] Could not persist contacts: {e}")
    return jsonify({'status': 'saved'})

@app.route('/contact_status')
def contact_status():
    return jsonify(_last_alert_contacts)

@app.route('/flagged', methods=['GET', 'POST', 'DELETE'])
def flagged_persons():
    db_path = '/home/emmanuel/camera_project/datasets/flagged_persons.json'
    if request.method == 'GET':
        if os.path.exists(db_path):
            with open(db_path) as f:
                return jsonify(json.load(f).get('persons', []))
        return jsonify([])
    if request.method == 'POST':
        data  = request.get_json(silent=True) or {}
        name  = data.get('name','')
        score = int(data.get('threat_score', 50))
        reason= data.get('reason','')
        if not name:
            return 'Name required', 400
        db = {'version':'1.0','persons':[]}
        if os.path.exists(db_path):
            with open(db_path) as f:
                db = json.load(f)
        record = {
            'id'          : str(uuid.uuid4())[:8],
            'name'        : name,
            'threat_score': score,
            'reason'      : reason,
            'added'       : datetime.now().isoformat(),
            'sightings'   : 0,
            'last_seen'   : None,
        }
        db['persons'].append(record)
        with open(db_path,'w') as f:
            json.dump(db, f, indent=2)
        if hasattr(face_detector,'reload_db'):
            face_detector.reload_db()
        return jsonify(record), 201
    if request.method == 'DELETE':
        pid = request.args.get('id')
        if not pid or not os.path.exists(db_path):
            return 'Not found', 404
        with open(db_path) as f:
            db = json.load(f)
        db['persons'] = [p for p in db['persons'] if p['id'] != pid]
        with open(db_path,'w') as f:
            json.dump(db, f, indent=2)
        if hasattr(face_detector,'reload_db'):
            face_detector.reload_db()
        return jsonify({'status':'deleted'})

def _flags_to_desc(flags):
    for f in flags:
        if 'AIMING'            in f: return 'Weapon aimed at camera'
        if 'BREAK_IN'          in f: return 'Break-in attempt'
        if 'weapon:gun'        in f: return 'Gun detected'
        if 'weapon:knife'      in f: return 'Knife detected'
        if 'weapon:'           in f: return 'Weapon detected'
        if 'face:known'        in f: return 'Known dangerous person'
        if 'face:masked'       in f: return 'Masked person'
        if 'behavior:loitering'in f: return 'Loitering detected'
    return 'Suspicious activity'
