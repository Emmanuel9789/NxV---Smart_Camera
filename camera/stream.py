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
import threading
import cv2
from flask import Flask, Response
from flask import send_file, request, jsonify
import json, os, glob, time, uuid
from datetime import datetime
from utils.clip_recorder import ClipRecorder
from ai.delivery import DeliveryDetector
from ai.neighborhood import NeighborhoodNetwork
from alerts.notification_tiers import NotificationManager, NotificationEvent
from ai.safe_zone import SafeZoneManager
from ai.predictive import PredictiveModel
predictive_model = PredictiveModel()
from utils.db import (
    init_db, get_all_persons, add_person, delete_person,
    get_clips, get_clip, save_clip, get_contacts, add_contact,
    remove_contact, save_contacts, get_alerts, save_alert,
    log_motion, get_motion_log, get_all_settings, set_settings,
    get_setting, set_setting, update_person_sighting,
    delete_expired_clips
)
init_db()


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


_latest_frame = None
_pipeline_lock = threading.Lock()



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
clip_recorder = ClipRecorder()
safe_zone = SafeZoneManager()
delivery_detector    = DeliveryDetector()
neighborhood_network = NeighborhoodNetwork()
notification_manager = NotificationManager()
neighborhood_network.start()

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
        
        if _capture_session['active']:
         if len(_capture_session['frames']) < _capture_session['target']:
             _capture_session['frames'].append(frame.copy())
             print(f"[NxV SafeZone] Frame {len(_capture_session['frames'])}"
                   f"/{_capture_session['target']}")
         else:
             _capture_session['active'] = False
             print("[NxV SafeZone] Capture complete — call /register_face/complete")


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
            
            log_motion(
                description   = _flags_to_desc(top_ts.all_flags) if top_ts and top_ts.all_flags else 'Motion detected',
                persons_count = len(persons),
                score         = top_ts.final_score if top_ts else 0,
                escalation    = top_ts.escalation if top_ts else 'NONE',
                is_threat     = top_ts.escalation != 'NONE' if top_ts else False,
)
            
            if len(_motion_history) > 500:
                _motion_history.pop()
                
                
        is_flagged = any('face:known' in f for f in
                 (threat_scores[0].all_flags if threat_scores else []))
        current_esc = threat_scores[0].escalation if threat_scores else 'NONE'
    
        
        clip_recorder.on_frame(
            frame, motion=motion_detected,
            escalation=current_esc, flagged=is_flagged,)
        

        if threat_scores and threat_scores[0].escalation == 'EMERGENCY':
            clip_recorder.force_save('EMERGENCY', is_flagged)
            
            
        # ── Predictive model — log event ──────────────────────────────────────
        if persons and motion_detected:
            top_ts = _last_threat_scores[0] if _last_threat_scores else None
            predictive_model.log_event(
                hour       = datetime.now().hour,
                duration   = persons[0].dwell_time if persons else 0,
                had_person = True,
                score      = top_ts.final_score if top_ts else 0,
                escalation = top_ts.escalation  if top_ts else 'NONE',
            )

            # Get predictive boost and apply to threat score
            is_new_face = not any(
                fr.get('match') for fr in face_results
            ) if face_results else False

            p_boost, p_reason = predictive_model.get_score_boost(
                hour          = datetime.now().hour,
                current_dwell = persons[0].dwell_time if persons else 0,
                is_new_face   = is_new_face,
            )

            if p_boost > 0 and _last_threat_scores:
                _last_threat_scores[0].final_score = min(
                    100,
                    _last_threat_scores[0].final_score + p_boost
                )
                if p_reason:
                    _last_threat_scores[0].all_flags.append(
                        f"predict:{p_reason}"
                    )
                print(f"[NxV Predict] Boost +{p_boost} → {p_reason}")    
        

        # 9 ── Escalation (rate limited) ───────────────────────────────────────
        now = time.time()
        if (threat_scores and
                now - _last_escalation >= ESCALATION_COOLDOWN):
            top = threat_scores[0]
            if top.escalation != "NONE":
                escalation_engine.handle(top, user_away=_user_away)
                _last_escalation = now
                
                save_alert(
                    escalation = top.escalation,
                    score      = top.final_score,
                    flags      = top.all_flags,
                )
        
        # ── Delivery detection ────────────────────────────────────────
        delivery_result = delivery_detector.analyze(frame, persons, motion_boxes)
        if delivery_result.is_delivery and motion_detected:
            print(f"[NxV Delivery] {delivery_result.notification_label}")

        # ── Neighborhood network face check ──────────────────────────
        if face_results and neighborhood_network.is_enabled:
            for fr in face_results:
                enc = fr.get('encoding')
                if enc is not None:
                    b_score = 0
                    if persons and behavior_results:
                        b_result = behavior_results.get(persons[0].id)
                        b_score  = b_result.score if b_result else 0

                    net_result = neighborhood_network.check_face(
                        enc,
                        behavior_score = b_score,
                        has_weapon     = len(weapon_detections) > 0,
                        is_at_door     = any('near_door' in f
                                            for f in (threat_scores[0].all_flags
                                            if threat_scores else [])),
                    )
                    if net_result:
                        print(f"[NxV Network] {net_result['layer']}: "
                            f"{net_result['name']} "
                            f"(boost: {net_result['score_boost']})")
                        # Apply score boost
                        if threat_scores:
                            threat_scores[0].final_score = min(100,
                                threat_scores[0].final_score + net_result['score_boost'])
                            threat_scores[0].all_flags.append(
                                f"network:{net_result['name']}"
                            )
                            if net_result['should_escalate']:
                                threat_scores[0].escalation = net_result['min_escalation']

        # ── Share to network on EMERGENCY ────────────────────────────
        if threat_scores and neighborhood_network.is_enabled:
            top = threat_scores[0]
            if top.escalation == 'EMERGENCY':
                for fr in face_results:
                    enc = fr.get('encoding')
                    if enc is not None and not fr.get('match'):
                        from ai.neighborhood import NetworkThreatRecord
                        record = NetworkThreatRecord(
                            person_id      = f"p{top.person_id}_{int(time.time())}",
                            name           = "Unknown threat",
                            crime_severity = "SEVERE",
                            reason         = ", ".join(top.all_flags[:3]),
                            threat_score   = top.final_score,
                            embedding      = enc,
                            camera_id      = os.environ.get("NXV_CAMERA_ID","unknown"),
                        )
                        neighborhood_network.share_threat(record)
        
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

        
        
        # Fix colors — convert BGR to RGB for correct color display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 11 ── Encode and stream ──────────────────────────────────────────────
        _, buffer = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

def run_pipeline():
    """Background thread — runs detection pipeline constantly."""
    global _latest_frame
    for frame in generate_frames():
        with _pipeline_lock:
            _latest_frame = frame
            
def start_pipeline():
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    print("[NxV] Detection pipeline running in background")


# ── Flask routes ──────────────────────────────────────────────────────────────

"""
NxV - DB-backed routes for stream.py
Replace existing routes in stream.py with these versions.

Add at top of stream.py:
    from utils.db import (
        init_db, get_all_persons, add_person, delete_person,
        get_clips, get_clip, save_clip, get_contacts, add_contact,
        remove_contact, save_contacts, get_alerts, save_alert,
        log_motion, get_motion_log, get_all_settings, set_settings,
        get_setting, set_setting, update_person_sighting,
        delete_expired_clips
    )
    init_db()   # initialize DB on startup
"""




# ── Safe zone (trusted persons) ───────────────────────────────────────────────

@app.route('/trusted', methods=['GET', 'POST', 'DELETE'])
def trusted_persons():
    if request.method == 'GET':
        return jsonify(get_all_persons(trusted_only=True))

    if request.method == 'POST':
        data   = request.get_json(silent=True) or {}
        name   = data.get('name', '').strip()
        reason = data.get('reason', 'Trusted person')
        if not name:
            return 'Name required', 400
        import uuid
        pid    = str(uuid.uuid4())[:8]
        person = add_person(pid, name, 0, reason, is_trusted=True)
        return jsonify(person), 201

    if request.method == 'DELETE':
        pid = request.args.get('id')
        if not pid:
            return 'ID required', 400
        delete_person(pid)
        return jsonify({'status': 'deleted'})


# ── Contacts ──────────────────────────────────────────────────────────────────

@app.route('/contacts', methods=['GET', 'POST'])
def contacts_route():
    if request.method == 'GET':
        return jsonify(get_contacts())

    data = request.get_json(silent=True) or {}

    # Full replace
    if 'contacts' in data:
        save_contacts(data['contacts'])
        return jsonify({'status': 'saved'})

    # Add single contact
    name         = data.get('name', '').strip()
    phone        = data.get('phone', '').strip()
    country_code = data.get('country_code', '+1')
    if not name or not phone:
        return 'Name and phone required', 400
    contact = add_contact(name, phone, 'contact', country_code)
    return jsonify(contact), 201


@app.route('/contacts/<int:contact_id>', methods=['DELETE'])
def delete_contact_route(contact_id):
    existing = get_contacts()
    active   = [c for c in existing if c['role'] == 'contact']
    if len(active) <= 1:
        return 'At least 1 contact required', 400
    remove_contact(contact_id)
    return jsonify({'status': 'deleted'})


# ── Clips ─────────────────────────────────────────────────────────────────────

@app.route('/clips')
def clips_list():
    f     = request.args.get('filter', 'all')
    clips = get_clips(filter_type=f, limit=100)
    return jsonify(clips)

@app.route('/clip/<clip_id>')
def serve_clip(clip_id):
    safe_id = os.path.basename(clip_id).replace('..', '')
    path = clip_recorder.get_clip_path(safe_id)
    if not path:
        return 'Clip not found', 404
    mimetype = 'video/x-msvideo' if path.endswith('.avi') else 'video/mp4'
    return send_file(path, mimetype=mimetype)

@app.route('/clips/storage')
def clips_storage():
    clips    = get_clips(limit=10000)
    total    = sum(
        os.path.getsize(c['path'])
        for c in clips if os.path.exists(c.get('path',''))
    )
    kept     = sum(1 for c in clips if c.get('keep_forever'))
    return jsonify({
        'total_mb'       : round(total / 1024 / 1024, 1),
        'clip_count'     : len(clips),
        'kept_forever'   : kept,
        'auto_delete_days': int(get_setting('auto_delete_days', 7)),
    })


# ── Alerts ────────────────────────────────────────────────────────────────────

@app.route('/alerts_history')
def alerts_history():
    esc    = request.args.get('escalation')
    limit  = int(request.args.get('limit', 100))
    alerts = get_alerts(limit=limit, escalation=esc)
    return jsonify(alerts)


# ── Motion log ────────────────────────────────────────────────────────────────

@app.route('/motion_history')
def motion_history_route():
    threats_only = request.args.get('threats') == '1'
    rows         = get_motion_log(limit=200, threats_only=threats_only)
    return jsonify(rows)


# ── Settings ──────────────────────────────────────────────────────────────────

@app.route('/settings', methods=['GET', 'POST'])
def settings_route():
    if request.method == 'GET':
        s = get_all_settings()
        # Convert string booleans to real booleans for JS
        return jsonify({
            k: (v == 'true' if v in ('true','false') else v)
            for k, v in s.items()
        })

    data = request.get_json(silent=True) or {}
    set_settings({k: str(v) for k, v in data.items()})
    return jsonify({'status': 'saved'})


# ── Status (enhanced with DB) ─────────────────────────────────────────────────

@app.route('/status')
def status():
    top  = _last_threat_scores[0] if _last_threat_scores else None
    now  = time.time()
    recent = [t for t in _frame_times if now - t < 2.0]
    fps  = round(len(recent) / 2.0, 1) if len(recent) > 1 else 0.0
    s    = get_all_settings()

    return jsonify({
        "user_away"    : _user_away,
        "persons"      : person_tracker.active_count(),
        "top_threat"   : top.final_score if top else 0,
        "escalation"   : top.escalation  if top else "NONE",
        "flags"        : top.all_flags   if top else [],
        "fps"          : fps,
        "settings"     : {
            k: (v == 'true' if v in ('true','false') else v)
            for k, v in s.items()
        },
        "clip_count"   : len(get_clips(limit=10000)),
        "alert_count"  : len(get_alerts(limit=10000)),
    })


# ── Evidence report ───────────────────────────────────────────────────────────

@app.route('/evidence_report/<incident_id>')
def evidence_report(incident_id):
    safe_id = os.path.basename(incident_id)
    summary = f'/home/emmanuel/camera_project/evidence/{safe_id}/summary.txt'
    if os.path.exists(summary):
        with open(summary) as f:
            content = f.read()
        return f'<pre style="font-family:monospace;padding:20px;white-space:pre-wrap;background:#0a0a0a;color:#f0f0f0;min-height:100vh;margin:0">{content}</pre>'
    return 'Report not found', 404


# ── Snapshot ──────────────────────────────────────────────────────────────────

def snapshot_feed():
    global _latest_frame
    with _pipeline_lock:
        frame = _latest_frame
    if frame is None:
        return 'No frame yet', 503
    # frame is already JPEG bytes from generate_frames
    return Response(
        frame.split(b'\r\n\r\n')[1].split(b'\r\n')[0],
        mimetype='image/jpeg',
        headers={'Cache-Control': 'no-cache, no-store'}
    )


# ── PWA ───────────────────────────────────────────────────────────────────────

@app.route('/app')
def pwa_app():
    return send_file('/home/emmanuel/camera_project/camera/app.html')

@app.route('/manifest.json')
def manifest():
    return send_file('/home/emmanuel/camera_project/camera/manifest.json',
                     mimetype='application/json')


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

"""
NxV - Safe Zone Routes
Add to camera/stream.py

STEP 1 — Add import at top of stream.py:
    from ai.safe_zone import SafeZoneManager
    safe_zone = SafeZoneManager()

STEP 2 — Paste these routes at bottom of stream.py
"""

import base64
import uuid
import numpy as np


# ── Registration state (captures frames while user looks at camera) ────────────
_capture_session = {
    'active'    : False,
    'person_id' : None,
    'name'      : None,
    'frames'    : [],
    'started_at': None,
    'target'    : 15,    # number of frames to capture
}


@app.route('/register_face/start', methods=['POST'])
def register_face_start():
    """
    Step 1 — App tells Pi to start capturing frames.
    The live stream continues normally while frames are collected.
    """
    global _capture_session
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()

    if not name:
        return jsonify({'error': 'Name required'}), 400

    if _capture_session['active']:
        return jsonify({'error': 'Registration already in progress'}), 409

    person_id = str(uuid.uuid4())[:8]

    _capture_session = {
        'active'    : True,
        'person_id' : person_id,
        'name'      : name,
        'frames'    : [],
        'started_at': time.time(),
        'target'    : 15,
    }

    print(f"[NxV SafeZone] Starting capture for: {name} (ID: {person_id})")
    return jsonify({
        'status'   : 'capturing',
        'person_id': person_id,
        'message'  : f'Look at the camera. Capturing {_capture_session["target"]} frames...',
        'target'   : _capture_session['target'],
    })


@app.route('/register_face/status')
def register_face_status():
    """App polls this to see how many frames captured."""
    session = _capture_session
    if not session['active'] and not session['frames']:
        return jsonify({'status': 'idle', 'frames': 0, 'target': 15})
    return jsonify({
        'status'   : 'capturing' if session['active'] else 'ready',
        'frames'   : len(session['frames']),
        'target'   : session['target'],
        'name'     : session['name'],
        'person_id': session['person_id'],
        'progress' : min(100, int(len(session['frames']) / session['target'] * 100)),
    })


@app.route('/register_face/complete', methods=['POST'])
def register_face_complete():
    """
    Step 2 — App tells Pi to process the captured frames.
    Pi computes embedding and saves to DB.
    """
    global _capture_session

    if not _capture_session['frames']:
        return jsonify({'error': 'No frames captured yet'}), 400

    frames    = list(_capture_session['frames'])
    name      = _capture_session['name']
    person_id = _capture_session['person_id']

    # Reset session
    _capture_session = {
        'active': False, 'person_id': None,
        'name': None, 'frames': [], 'started_at': None, 'target': 15,
    }

    # Process frames
    result = safe_zone.register_from_frames(frames, person_id, name)

    if result['success']:
        # Save to DB as trusted person
        add_person(
            id           = person_id,
            name         = name,
            threat_score = 0,
            reason       = 'Trusted — registered via NxV app',
            is_trusted   = True,
        )
        # Reload face detector
        if hasattr(face_detector, 'reload_db'):
            face_detector.reload_db()
        safe_zone.reload()
        print(f"[NxV SafeZone] ✓ {name} registered and active")

    return jsonify(result)


@app.route('/register_face/cancel', methods=['POST'])
def register_face_cancel():
    """Cancel an in-progress registration."""
    global _capture_session
    _capture_session = {
        'active': False, 'person_id': None,
        'name': None, 'frames': [], 'started_at': None, 'target': 15,
    }
    return jsonify({'status': 'cancelled'})


@app.route('/trusted', methods=['GET'])
def trusted_list():
    """Return all registered trusted faces."""
    return jsonify(safe_zone.get_trusted_list())


@app.route('/trusted/<person_id>', methods=['DELETE'])
def remove_trusted(person_id):
    """Remove a trusted person."""
    safe_zone.remove_trusted(person_id)
    delete_person(person_id)
    return jsonify({'status': 'removed'})
@app.route('/network/status')
def network_status():
    return jsonify({
        "enabled"    : neighborhood_network.is_enabled,
        "camera_id"  : os.environ.get("NXV_CAMERA_ID","not-set"),
        "threats"    : neighborhood_network.threat_count,
        "trusted"    : neighborhood_network.trusted_count,
        "network_db" : neighborhood_network.get_all_threats(),
    })

@app.route('/notifications/settings', methods=['GET','POST'])
def notification_settings():
    if request.method == 'GET':
        return jsonify(notification_manager.get_settings())
    data = request.get_json(silent=True) or {}
    notification_manager.update_settings(data)
    return jsonify(notification_manager.get_settings())

@app.route('/snapshot_feed')
def snapshot_feed():
    """Single frame for mobile app feed polling."""
    try:
        frame = app.camera.get_frame()
        import cv2
        _, buffer = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 60])
        from flask import Response
        return Response(buffer.tobytes(),
                       mimetype='image/jpeg',
                       headers={'Cache-Control': 'no-cache, no-store'})
    except Exception as e:
        return str(e), 500
    
@app.route('/video_feed')
def video_feed():
    def stream():
        while True:
            with _pipeline_lock:
                frame = _latest_frame
            if frame:
                yield frame
            time.sleep(0.05)
    return Response(stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/')
def index():
    return send_file('/home/emmanuel/camera_project/camera/app.html')


@app.route('/predict/status')
def predict_status():
    """Return predictive model baseline summary."""
    return jsonify(predictive_model.get_baseline_summary())

@app.route('/predict/reset', methods=['POST'])
def predict_reset():
    """Reset predictive baseline — start learning fresh."""
    predictive_model.reset()
    return jsonify({'status': 'reset'})

@app.route('/predict/baseline')
def predict_baseline():
    """Return full hourly baseline for visualization."""
    if not predictive_model._has_enough_data():
        return jsonify({
            'ready'  : False,
            'message': predictive_model.get_baseline_summary()['message']
        })

    baseline = []
    for hour in range(24):
        stats = predictive_model._baseline.get(hour, {})
        baseline.append({
            'hour'             : hour,
            'label'            : f"{hour:02d}:00",
            'avg_events_per_day': round(stats.get('avg_events_per_day', 0), 2),
            'avg_duration'     : round(stats.get('avg_duration', 0), 1),
            'avg_score'        : round(stats.get('avg_score', 0), 1),
            'event_count'      : stats.get('event_count', 0),
            'is_quiet'         : stats.get('avg_events_per_day', 0) < 0.2,
            'is_busy'          : stats.get('avg_events_per_day', 0) > 1.0,
        })

    return jsonify({
        'ready'   : True,
        'baseline': baseline,
        'summary' : predictive_model.get_baseline_summary(),
    })