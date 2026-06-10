import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re, time, uuid, json, glob
import threading
from datetime import datetime


import cv2
from flask import Flask, Response, send_file, request, jsonify

app = Flask(__name__)


from utils.security import (
    rate_limit, sanitize_name, sanitize_phone,
    sanitize_string, sanitize_integer,
    validate_json_payload, check_payload_size,
    add_security_headers, audit_on_startup, InputError 
)
app.after_request(add_security_headers)


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



_latest_frame = None
_pipeline_lock = threading.Lock()



_weapon_detections_async = []
_weapon_lock = threading.Lock()

def _run_weapon_detection(frame):
    global _weapon_detections_async
    result = detect_weapons(frame, conf_threshold=0.25)
    with _weapon_lock:#Queue ()
        _weapon_detections_async = result

_weapon_thread = None

# Module instances 
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

# Config
# Door zone (x, y, w, h) in pixels — adjust to where door appears
DOOR_ZONE = None

FACE_EVERY_N_FRAMES     = 3
BEHAVIOR_EVERY_N_FRAMES = 5
ESCALATION_COOLDOWN     = 10

# Runtime state 
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


        # 1 Grab frame 
        frame = app.camera.get_frame()
        
        if _capture_session['active']:
         if len(_capture_session['frames']) < _capture_session['target']:
             _capture_session['frames'].append(frame.copy())
             print(f"[NxV SafeZone] Frame {len(_capture_session['frames'])}"
                   f"/{_capture_session['target']}")
         else:
             _capture_session['active'] = False
             print("[NxV SafeZone] Capture complete — call /register_face/complete")


        # 2 Motion detection 
        motion_detected, motion_boxes = app.motion_detector.detect(frame)

        # 3 Weapon detection 
        global _weapon_thread, _weapon_detections_async
        if motion_detected:
            if _weapon_thread is None or not _weapon_thread.is_alive():
                f = frame.copy()
                _weapon_thread = threading.Thread(
                    target=_run_weapon_detection, args=(f,), daemon=True)
                _weapon_thread.start()
        with _weapon_lock:
            
            weapon_detections = list(_weapon_detections_async)

        # 4 Face detection
        if motion_detected and _frame_count % 5 == 0:
            _last_face_results = face_detector.detect(frame)
        face_results = _last_face_results

        # 5 Person tracking 
        tracking_boxes = list(motion_boxes)
        for det in weapon_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            tracking_boxes.append((x1, y1, x2 - x1, y2 - y1))

        persons = person_tracker.update(tracking_boxes, timestamp=time.time())

        # 6 + 7 Behavior + violence (every N frames) 
        behavior_results = {}
        violence_result  = ViolenceResult(0, [])

        if persons and _frame_count % BEHAVIOR_EVERY_N_FRAMES == 0:
            behavior_results = behavior_analyzer.analyze(persons)
            violence_result  = violence_detector.analyze(persons)

        # 8 Threat scoring 
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
            
            
        # Predictive model — log event 
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
        

        # 9 Escalation (rate limited) 
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
        
        #  Delivery detection 
        delivery_result = delivery_detector.analyze(frame, persons, motion_boxes)
        if delivery_result.is_delivery and motion_detected:
            print(f"[NxV Delivery] {delivery_result.notification_label}")

        #  Neighborhood network face check 
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

        # Share to network on EMERGENCY 
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
        
        # Social media search (unknown faces at medium+ threat) 
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
        
        # 10 Draw overlays 

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
        
        # 11 Encode and stream 
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


# Flask routes 




# Safe zone (trusted persons) 

@app.route('/trusted', methods=['GET', 'POST', 'DELETE'])
@rate_limit
def trusted_route():
    """GET: list all trusted faces. POST: add one. DELETE: remove by ?id="""

    # GET 
    if request.method == 'GET':
        # Merge DB trusted persons + safe_zone embeddings into one response
        return jsonify(safe_zone.get_trusted_list())

    # POST 
    if request.method == 'POST':
        data, err = validate_json_payload(required_fields=['name'])
        if err:
            return err
        try:
            name   = sanitize_name(data.get('name', ''))
            reason = sanitize_string(
                data.get('reason', 'Trusted person'),
                'reason', max_len=200, allow_empty=True
            )
        except InputError as e:
            return jsonify({'error': str(e)}), 400

        pid    = str(uuid.uuid4())[:8]
        person = add_person(pid, name, 0, reason, is_trusted=True)
        return jsonify(person), 201

    #  DELETE 
    if request.method == 'DELETE':
        pid = request.args.get('id', '').strip()
        if not pid or len(pid) > 40:
            return jsonify({'error': 'Valid ID required'}), 400
        if not re.match(r'^[a-zA-Z0-9\-_]+$', pid):
            return jsonify({'error': 'Invalid ID format'}), 400
        safe_zone.remove_trusted(pid)
        delete_person(pid)
        return jsonify({'status': 'deleted'})


@app.route('/trusted/<person_id>', methods=['DELETE'])
@rate_limit
def remove_trusted_by_path(person_id):
    """Remove a trusted person by path param (used by app)."""
    clean = person_id.strip()
    if not clean or len(clean) > 40:
        return jsonify({'error': 'Invalid ID'}), 400
    if not re.match(r'^[a-zA-Z0-9\-_]+$', clean):
        return jsonify({'error': 'Invalid ID format'}), 400
    safe_zone.remove_trusted(clean)
    delete_person(clean)
    return jsonify({'status': 'removed'})


# ══════════════════════════════════════════════════════════════════════════════
# CONTACTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/contacts', methods=['GET', 'POST'])
@rate_limit
def contacts_route():

    if request.method == 'GET':
        return jsonify(get_contacts())

    data, err = validate_json_payload()
    if err:
        return err

    # Full replace list
    if 'contacts' in data:
        contacts_list = data['contacts']
        if not isinstance(contacts_list, list):
            return jsonify({'error': 'contacts must be a list'}), 400
        if len(contacts_list) > 10:
            return jsonify({'error': 'Maximum 10 contacts allowed'}), 400
        save_contacts(contacts_list)
        return jsonify({'status': 'saved'})

    # Add single contact
    try:
        name         = sanitize_name(data.get('name', ''))
        phone        = sanitize_phone(data.get('phone', ''))
        country_code = sanitize_string(
            data.get('country_code', '+1'), 'country_code', max_len=5
        )
    except InputError as e:
        return jsonify({'error': str(e)}), 400

    contact = add_contact(name, phone, 'contact', country_code)
    return jsonify(contact), 201


@app.route('/contacts/<int:contact_id>', methods=['DELETE'])
@rate_limit
def delete_contact_route(contact_id):
    if contact_id < 0 or contact_id > 99999:
        return jsonify({'error': 'Invalid contact ID'}), 400
    existing = get_contacts()
    active   = [c for c in existing if c['role'] == 'contact']
    if len(active) <= 1:
        return jsonify({'error': 'At least 1 contact required'}), 400
    remove_contact(contact_id)
    return jsonify({'status': 'deleted'})


# ══════════════════════════════════════════════════════════════════════════════
# CLIPS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/clips')
def clips_list():
    f = request.args.get('filter', 'all')
    # Whitelist filter values
    if f not in ('all', 'threats', 'history'):
        f = 'all'
    limit = min(int(request.args.get('limit', 100)), 500)
    clips = get_clips(filter_type=f, limit=limit)
    return jsonify(clips)


@app.route('/clip/<clip_id>')
def serve_clip(clip_id):
    # Strip everything except safe characters
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '', os.path.basename(clip_id))
    if not clean_id:
        return jsonify({'error': 'Invalid clip ID'}), 400

    path = clip_recorder.get_clip_path(clean_id)
    if not path:
        return jsonify({'error': 'Clip not found'}), 404

    # Path traversal check — clip must be inside the clips directory
    clips_dir = '/home/emmanuel/camera_project/evidence/clips'
    real_path = os.path.realpath(path)
    if not real_path.startswith(os.path.realpath(clips_dir)):
        return jsonify({'error': 'Access denied'}), 403

    mimetype = 'video/x-msvideo' if path.endswith('.avi') else 'video/mp4'
    return send_file(path, mimetype=mimetype)


@app.route('/clips/storage')
def clips_storage():
    clips = get_clips(limit=10000)
    total = sum(
        os.path.getsize(c['path'])
        for c in clips if os.path.exists(c.get('path', ''))
    )
    kept = sum(1 for c in clips if c.get('keep_forever'))
    return jsonify({
        'total_mb'        : round(total / 1024 / 1024, 1),
        'clip_count'      : len(clips),
        'kept_forever'    : kept,
        'auto_delete_days': int(get_setting('auto_delete_days', 7)),
    })


# ══════════════════════════════════════════════════════════════════════════════
# ALERTS + MOTION LOG
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/alerts_history')
def alerts_history():
    esc   = request.args.get('escalation', '')
    # Whitelist escalation values
    if esc not in ('', 'EMERGENCY', 'ALERT', 'NOTIFY', 'NONE'):
        esc = None
    limit = min(int(request.args.get('limit', 100)), 500)
    alerts = get_alerts(limit=limit, escalation=esc or None)
    return jsonify(alerts)


@app.route('/motion_history')
def motion_history_route():
    threats_only = request.args.get('threats') == '1'
    rows = get_motion_log(limit=200, threats_only=threats_only)
    return jsonify(rows)


@app.route('/evidence_list')
def evidence_list():
    clips = get_clips(filter_type='threats', limit=30)
    return jsonify(clips)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_SETTINGS = {
    'gps', 'deterrent', 'audio', 'social', 'history',
    'home_lat', 'home_lon', 'away_threshold',
    'auto_delete_days', 'motion_sensitivity',
}

@app.route('/settings', methods=['GET', 'POST'])
@rate_limit
def settings_route():
    if request.method == 'GET':
        s = get_all_settings()
        return jsonify({
            k: (v == 'true' if v in ('true', 'false') else v)
            for k, v in s.items()
        })

    data, err = validate_json_payload()
    if err:
        return err

    cleaned = {}
    for key, value in data.items():
        if key not in ALLOWED_SETTINGS:
            continue  # silently drop unknown keys
        try:
            cleaned[key] = sanitize_string(str(value), key, max_len=50)
        except InputError:
            continue

    set_settings(cleaned)
    return jsonify({'status': 'saved'})


# ══════════════════════════════════════════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/status')
def status():
    top    = _last_threat_scores[0] if _last_threat_scores else None
    now    = time.time()
    recent = [t for t in _frame_times if now - t < 2.0]
    fps    = round(len(recent) / 2.0, 1) if len(recent) > 1 else 0.0
    s      = get_all_settings()

    return jsonify({
        'user_away'  : _user_away,
        'persons'    : person_tracker.active_count(),
        'top_threat' : top.final_score if top else 0,
        'escalation' : top.escalation  if top else 'NONE',
        'flags'      : top.all_flags   if top else [],
        'fps'        : fps,
        'settings'   : {
            k: (v == 'true' if v in ('true', 'false') else v)
            for k, v in s.items()
        },
        'clip_count' : len(get_clips(limit=10000)),
        'alert_count': len(get_alerts(limit=10000)),
    })


# ══════════════════════════════════════════════════════════════════════════════
# GPS / AWAY
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/set_away/<int:value>')
@rate_limit
def set_away_route(value):
    global _user_away
    if value not in (0, 1):
        return jsonify({'error': 'Value must be 0 or 1'}), 400
    _user_away = bool(value)
    return jsonify({'user_away': _user_away})


# ══════════════════════════════════════════════════════════════════════════════
# ACKNOWLEDGE
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/acknowledge/<int:person_id>')
@rate_limit
def acknowledge(person_id):
    global _last_threat_scores
    if person_id < 0 or person_id > 9999:
        return jsonify({'error': 'Invalid person ID'}), 400
    _last_threat_scores = [
        ts for ts in _last_threat_scores if ts.person_id != person_id
    ]
    return jsonify({'status': 'acknowledged'})


# ══════════════════════════════════════════════════════════════════════════════
# FACE REGISTRATION (safe zone)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/register_face/start', methods=['POST'])
@rate_limit
def register_face_start():
    global _capture_session

    data, err = validate_json_payload(required_fields=['name'])
    if err:
        return err

    try:
        name = sanitize_name(data.get('name', ''))
    except InputError as e:
        return jsonify({'error': str(e)}), 400

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
@rate_limit
def register_face_complete():
    global _capture_session

    if not _capture_session['frames']:
        return jsonify({'error': 'No frames captured yet'}), 400

    frames    = list(_capture_session['frames'])
    name      = _capture_session['name']
    person_id = _capture_session['person_id']

    _capture_session = {
        'active': False, 'person_id': None,
        'name': None, 'frames': [], 'started_at': None, 'target': 15,
    }

    result = safe_zone.register_from_frames(frames, person_id, name)

    if result['success']:
        add_person(
            id=person_id, name=name, threat_score=0,
            reason='Trusted — registered via NxV app', is_trusted=True,
        )
        if hasattr(face_detector, 'reload_db'):
            face_detector.reload_db()
        safe_zone.reload()
        print(f"[NxV SafeZone] ✓ {name} registered and active")

    return jsonify(result)


@app.route('/register_face/cancel', methods=['POST'])
def register_face_cancel():
    global _capture_session
    _capture_session = {
        'active': False, 'person_id': None,
        'name': None, 'frames': [], 'started_at': None, 'target': 15,
    }
    return jsonify({'status': 'cancelled'})


# ══════════════════════════════════════════════════════════════════════════════
# FLAGGED PERSONS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/flagged', methods=['GET', 'POST', 'DELETE'])
@rate_limit
def flagged_persons():
    if request.method == 'GET':
        return jsonify(get_all_persons(flagged_only=True))

    if request.method == 'POST':
        data, err = validate_json_payload(required_fields=['name'])
        if err:
            return err
        try:
            name   = sanitize_name(data.get('name', ''))
            score  = sanitize_integer(
                data.get('threat_score', 50), 'threat_score', 0, 100
            )
            reason = sanitize_string(
                data.get('reason', ''), 'reason',
                max_len=500, allow_empty=True
            )
        except InputError as e:
            return jsonify({'error': str(e)}), 400

        pid    = str(uuid.uuid4())[:8]
        person = add_person(pid, name, score, reason)
        if hasattr(face_detector, 'reload_db'):
            face_detector.reload_db()
        return jsonify(person), 201

    if request.method == 'DELETE':
        pid = request.args.get('id', '').strip()
        if not pid or len(pid) > 40:
            return jsonify({'error': 'Valid ID required'}), 400
        if not re.match(r'^[a-zA-Z0-9\-_]+$', pid):
            return jsonify({'error': 'Invalid ID format'}), 400
        delete_person(pid)
        if hasattr(face_detector, 'reload_db'):
            face_detector.reload_db()
        return jsonify({'status': 'deleted'})


# ══════════════════════════════════════════════════════════════════════════════
# NEIGHBORHOOD NETWORK
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/network/status')
def network_status():
    return jsonify({
        'enabled'   : neighborhood_network.is_enabled,
        'camera_id' : os.environ.get('NXV_CAMERA_ID', 'not-set'),
        'threats'   : neighborhood_network.threat_count,
        'trusted'   : neighborhood_network.trusted_count,
        'network_db': neighborhood_network.get_all_threats(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/notifications/settings', methods=['GET', 'POST'])
@rate_limit
def notification_settings():
    if request.method == 'GET':
        return jsonify(notification_manager.get_settings())

    data, err = validate_json_payload()
    if err:
        return err

    # Only allow toggling tiers 0 and 1 — tiers 2-4 are always on
    ALLOWED_NOTIF_KEYS = {'tier_0_enabled', 'tier_1_enabled'}
    cleaned = {k: v for k, v in data.items() if k in ALLOWED_NOTIF_KEYS}
    notification_manager.update_settings(cleaned)
    return jsonify(notification_manager.get_settings())


# ══════════════════════════════════════════════════════════════════════════════
# LIVE FEED — snapshot and MJPEG
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/snapshot_feed')
def snapshot_feed():
    """Single JPEG frame — used by mobile app polling every 500ms."""
    try:
        frame = app.camera.get_frame()
        import cv2
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store', 'Pragma': 'no-cache'}
        )
    except Exception:
        # Never expose internal error details to client
        return jsonify({'error': 'Frame not available'}), 503


@app.route('/video_feed')
def video_feed():
    """MJPEG stream — for desktop browsers."""
    def stream():
        while True:
            with _pipeline_lock:
                frame = _latest_frame
            if frame:
                yield frame
            time.sleep(0.05)
    return Response(
        stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/snapshot')
def snapshot():
    try:
        frame   = app.camera.get_frame()
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        snap_dir = '/home/emmanuel/camera_project/evidence/snapshots'
        os.makedirs(snap_dir, exist_ok=True)
        path    = os.path.join(snap_dir, f'snapshot_{ts}.jpg')
        import cv2
        cv2.imwrite(path, frame)
        return send_file(
            path, mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'nxv_{ts}.jpg'
        )
    except Exception:
        return jsonify({'error': 'Snapshot failed'}), 500


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/evidence_report/<incident_id>')
def evidence_report(incident_id):
    # Sanitize — only alphanumeric + underscores allowed in incident ID
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', os.path.basename(incident_id))
    if not safe_id:
        return 'Invalid ID', 400

    evidence_base = '/home/emmanuel/camera_project/evidence'
    summary_path  = os.path.join(evidence_base, safe_id, 'summary.txt')

    # Path traversal check
    if not os.path.realpath(summary_path).startswith(
        os.path.realpath(evidence_base)
    ):
        return 'Access denied', 403

    if not os.path.exists(summary_path):
        return 'Report not found', 404

    with open(summary_path) as f:
        content = f.read()

    # Escape HTML entities to prevent XSS in the pre tag
    import html
    safe_content = html.escape(content)
    return (
        f'<pre style="font-family:monospace;padding:20px;white-space:pre-wrap;'
        f'background:#0a0a0a;color:#f0f0f0;min-height:100vh;margin:0">'
        f'{safe_content}</pre>'
    )


# PREDICTIVE MODEL

@app.route('/predict/status')
def predict_status():
    return jsonify(predictive_model.get_baseline_summary())


@app.route('/predict/reset', methods=['POST'])
@rate_limit
def predict_reset():
    predictive_model.reset()
    return jsonify({'status': 'reset'})


@app.route('/predict/baseline')
def predict_baseline():
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
            'label'            : f'{hour:02d}:00',
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



# GPS STATUS


@app.route('/gps/status')
def gps_status():
    try:
        status = _gps_tracker.get_status()
        return jsonify(status)
    except Exception:
        return jsonify({'user_away': _user_away})



# PWA ROUTES

@app.route('/')
@app.route('/app')
def pwa_app():
    return send_file('/home/emmanuel/camera_project/camera/app.html')


@app.route('/manifest.json')
def manifest():
    return send_file(
        '/home/emmanuel/camera_project/camera/manifest.json',
        mimetype='application/json'
    )



# HELPER


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
