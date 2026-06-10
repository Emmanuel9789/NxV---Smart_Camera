"""
Handles both threat sharing and street trust circle.
Run on any VPS or your PC for testing.

Run:
  python neighborhood_server.py

Environment:
  NXV_NETWORK_KEY   shared secret (all cameras must use same key)
  PORT              port to listen on (default 6000)
"""

import os
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

NETWORK_KEY = os.environ.get("NXV_NETWORK_KEY", "nxv-default-key-change-me")
PORT        = int(os.environ.get("PORT", 6000))

# Storage (replace with Redis/PostgreSQL for production) 
_threats    = {}   # person_id → threat record
_trusted    = {}   # person_id → trusted record
_cameras    = {}   # camera_id → last_seen
_audit_log  = []


def auth(req) -> bool:
    return req.headers.get("X-Network-Key") == NETWORK_KEY


def log(action, camera_id, name, extra=""):
    entry = {
        "at"       : datetime.now().isoformat(),
        "action"   : action,
        "camera"   : camera_id,
        "name"     : name,
        "extra"    : extra,
    }
    _audit_log.insert(0, entry)
    if len(_audit_log) > 2000:
        _audit_log.pop()
    print(f"[Relay] {action}: {name} from {camera_id} {extra}")


# Health 

@app.route('/health')
def health():
    return jsonify({
        "status"   : "online",
        "threats"  : len(_threats),
        "trusted"  : len(_trusted),
        "cameras"  : len(_cameras),
        "timestamp": datetime.now().isoformat(),
    })


# Threat sharing 

@app.route('/share', methods=['POST'])
def share_threat():
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    data      = request.get_json(silent=True) or {}
    person_id = data.get("person_id")
    camera_id = data.get("camera_id", "unknown")

    if not person_id or "embedding" not in data:
        return jsonify({"error": "person_id and embedding required"}), 400

    _threats[person_id] = {
        **data,
        "received_at": datetime.now().isoformat(),
    }
    _cameras[camera_id] = datetime.now().isoformat()

    severity = data.get("crime_severity", "MINOR")
    log("THREAT_SHARED", camera_id, data.get("name","?"),
        f"severity={severity}")

    return jsonify({
        "status"     : "shared",
        "person_id"  : person_id,
        "distributed": max(0, len(_cameras) - 1),
    })


@app.route('/threats')
def get_threats():
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    camera_id = request.headers.get("X-Camera-ID", "unknown")
    _cameras[camera_id] = datetime.now().isoformat()

    # Return threats not from this camera
    threats = [
        t for t in _threats.values()
        if t.get("camera_id") != camera_id
    ]

    return jsonify({
        "threats"  : threats,
        "count"    : len(threats),
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/threats/<person_id>', methods=['DELETE'])
def remove_threat(person_id):
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    if person_id in _threats:
        del _threats[person_id]
        return jsonify({"status": "removed"})
    return jsonify({"error": "Not found"}), 404


# Street trust circle 

@app.route('/share_trusted', methods=['POST'])
def share_trusted():
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    data      = request.get_json(silent=True) or {}
    person_id = data.get("person_id")
    camera_id = data.get("camera_id", "unknown")

    if not person_id:
        return jsonify({"error": "person_id required"}), 400

    _trusted[person_id] = {
        **data,
        "received_at": datetime.now().isoformat(),
    }
    _cameras[camera_id] = datetime.now().isoformat()

    log("TRUSTED_SHARED", camera_id, data.get("name","?"),
        f"relationship={data.get('relationship','neighbor')}")

    return jsonify({"status": "shared", "person_id": person_id})


@app.route('/trusted')
def get_trusted():
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    camera_id = request.headers.get("X-Camera-ID", "unknown")

    # Return trusted persons not from this camera
    trusted = [
        t for t in _trusted.values()
        if t.get("camera_id") != camera_id
    ]

    return jsonify({
        "trusted"  : trusted,
        "count"    : len(trusted),
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/trusted/<person_id>', methods=['DELETE'])
def remove_trusted(person_id):
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    if person_id in _trusted:
        del _trusted[person_id]
        return jsonify({"status": "removed"})
    return jsonify({"error": "Not found"}), 404


# Admin 

@app.route('/cameras')
def get_cameras():
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"cameras": _cameras, "count": len(_cameras)})


@app.route('/audit')
def audit():
    if not auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"log": _audit_log[:100]})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    print(f"""
NxV Neighborhood Relay Server v2
══════════════════════════════════════
  URL        : http://{args.host}:{args.port}
  Network key: {NETWORK_KEY[:8]}...

  Supports:
    → Global threat sharing
    → Street trust circle
    → Audit log

  To connect cameras:
    export NXV_RELAY_URL="http://YOUR_IP:{args.port}"
    export NXV_NETWORK_KEY="{NETWORK_KEY}"
    export NXV_CAMERA_ID="your-camera-name"
══════════════════════════════════════
""")
    app.run(host=args.host, port=args.port, debug=False)
