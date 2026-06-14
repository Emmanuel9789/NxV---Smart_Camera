"""


Starts the NxV security system:
  - Initializes camera
  - Initializes motion detector
  - Passes everything into the Flask stream app
  - Starts the web server

"""

import os
import sys

# Startup banner
print("""
███╗   ██╗██╗  ██╗██╗   ██╗
████╗  ██║╚██╗██╔╝██║   ██║
██╔██╗ ██║ ╚███╔╝ ██║   ██║
██║╚██╗██║ ██╔██╗ ╚██╗ ██╔╝
██║ ╚████║██╔╝ ██╗ ╚████╔╝
╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝
NxV — Next Vision Security System
""")

# ── Camera 
print("[NxV] Starting camera...")
try:
    from camera.input import Camera
    camera = Camera(resolution=(256, 192))
    print("[NxV] Camera OK")
except Exception as e:
    print(f"[NxV] Camera failed: {e}")
    print("  Make sure picamera2 is installed and camera is enabled.")
    sys.exit(1)

# Motion detector 
print("[NxV] Starting motion detector...")
from detection.motion import MotionDetector
motion_detector = MotionDetector(min_area=1500)
print("[NxV] Motion detector OK")

# Evidence packager 
print("[NxV] Starting evidence packager...")
from utils.evidence import EvidencePackager
evidence_packager = EvidencePackager()
print("[NxV] Evidence packager OK")

# Flask stream app 
print("[NxV] Loading stream and all AI modules...")
from camera.stream import app, escalation_engine
from camera.stream import start_pipeline


# Inject dependencies into stream app
app.camera           = camera
app.motion_detector  = motion_detector
app.evidence         = evidence_packager

#Start Pipeline
start_pipeline()



print("[NxV] All modules loaded OK")

# Wire evidence packager into escalation 
# Monkey-patch escalation engine to save evidence on ALERT/EMERGENCY
_original_handle = escalation_engine.handle

def handle_with_evidence(threat_score, user_away=False):
    result = _original_handle(threat_score, user_away=user_away)

    # Start recording on any non-NONE threat
    if threat_score.escalation != "NONE":
        if not evidence_packager.is_recording:
            frame = camera.get_frame()
            evidence_packager.start_recording(frame)

    # Save clip on ALERT or EMERGENCY
    if threat_score.escalation in ("ALERT", "EMERGENCY"):
        clip_dir = evidence_packager.save(threat_score)
        if clip_dir:
            print(f"[NxV] Evidence saved → {clip_dir}")

    return result

escalation_engine.handle = handle_with_evidence

# Print startup info 
import socket
try:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
except Exception:
    local_ip = "unknown"

print(f"""
[NxV] System ready!
─────────────────────────────────
  Live feed : http://{local_ip}:5000
  Set away  : http://{local_ip}:5000/set_away/1
  Set home  : http://{local_ip}:5000/set_away/0
  Status    : http://{local_ip}:5000/status
  Evidence  : ./evidence/
─────────────────────────────────
  Press Ctrl+C to stop
""")

    
from utils.security import (
    rate_limit, sanitize_name, sanitize_phone,
    sanitize_string, sanitize_integer,
    validate_json_payload, check_payload_size,
    add_security_headers, InputError
)


# Start server 
if __name__ == "__main__":
    app.run(
        host       = "0.0.0.0",
        port       = 5000,
        debug      = False,
        use_reloader = False,
        threaded   = True,
    )
