# NxV — Next Vision Security System

An AI-powered security camera system built on Raspberry Pi 4. NxV performs real-time threat detection, face recognition, weapon detection, and behavioral analysis — and sends alerts through a full emergency call chain when threats are detected.

Built as both an independent project, with every major data structure implemented and justified in the codebase.

---

## What it does

- **Weapon detection** — YOLOv8 detects guns and knives in real time via an async background thread so the stream never drops below 30 FPS
- **Face recognition** — identifies registered trusted persons and flags known dangerous persons using 128-dimensional face embeddings
- **Behavioral analysis** — detects loitering, pacing, and scanning patterns using sliding window path history
- **Threat scoring** — multi-factor weighted algorithm combining behavior, violence, weapon, face, and time-of-night signals into a 0–100 score with human-readable explanations
- **Emergency call chain** — on EMERGENCY, automatically calls Owner → Contact 1 → Contact 2 → Contact 3 → SMS 911 prompt via Twilio
- **Delivery detection** — identifies UPS, FedEx, Amazon, USPS, and DHL carriers at the door
- **Predictive modeling** — learns hourly activity patterns and boosts threat scores during abnormally quiet or busy periods
- **Neighborhood network** — shares anonymized face embeddings across cameras via a relay server so threats flagged by one camera are recognized by all
- **GPS-aware escalation** — detects when the homeowner leaves and raises alert thresholds automatically
- **Evidence packaging** — clips are SHA256-hashed and formatted for law enforcement

---

## Stack

| Layer | Technology |
|---|---|
| Hardware | Raspberry Pi 4 (4GB), IMX708 camera |
| Detection | YOLOv8 (ultralytics), face_recognition, OpenCV |
| Backend | Python 3.11, Flask, SQLite |
| Alerts | Twilio (calls + SMS) |
| Frontend | PWA (HTML/CSS/JS, IBM Plex Sans, cyan/grey design) |
| Containerization | Docker, Docker Compose |
| Network | Neighborhood relay server (Flask, containerized) |

---

## Architecture

```
camera/
  stream.py          Flask app, all routes, live feed pipeline
  app.html           PWA mobile frontend

detection/
  motion.py          Frame differencing + Gaussian blur
  person.py          YOLOv8 weapon detection (async thread)
  face.py            face_recognition embeddings

tracking/
  tracker.py         IoU bipartite graph matching — maintains person IDs across frames

ai/
  behavior.py        Sliding window path history — loitering, pacing, scanning
  violence.py        Aggression detection
  safe_zone.py       Trusted face registration and matching
  delivery.py        Carrier uniform + package detection
  neighborhood.py    Network threat sharing
  predictive.py      Time-based baseline model (Option A)
  social_search.py   Reverse image search on unknown faces

scoring/
  threat_score.py    Weighted sum + hard rule decision tree

alerts/
  escalation.py      Threshold → escalation level
  call_chain.py      Owner → contacts → 911 call queue
  notification_tiers.py  5-tier notification system
  deterrent.py       Siren on ALERT+

utils/
  db.py              SQLite with B-tree indexes
  security.py        Rate limiting, input sanitization, security headers
  clip_recorder.py   Evidence clips + SHA256 hashing

neighborhood_server.py   Relay server (runs in Docker on VPS)
main.py                  Entry point
```



## Threat scoring

```
Score = (behavior × 0.20) + (violence × 0.25) + (weapon × 0.30)
      + (face × 0.15) + (time × 0.10)
```

Hard rule overrides (evaluated first, O(1) each):
- Weapon aimed at camera → score 100, EMERGENCY immediately
- Break-in at door during nighttime → score 100, EMERGENCY immediately
- Passing motion (dwell < 4s, speed > 80px/s) → score capped at 10, no alert

Escalation levels:

| Score | Level | Action |
|---|---|---|
| 75–100 | EMERGENCY | Full call chain + SMS + siren |
| 55–74 | ALERT | SMS to owner + siren |
| 30–54 | NOTIFY | Push notification only |
| 0–29 | NONE | Silent log |

---

## Notification tiers

| Tier | Name | Can disable |
|---|---|---|
| 0 | Silent motion log | Yes |
| 1 | Deliveries and visitors | Yes |
| 2 | Security alerts | No — always on |
| 3 | Threat alerts | No — always on |
| 4 | Emergency | No — always on |

---

## Setup

### Requirements

- Raspberry Pi 4 (4GB RAM recommended)
- Raspberry Pi Camera Module 3 (IMX708)
- Raspberry Pi OS Bookworm (64-bit)
- Docker (for containerized deployment)
- Twilio account (for call chain)

### Installation

```bash
# Clone the repo
git clone https://github.com/USERNAME/NxV---Smart_Camera.git
cd NxV---Smart_Camera

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
nano .env
```

### Environment variables

```bash
NXV_TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NXV_TWILIO_TOKEN=your_auth_token
NXV_TWILIO_FROM=+1xxxxxxxxxx
NXV_OWNER_PHONE=+1xxxxxxxxxx
NXV_CONTACT_1_PHONE=+1xxxxxxxxxx
NXV_CONTACT_2_PHONE=+1xxxxxxxxxx
NXV_CONTACT_3_PHONE=+1xxxxxxxxxx
NXV_RELAY_URL=http://your-relay-server:6001
NXV_NETWORK_KEY=your-secret-key
NXV_CAMERA_ID=your-camera-id
NXV_GOOGLE_API_KEY=your-google-api-key
NXV_TEST_MODE=true
```

### Run without Docker

```bash
source .env
python main.py
```

Access the app at `http://<pi-ip>:5000/app`

### Run with Docker

```bash
# Start relay server (run this on a VPS or separate machine)
cd /path/to/nxv-relay
docker compose up -d

# Start Pi app
docker compose -f docker-compose.pi.yml up -d

# Check logs
docker compose -f docker-compose.pi.yml logs -f
```

---

## Mobile app

The PWA frontend is served at `/app`. It works on any phone browser — add it to your home screen for a native app experience.

Features:
- Live camera feed with threat overlay
- Real-time score, FPS, and person count
- Playback controls for recorded clips
- Mic toggle, snapshot, Call 911 button
- Settings: GPS override, notification tiers, motion sensitivity, privacy features
- Trusted face registration (captures 15 frames from the live feed)
- Account management: name, email, phone, two-step verification
- Device health: Wi-Fi signal, firmware version, IP address

---

## Security

- Rate limiting on all write endpoints (max 5 attempts per 15 minutes)
- Input sanitization on all user-facing routes
- Path traversal protection on file serving routes
- Security headers on all responses (CSP, X-Frame-Options, HSTS)
- No secrets in source code — all sensitive data in environment variables
- Face embeddings stored locally only — never transmitted as raw data
- `.env`, `nxv.db`, and `face_embeddings.pkl` excluded from version control

---

## Legal

- Surveillance notice posted at property (Maryland law)
- Biometric data (face embeddings) collected with explicit consent only
- Face embeddings cannot be reversed to reconstruct a face image
- DMCA Designated Agent registered at copyright.gov
- CAN-SPAM compliant for future email features

---

## Author

Emmanuel  
Computer Science, AI concentration  


---

## License

Proprietary. All rights reserved.
