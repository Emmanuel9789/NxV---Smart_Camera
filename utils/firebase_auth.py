import os
import firebase_admin
from firebase_admin import credentials, auth

_initialized = False

def _init():
    global _initialized
    if _initialized:
        return
    cred_path = os.environ.get('NXV_FIREBASE_CREDENTIALS')
    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    else:
        # App will run without auth enforcement — log warning
        print("[NxV Auth] WARNING: No Firebase credentials found. "
              "Authentication is DISABLED. Set NXV_FIREBASE_CREDENTIALS "
              "in .env to enable.")
        return
    _initialized = True
    print("[NxV Auth] Firebase Admin SDK initialized — JWT verification active")

def verify_token(id_token: str) -> dict | None:
    """Verify a Firebase ID token. Returns decoded claims or None."""
    _init()
    if not _initialized:
        return {'uid': 'unauthenticated', 'disabled': True}
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        print(f"[NxV Auth] Token verification failed: {e}")
        return None

def require_auth(f):
    """Flask decorator — rejects requests without a valid Firebase token."""
    from functools import wraps
    from flask import request, jsonify

    @wraps(f)
    def decorated(*args, **kwargs):
        _init()
        if not _initialized:
            # Auth disabled — allow all requests (development mode)
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization required'}), 401

        id_token = auth_header[7:]
        claims = verify_token(id_token)
        if not claims:
            return jsonify({'error': 'Invalid or expired token'}), 401

        request.firebase_user = claims
        return f(*args, **kwargs)

    return decorated
