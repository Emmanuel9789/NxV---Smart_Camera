"""
NxV - Security Hardening
utils/security.py

Covers:
  1. Rate limiting — max 5 attempts on auth routes per 15 minutes
  2. Input sanitization — reject oversized or malformed payloads
  3. Security headers — added to every response
  4. Secret scanning helpers — check nothing is exposed

Add to stream.py:
  from utils.security import (
      rate_limit, sanitize_input, add_security_headers,
      MAX_PAYLOAD_BYTES
  )
  app.after_request(add_security_headers)
"""

import time
import re
import os
import hashlib
from functools import wraps
from flask import request, jsonify
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
RATE_LIMIT_WINDOW    = 15 * 60   # 15 minutes in seconds
RATE_LIMIT_MAX       = 5         # max attempts per window for auth routes
MAX_PAYLOAD_BYTES    = 64 * 1024 # 64KB max request body
MAX_STRING_LENGTH    = 500       # max length for any string input
MAX_NAME_LENGTH      = 100       # max length for name fields

# Routes that get strict rate limiting (auth / write operations)
RATE_LIMITED_ROUTES = {
    '/register_face/start',
    '/register_face/complete',
    '/contacts',
    '/flagged',
    '/trusted',
    '/set_away',
    '/settings',
    '/notifications/settings',
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
# Simple in-memory store: {ip: [(timestamp, route), ...]}
_rate_store = defaultdict(list)

def _get_client_ip():
    """Get real client IP, respecting proxy headers."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'

def rate_limit(f):
    """
    Decorator — applies rate limiting to a route.
    Returns 429 if client exceeds RATE_LIMIT_MAX attempts
    in the last RATE_LIMIT_WINDOW seconds.

    Usage:
        @app.route('/register_face/start', methods=['POST'])
        @rate_limit
        def register_face_start():
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        ip    = _get_client_ip()
        now   = time.time()
        route = request.path

        # Clean old entries outside the window
        _rate_store[ip] = [
            (ts, r) for ts, r in _rate_store[ip]
            if now - ts < RATE_LIMIT_WINDOW
        ]

        # Count attempts on this route from this IP
        attempts = sum(1 for _, r in _rate_store[ip] if r == route)

        if attempts >= RATE_LIMIT_MAX:
            retry_after = int(RATE_LIMIT_WINDOW - (now - min(
                ts for ts, r in _rate_store[ip] if r == route
            )))
            return jsonify({
                'error'      : 'Too many requests',
                'retry_after': max(0, retry_after),
                'message'    : f'Max {RATE_LIMIT_MAX} attempts per 15 minutes'
            }), 429

        # Record this attempt
        _rate_store[ip].append((now, route))
        return f(*args, **kwargs)
    return decorated


def check_rate_limit(route: str = None) -> bool:
    """
    Functional version — call inside a route handler.
    Returns True if request should be blocked.
    """
    ip    = _get_client_ip()
    now   = time.time()
    route = route or request.path

    _rate_store[ip] = [
        (ts, r) for ts, r in _rate_store[ip]
        if now - ts < RATE_LIMIT_WINDOW
    ]

    attempts = sum(1 for _, r in _rate_store[ip] if r == route)
    if attempts >= RATE_LIMIT_MAX:
        return True

    _rate_store[ip].append((now, route))
    return False


# ── Input sanitization ────────────────────────────────────────────────────────

class InputError(ValueError):
    """Raised when input fails sanitization."""
    pass

def sanitize_string(value, field_name: str = 'field',
                    max_len: int = MAX_STRING_LENGTH,
                    allow_empty: bool = False) -> str:
    """
    Sanitize a string input:
      - Must be a string
      - Strip whitespace
      - Reject if empty (unless allow_empty=True)
      - Reject if too long
      - Strip HTML/script tags
      - Reject null bytes

    Returns clean string or raises InputError.
    """
    if not isinstance(value, str):
        raise InputError(f'{field_name} must be a string')

    value = value.strip()

    if not allow_empty and not value:
        raise InputError(f'{field_name} cannot be empty')

    if len(value) > max_len:
        raise InputError(f'{field_name} is too long (max {max_len} characters)')

    # Reject null bytes
    if '\x00' in value:
        raise InputError(f'{field_name} contains invalid characters')

    # Strip HTML tags — prevents XSS in any rendered output
    value = re.sub(r'<[^>]+>', '', value)

    # Strip script-like content
    value = re.sub(r'javascript\s*:', '', value, flags=re.IGNORECASE)

    return value


def sanitize_name(value) -> str:
    """Sanitize a person or camera name."""
    name = sanitize_string(value, 'name', max_len=MAX_NAME_LENGTH)
    # Names: only letters, spaces, hyphens, apostrophes, numbers
    if not re.match(r"^[A-Za-z0-9 '\-\.]+$", name):
        raise InputError('Name contains invalid characters')
    return name


def sanitize_phone(value) -> str:
    """Sanitize a phone number — digits, +, spaces, hyphens, parens only."""
    phone = sanitize_string(value, 'phone', max_len=20)
    if not re.match(r'^[\d\s\+\-\(\)\.]+$', phone):
        raise InputError('Phone number contains invalid characters')
    digits = re.sub(r'\D', '', phone)
    if len(digits) < 7 or len(digits) > 15:
        raise InputError('Phone number must be 7-15 digits')
    return phone


def sanitize_integer(value, field_name: str = 'value',
                     min_val: int = None, max_val: int = None) -> int:
    """Sanitize an integer input with optional range check."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise InputError(f'{field_name} must be an integer')
    if min_val is not None and n < min_val:
        raise InputError(f'{field_name} must be at least {min_val}')
    if max_val is not None and n > max_val:
        raise InputError(f'{field_name} must be at most {max_val}')
    return n


def check_payload_size() -> bool:
    """
    Returns True if request body exceeds MAX_PAYLOAD_BYTES.
    Call at the start of any route that accepts a body.
    """
    length = request.content_length
    if length and length > MAX_PAYLOAD_BYTES:
        return True
    # Also check actual data for chunked requests
    try:
        data = request.get_data(as_text=False)
        return len(data) > MAX_PAYLOAD_BYTES
    except Exception:
        return False


def validate_json_payload(required_fields: list = None) -> tuple:
    """
    Parse and validate a JSON request body.

    Returns (data, error_response) where error_response is None on success.
    """
    if check_payload_size():
        return None, (jsonify({'error': 'Payload too large'}), 413)

    data = request.get_json(silent=True)
    if data is None:
        return None, (jsonify({'error': 'Invalid or missing JSON body'}), 400)

    if required_fields:
        for field in required_fields:
            if field not in data:
                return None, (jsonify({'error': f'Missing required field: {field}'}), 400)

    return data, None


# ── Security headers ──────────────────────────────────────────────────────────

def add_security_headers(response):
    """
    Add security headers to every response.
    Register with: app.after_request(add_security_headers)
    """
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'

    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # XSS protection (legacy browsers)
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # Only send referrer for same-origin requests
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Permissions policy — disable features NxV doesn't use
    response.headers['Permissions-Policy'] = (
        'geolocation=(), '
        'microphone=(), '
        'camera=(), '
        'payment=()'
    )

    # Content Security Policy — restrict what the browser can load
    # Allows: same origin, Google Fonts, inline styles (needed for PWA)
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    # HSTS — only add if running over HTTPS
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = (
            'max-age=31536000; includeSubDomains'
        )

    return response


# ── Secret scanner ────────────────────────────────────────────────────────────

# Patterns that should never appear in source code
SECRET_PATTERNS = [
    (r'AC[a-f0-9]{32}',            'Twilio Account SID'),
    (r'SK[a-f0-9]{32}',            'Twilio API Key'),
    (r'AIza[0-9A-Za-z\-_]{35}',    'Google API Key'),
    (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
    (r'secret\s*=\s*["\'][^"\']+["\']',   'Hardcoded secret'),
    (r'api_key\s*=\s*["\'][^"\']+["\']',  'Hardcoded API key'),
]

def scan_file_for_secrets(filepath: str) -> list:
    """
    Scan a file for hardcoded secrets.
    Returns list of (line_number, pattern_name, snippet) tuples.
    """
    findings = []
    try:
        with open(filepath, 'r', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                # Skip .env files and comments
                stripped = line.strip()
                if stripped.startswith('#') or 'export ' in stripped:
                    continue
                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        snippet = line.strip()[:80]
                        findings.append((i, name, snippet))
    except Exception:
        pass
    return findings


def scan_project(project_root: str) -> dict:
    """
    Scan entire project for hardcoded secrets.
    Returns dict of {filepath: [findings]}
    """
    results = {}
    extensions = {'.py', '.js', '.html', '.json', '.yaml', '.yml', '.sh'}
    skip_dirs  = {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}

    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if any(fname.endswith(ext) for ext in extensions):
                fpath = os.path.join(root, fname)
                findings = scan_file_for_secrets(fpath)
                if findings:
                    results[fpath] = findings

    return results


# ── Environment variable checker ──────────────────────────────────────────────

REQUIRED_ENV_VARS = [
    'NXV_TWILIO_SID',
    'NXV_TWILIO_TOKEN',
    'NXV_TWILIO_FROM',
    'NXV_OWNER_PHONE',
    'NXV_NETWORK_KEY',
]

def check_env_vars() -> list:
    """
    Check all required env vars are set.
    Returns list of missing variable names.
    """
    return [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]


def audit_on_startup():
    """
    Run security checks on startup and print report.
    Call this in main.py after all imports.
    """
    print("\n[NxV Security] Running startup audit...")

    # Check env vars
    missing = check_env_vars()
    if missing:
        print(f"[NxV Security] ⚠ Missing env vars: {', '.join(missing)}")
        print("  → Run: source /home/emmanuel/camera_project/.env")
    else:
        print("[NxV Security] ✓ All required env vars present")

    # Scan for hardcoded secrets in Python files only (fast)
    project = '/home/emmanuel/camera_project'
    findings = scan_project(project)
    if findings:
        print(f"[NxV Security] ⚠ Potential hardcoded secrets found:")
        for fpath, items in findings.items():
            rel = fpath.replace(project + '/', '')
            for line_no, name, snippet in items:
                print(f"  {rel}:{line_no} — {name}")
                print(f"    {snippet[:60]}...")
    else:
        print("[NxV Security] ✓ No hardcoded secrets detected")

    print("[NxV Security] Audit complete\n")
