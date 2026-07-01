import time
import threading

class SimpleCache:
    """Lightweight in-memory cache for Flask responses."""
    
    def __init__(self):
        self._store = {}
        self._lock  = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if time.time() > entry['expires']:
                del self._store[key]
                return None
            return entry['value']

    def set(self, key: str, value, ttl_seconds: int = 2):
        with self._lock:
            self._store[key] = {
                'value'  : value,
                'expires': time.time() + ttl_seconds,
            }

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()

    def cache_route(self, ttl_seconds: int = 2):
        """Decorator — caches a Flask route response."""
        from functools import wraps
        from flask import request, jsonify, Response
        import json

        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                key = f"{f.__name__}:{request.full_path}"
                cached = self.get(key)
                if cached:
                    return Response(
                        cached['data'],
                        status=200,
                        mimetype='application/json',
                        headers=cached['headers'],
                    )
                response = f(*args, **kwargs)
                if hasattr(response, 'status_code') and response.status_code == 200:
                    self.set(key, {
                        'data'   : response.get_data(),
                        'headers': dict(response.headers),
                    }, ttl_seconds)
                return response
            return decorated
        return decorator

_cache = SimpleCache()
