"""
Structured JSON logging for NxV.
Replaces bare print() calls with machine-parseable logs.
Output goes to journald (already persistent) AND a rotating log file.
"""

import json
import time
import logging
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_FILE    = '/home/emmanuel/camera_project/nxv.log'
MAX_BYTES   = 10 * 1024 * 1024  # 10MB per file
BACKUP_COUNT = 3                  # keep 3 rotated files

class NxVLogger:
    def __init__(self):
        self._lock = threading.Lock()

        # File handler — rotating JSON logs
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes    = MAX_BYTES,
            backupCount = BACKUP_COUNT,
        )
        handler.setFormatter(logging.Formatter('%(message)s'))

        self._logger = logging.getLogger('nxv')
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(handler)

    def _write(self, level: str, component: str, message: str, **kwargs):
        entry = {
            'ts'       : datetime.utcnow().isoformat() + 'Z',
            'level'    : level,
            'component': component,
            'message'  : message,
            **kwargs,
        }
        line = json.dumps(entry)
        with self._lock:
            self._logger.info(line)
            # Also print to stdout so journald picks it up
            print(line)

    def info(self, component: str, message: str, **kwargs):
        self._write('INFO', component, message, **kwargs)

    def warn(self, component: str, message: str, **kwargs):
        self._write('WARN', component, message, **kwargs)

    def error(self, component: str, message: str, **kwargs):
        self._write('ERROR', component, message, **kwargs)

    def threat(self, component: str, message: str, **kwargs):
        self._write('THREAT', component, message, **kwargs)

    def event(self, component: str, message: str, **kwargs):
        self._write('EVENT', component, message, **kwargs)


# Singleton — import and use anywhere
log = NxVLogger()
