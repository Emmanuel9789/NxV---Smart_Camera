"""
NxV Push Notifications via Firebase Cloud Messaging (FCM).
Sends push alerts to all registered devices when escalation fires.
"""

import os
import threading
from datetime import datetime
from utils.logger import log

try:
    from firebase_admin import messaging
    FCM_AVAILABLE = True
except ImportError:
    FCM_AVAILABLE = False
    print("[NxV Push] firebase-admin not available — push disabled")


class PushNotificationManager:
    def __init__(self):
        self._tokens_file = '/home/emmanuel/camera_project/fcm_tokens.json'
        self._lock        = threading.Lock()
        self._tokens      = self._load_tokens()
        print(f"[NxV Push] Ready — {len(self._tokens)} device(s) registered")

    def _load_tokens(self) -> dict:
        import json, os
        if not os.path.exists(self._tokens_file):
            return {}
        try:
            with open(self._tokens_file) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_tokens(self):
        import json
        with open(self._tokens_file, 'w') as f:
            json.dump(self._tokens, f, indent=2)

    def register_token(self, uid: str, token: str, device_name: str = ''):
        """Register or update a device FCM token."""
        with self._lock:
            self._tokens[uid] = {
                'token'      : token,
                'device_name': device_name,
                'registered_at': datetime.now().isoformat(),
            }
            self._save_tokens()
        log.info('push', f'Device registered', uid=uid, device=device_name)

    def unregister_token(self, uid: str):
        """Remove a device token."""
        with self._lock:
            self._tokens.pop(uid, None)
            self._save_tokens()

    def send_alert(self, escalation: str, title: str,
                   body: str, data: dict = None):
        """Send push notification to all registered devices."""
        if not FCM_AVAILABLE:
            log.warn('push', 'FCM not available — skipping push')
            return
        if not self._tokens:
            log.warn('push', 'No devices registered — skipping push')
            return

        threading.Thread(
            target=self._send_all,
            args=(escalation, title, body, data or {}),
            daemon=True,
        ).start()

    def _send_all(self, escalation: str, title: str,
                  body: str, data: dict):
        with self._lock:
            tokens = dict(self._tokens)

        failed_uids = []
        sent        = 0

        for uid, info in tokens.items():
            token = info.get('token')
            if not token:
                continue
            try:
                # Color per escalation level
                color = {
                    'EMERGENCY': '#E24B4A',
                    'ALERT'    : '#EF9F27',
                    'NOTIFY'   : '#00D4E0',
                }.get(escalation, '#888888')

                message = messaging.Message(
                    token        = token,
                    notification = messaging.Notification(
                        title = title,
                        body  = body,
                    ),
                    android = messaging.AndroidConfig(
                        priority = 'high' if escalation in ('EMERGENCY', 'ALERT') else 'normal',
                        notification = messaging.AndroidNotification(
                            color        = color,
                            sound        = 'default',
                            channel_id   = f'nxv_{escalation.lower()}',
                            click_action = 'FLUTTER_NOTIFICATION_CLICK',
                        ),
                    ),
                    data = {
                        'escalation': escalation,
                        'screen'    : 'camera',
                        **{k: str(v) for k, v in data.items()},
                    },
                )
                messaging.send(message)
                sent += 1
                log.info('push', f'Sent to {uid}', escalation=escalation)

            except messaging.UnregisteredError:
                log.warn('push', f'Token expired for {uid} — removing')
                failed_uids.append(uid)
            except Exception as e:
                log.error('push', f'Failed to send to {uid}: {e}')

        # Clean up expired tokens
        if failed_uids:
            with self._lock:
                for uid in failed_uids:
                    self._tokens.pop(uid, None)
                self._save_tokens()

        log.info('push', f'Notifications sent',
                 sent=sent, failed=len(failed_uids))


# Singleton
push_manager = PushNotificationManager()
