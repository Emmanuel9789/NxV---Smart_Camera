"""
NxV - Motion Clip Recorder
utils/clip_recorder.py

Saves a short clip for every motion event.
Auto-deletes based on escalation level:

  NONE / NOTIFY / ALERT  → delete after 7 days
  EMERGENCY              → keep forever
  Flagged person         → keep forever

Clips saved as MP4 for mobile browser compatibility.
"""

import cv2
import os
import time
import threading
import glob
import json
from datetime import datetime, timedelta
from utils.db import save_clip as db_save_clip




CLIPS_DIR        = '/home/emmanuel/camera_project/evidence/clips'
CLIP_FPS         = 10
CLIP_SIZE        = (320, 240)
MAX_CLIP_SECS    = 15
PRE_BUFFER_SECS  = 2
AUTO_DELETE_DAYS = 7
KEEP_LEVELS      = {"EMERGENCY"}


class ClipRecorder:
    """
    Records short video clips for every motion event.
    """

    def __init__(self):
        os.makedirs(CLIPS_DIR, exist_ok=True)
        self._recording  = False
        self._frames     = []
        self._pre_buffer = []
        self._start_time = None
        self._clip_id    = None
        self._lock       = threading.Lock()

        t = threading.Thread(target=self._auto_delete_loop, daemon=True)
        t.start()
        print(f"[NxV ClipRecorder] Ready → {CLIPS_DIR}")

    def on_frame(self, frame, motion: bool = False,
                 escalation: str = "NONE", flagged: bool = False):
        """Call every frame from generate_frames()."""
        with self._lock:
            small = cv2.resize(frame, CLIP_SIZE)

            if not motion:
                self._pre_buffer.append(small.copy())
                if len(self._pre_buffer) > PRE_BUFFER_SECS * CLIP_FPS:
                    self._pre_buffer.pop(0)
                if self._recording:
                    self._end_clip(escalation, flagged)
                return

            if not self._recording:
                self._start_clip()

            self._frames.append(small.copy())

            if len(self._frames) >= MAX_CLIP_SECS * CLIP_FPS:
                self._end_clip(escalation, flagged)

    def force_save(self, escalation: str = "EMERGENCY", flagged: bool = True):
        """Force save current clip — call on EMERGENCY."""
        with self._lock:
            if self._frames:
                self._end_clip(escalation, flagged, force=True)

    def get_clips(self, limit: int = 100,
                  escalation_filter: str = None) -> list:
        """Return saved clips with metadata, newest first."""
        meta_files = sorted(
            glob.glob(f'{CLIPS_DIR}/*.json'), reverse=True
        )[:limit * 2]

        clips = []
        for mf in meta_files:
            try:
                with open(mf) as f:
                    meta = json.load(f)
                if not os.path.exists(meta.get('path', '')):
                    continue
                if escalation_filter:
                    if escalation_filter == 'history':
                        # All non-emergency
                        if meta.get('escalation') == 'EMERGENCY':
                            continue
                    elif escalation_filter == 'threats':
                        # Only ALERT and EMERGENCY
                        if meta.get('escalation') not in ('ALERT','EMERGENCY'):
                            continue
                clips.append(meta)
                if len(clips) >= limit:
                    break
            except Exception:
                continue
        return clips

    def get_clip_path(self, clip_id: str):
        # Try both extensions
        for ext in ['.mp4', '.avi']:
            path = os.path.join(CLIPS_DIR, f'{clip_id}{ext}')
            if os.path.exists(path):
                return path
        return None

    def _start_clip(self):
        self._recording  = True
        self._start_time = datetime.now()
        self._clip_id    = self._start_time.strftime('%Y%m%d_%H%M%S_%f')[:19]
        self._frames     = list(self._pre_buffer)
        self._pre_buffer = []
        print(f"[NxV ClipRecorder] Recording → {self._clip_id}")

    def _end_clip(self, escalation, flagged, force=False):
        if not self._recording or not self._frames:
            self._recording = False
            return

        frames     = list(self._frames)
        clip_id    = self._clip_id
        start_time = self._start_time

        self._recording = False
        self._frames    = []
        self._clip_id   = None

        threading.Thread(
            target=self._write_clip,
            args=(frames, clip_id, start_time, escalation, flagged),
            daemon=True
        ).start()

    def _write_clip(self, frames, clip_id, start_time, escalation, flagged):
        try:
            video_path = os.path.join(CLIPS_DIR, f'{clip_id}.mp4')
            meta_path  = os.path.join(CLIPS_DIR, f'{clip_id}.json')

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = video_path.replace('.mp4', '.avi')
            writer = cv2.VideoWriter(video_path, fourcc, CLIP_FPS, CLIP_SIZE)
            for frame in frames:
                writer.write(frame)
            writer.release()

            duration     = round(len(frames) / CLIP_FPS, 1)
            keep_forever = escalation in KEEP_LEVELS or flagged
            delete_after = None if keep_forever else (
                datetime.now() + timedelta(days=AUTO_DELETE_DAYS)
            ).isoformat()

            meta = {
                'id'          : clip_id,
                'path'        : video_path,
                'url'         : f'/clip/{clip_id}',
                'time'        : start_time.strftime('%I:%M %p'),
                'date'        : start_time.strftime('%b %d, %Y'),
                'timestamp'   : start_time.isoformat(),
                'duration'    : duration,
                'escalation'  : escalation,
                'flagged'     : flagged,
                'keep_forever': keep_forever,
                'delete_after': delete_after,
            }

            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
                
                # Save to DB
                try:
                    import sys
                    sys.path.insert(0, '/home/emmanuel/camera_project')
                    from utils.db import save_clip as db_save_clip
                    db_save_clip(
                        id           = clip_id,
                        path         = video_path,
                        recorded_at  = start_time.isoformat(),
                        duration     = duration,
                        escalation   = escalation,
                        flagged      = flagged,
                        keep_forever = keep_forever,
                        delete_after = delete_after,
                        score        = 0,
                    )
                except Exception as e:
                    print(f"[NxV ClipRecorder] DB save error: {e}")

            keep_str = 'KEEP FOREVER' if keep_forever else f'auto-delete in {AUTO_DELETE_DAYS}d'
            print(f"[NxV ClipRecorder] Saved {clip_id} ({duration}s · {escalation} · {keep_str})")

        except Exception as e:
            print(f"[NxV ClipRecorder] Error: {e}")

    def _auto_delete_loop(self):
        while True:
            time.sleep(3600)
            self._delete_expired()

    def _delete_expired(self):
        now       = datetime.now()
        deleted   = 0
        for mf in glob.glob(f'{CLIPS_DIR}/*.json'):
            try:
                with open(mf) as f:
                    meta = json.load(f)
                if meta.get('keep_forever'):
                    continue
                da = meta.get('delete_after')
                if da and datetime.fromisoformat(da) < now:
                    vp = meta.get('path', '')
                    if os.path.exists(vp):
                        os.remove(vp)
                    os.remove(mf)
                    deleted += 1
            except Exception:
                continue
        if deleted:
            print(f"[NxV ClipRecorder] Auto-deleted {deleted} expired clips")
