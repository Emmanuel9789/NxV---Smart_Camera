"""
SQLite backup — runs nightly at 2am via systemd timer.
Keeps last 7 daily backups, rotates older ones automatically.
Backs up to /home/emmanuel/nxv_backups/
"""

import os
import shutil
import time
from datetime import datetime, timedelta

DB_PATH    = '/home/emmanuel/camera_project/nxv.db'
BACKUP_DIR = '/home/emmanuel/nxv_backups'
KEEP_DAYS  = 7

def run_backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)

    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_DIR, f'nxv_{ts}.db')

    # SQLite online backup — safe to run while DB is in use
    import sqlite3
    src  = sqlite3.connect(DB_PATH)
    dest = sqlite3.connect(backup_path)
    src.backup(dest)
    src.close()
    dest.close()

    size = os.path.getsize(backup_path) / 1024 / 1024
    print(f"[NxV Backup] Saved → {backup_path} ({size:.1f} MB)")

    # Rotate — delete backups older than KEEP_DAYS
    cutoff = datetime.now() - timedelta(days=KEEP_DAYS)
    deleted = 0
    for f in os.listdir(BACKUP_DIR):
        if not f.endswith('.db'):
            continue
        fp = os.path.join(BACKUP_DIR, f)
        mt = datetime.fromtimestamp(os.path.getmtime(fp))
        if mt < cutoff:
            os.remove(fp)
            deleted += 1

    if deleted:
        print(f"[NxV Backup] Rotated {deleted} old backup(s)")
    
    print(f"[NxV Backup] Done — {len(os.listdir(BACKUP_DIR))} backup(s) kept")

if __name__ == '__main__':
    run_backup()
