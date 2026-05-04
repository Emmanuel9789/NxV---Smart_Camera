"""
NxV - SQLite Database
utils/db.py

Single source of truth for all NxV data.
Replaces: flagged_persons.json, clip metadata JSONs, .env contacts

Tables:
  persons   → flagged persons + sighting history
  clips     → all motion/threat clip metadata
  alerts    → full alert history
  contacts  → trusted contacts + owner
  motion    → motion event log
  settings  → system settings key/value
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = '/home/emmanuel/camera_project/nxv.db'


def get_conn():
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    c    = conn.cursor()

    # ── Flagged persons ───────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            threat_score  INTEGER NOT NULL DEFAULT 50,
            reason        TEXT,
            added_at      TEXT NOT NULL,
            last_seen_at  TEXT,
            sightings     INTEGER DEFAULT 0,
            is_trusted    INTEGER DEFAULT 0,   -- 1 = safe zone (family/friends)
            notes         TEXT,
            embedding_path TEXT               -- path to face embedding file
        )
    """)

    # ── Clips ─────────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id            TEXT PRIMARY KEY,
            path          TEXT NOT NULL,
            url           TEXT,
            recorded_at   TEXT NOT NULL,
            duration      REAL DEFAULT 0,
            escalation    TEXT DEFAULT 'NONE',
            flagged       INTEGER DEFAULT 0,
            keep_forever  INTEGER DEFAULT 0,
            delete_after  TEXT,
            score         INTEGER DEFAULT 0,
            person_id     TEXT,
            flags         TEXT,              -- JSON array of flags
            FOREIGN KEY (person_id) REFERENCES persons(id)
        )
    """)

    # ── Alerts ────────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fired_at      TEXT NOT NULL,
            escalation    TEXT NOT NULL,
            score         INTEGER DEFAULT 0,
            flags         TEXT,              -- JSON array
            clip_id       TEXT,
            owner_reached INTEGER DEFAULT 0,
            contacts_tried TEXT,             -- JSON array of contact names tried
            contacts_reached TEXT,           -- JSON array of who answered
            FOREIGN KEY (clip_id) REFERENCES clips(id)
        )
    """)

    # ── Contacts ──────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            phone         TEXT NOT NULL,
            role          TEXT DEFAULT 'contact',  -- 'owner' or 'contact'
            priority      INTEGER DEFAULT 1,        -- call order
            country_code  TEXT DEFAULT '+1',
            active        INTEGER DEFAULT 1
        )
    """)

    # ── Motion log ────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS motion (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at   TEXT NOT NULL,
            description   TEXT,
            persons_count INTEGER DEFAULT 0,
            score         INTEGER DEFAULT 0,
            escalation    TEXT DEFAULT 'NONE',
            clip_id       TEXT,
            is_threat     INTEGER DEFAULT 0,
            FOREIGN KEY (clip_id) REFERENCES clips(id)
        )
    """)

    # ── Settings ──────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key           TEXT PRIMARY KEY,
            value         TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[NxV DB] Initialized → {DB_PATH}")


# ════════════════════════════════════════════════════════════════
# PERSONS
# ════════════════════════════════════════════════════════════════

def add_person(id: str, name: str, threat_score: int,
               reason: str, is_trusted: bool = False,
               embedding_path: str = None) -> dict:
    conn = get_conn()
    now  = datetime.now().isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO persons
        (id, name, threat_score, reason, added_at, is_trusted, embedding_path)
        VALUES (?,?,?,?,?,?,?)
    """, (id, name, threat_score, reason, now, int(is_trusted), embedding_path))
    conn.commit()
    person = get_person(id, conn=conn)
    conn.close()
    return person

def get_person(id: str, conn=None) -> dict | None:
    c = conn or get_conn()
    row = c.execute("SELECT * FROM persons WHERE id=?", (id,)).fetchone()
    return dict(row) if row else None

def get_all_persons(trusted_only: bool = False,
                    flagged_only: bool = False) -> list:
    conn = get_conn()
    if trusted_only:
        rows = conn.execute(
            "SELECT * FROM persons WHERE is_trusted=1 ORDER BY name"
        ).fetchall()
    elif flagged_only:
        rows = conn.execute(
            "SELECT * FROM persons WHERE is_trusted=0 ORDER BY threat_score DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM persons ORDER BY threat_score DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_person_sighting(person_id: str):
    conn = get_conn()
    conn.execute("""
        UPDATE persons
        SET sightings=sightings+1, last_seen_at=?
        WHERE id=?
    """, (datetime.now().isoformat(), person_id))
    conn.commit()
    conn.close()

def delete_person(person_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════
# CLIPS
# ════════════════════════════════════════════════════════════════

def save_clip(id: str, path: str, recorded_at: str,
              duration: float, escalation: str,
              flagged: bool = False, keep_forever: bool = False,
              delete_after: str = None, score: int = 0,
              person_id: str = None, flags: list = None):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO clips
        (id, path, url, recorded_at, duration, escalation,
         flagged, keep_forever, delete_after, score, person_id, flags)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        id, path, f'/clip/{id}', recorded_at, duration, escalation,
        int(flagged), int(keep_forever), delete_after, score,
        person_id, json.dumps(flags or [])
    ))
    conn.commit()
    conn.close()

def get_clips(filter_type: str = 'all', limit: int = 100) -> list:
    conn  = get_conn()
    if filter_type == 'threats':
        rows = conn.execute("""
            SELECT * FROM clips
            WHERE escalation IN ('ALERT','EMERGENCY')
            AND (path IS NOT NULL)
            ORDER BY recorded_at DESC LIMIT ?
        """, (limit,)).fetchall()
    elif filter_type == 'history':
        rows = conn.execute("""
            SELECT * FROM clips
            WHERE escalation NOT IN ('EMERGENCY')
            AND (path IS NOT NULL)
            ORDER BY recorded_at DESC LIMIT ?
        """, (limit,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM clips
            ORDER BY recorded_at DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if os.path.exists(d.get('path','')):
            d['flags'] = json.loads(d.get('flags') or '[]')
            result.append(d)
    return result

def get_clip(clip_id: str) -> dict | None:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['flags'] = json.loads(d.get('flags') or '[]')
        return d
    return None

def delete_expired_clips():
    """Delete clips past their expiry date."""
    conn    = get_conn()
    now     = datetime.now().isoformat()
    expired = conn.execute("""
        SELECT id, path FROM clips
        WHERE keep_forever=0
        AND delete_after IS NOT NULL
        AND delete_after < ?
    """, (now,)).fetchall()

    deleted = 0
    for row in expired:
        try:
            if row['path'] and os.path.exists(row['path']):
                os.remove(row['path'])
            # Also remove meta json if exists
            meta = row['path'].replace('.mp4', '.json')
            if os.path.exists(meta):
                os.remove(meta)
            conn.execute("DELETE FROM clips WHERE id=?", (row['id'],))
            deleted += 1
        except Exception as e:
            print(f"[NxV DB] Delete error: {e}")

    conn.commit()
    conn.close()
    if deleted:
        print(f"[NxV DB] Auto-deleted {deleted} expired clips")
    return deleted


# ════════════════════════════════════════════════════════════════
# ALERTS
# ════════════════════════════════════════════════════════════════

def save_alert(escalation: str, score: int, flags: list,
               clip_id: str = None, owner_reached: bool = False,
               contacts_tried: list = None,
               contacts_reached: list = None) -> int:
    conn = get_conn()
    cur  = conn.execute("""
        INSERT INTO alerts
        (fired_at, escalation, score, flags, clip_id,
         owner_reached, contacts_tried, contacts_reached)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(), escalation, score,
        json.dumps(flags or []), clip_id, int(owner_reached),
        json.dumps(contacts_tried or []),
        json.dumps(contacts_reached or [])
    ))
    alert_id = cur.lastrowid
    conn.commit()
    conn.close()
    return alert_id

def get_alerts(limit: int = 100, escalation: str = None) -> list:
    conn = get_conn()
    if escalation:
        rows = conn.execute("""
            SELECT * FROM alerts WHERE escalation=?
            ORDER BY fired_at DESC LIMIT ?
        """, (escalation, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM alerts ORDER BY fired_at DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['flags']             = json.loads(d.get('flags') or '[]')
        d['contacts_tried']    = json.loads(d.get('contacts_tried') or '[]')
        d['contacts_reached']  = json.loads(d.get('contacts_reached') or '[]')
        result.append(d)
    return result


# ════════════════════════════════════════════════════════════════
# CONTACTS
# ════════════════════════════════════════════════════════════════

def save_contacts(contacts: list):
    """Replace all contacts with new list."""
    conn = get_conn()
    conn.execute("DELETE FROM contacts")
    for i, c in enumerate(contacts):
        conn.execute("""
            INSERT INTO contacts (name, phone, role, priority, country_code, active)
            VALUES (?,?,?,?,?,?)
        """, (
            c.get('name',''), c.get('phone',''),
            c.get('role','contact'), i,
            c.get('country_code','+1'), 1
        ))
    conn.commit()
    conn.close()

def get_contacts() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM contacts WHERE active=1 ORDER BY priority"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_contact(name: str, phone: str, role: str = 'contact',
                country_code: str = '+1') -> dict:
    conn     = get_conn()
    max_pri  = conn.execute(
        "SELECT MAX(priority) FROM contacts"
    ).fetchone()[0] or 0
    cur = conn.execute("""
        INSERT INTO contacts (name, phone, role, priority, country_code, active)
        VALUES (?,?,?,?,?,1)
    """, (name, phone, role, max_pri + 1, country_code))
    cid = cur.lastrowid
    conn.commit()
    contact = dict(conn.execute(
        "SELECT * FROM contacts WHERE id=?", (cid,)
    ).fetchone())
    conn.close()
    return contact

def remove_contact(contact_id: int):
    conn = get_conn()
    conn.execute("UPDATE contacts SET active=0 WHERE id=?", (contact_id,))
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════
# MOTION LOG
# ════════════════════════════════════════════════════════════════

def log_motion(description: str, persons_count: int,
               score: int, escalation: str,
               clip_id: str = None, is_threat: bool = False):
    conn = get_conn()
    conn.execute("""
        INSERT INTO motion
        (detected_at, description, persons_count, score,
         escalation, clip_id, is_threat)
        VALUES (?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(), description,
        persons_count, score, escalation,
        clip_id, int(is_threat)
    ))
    conn.commit()
    conn.close()

def get_motion_log(limit: int = 200, threats_only: bool = False) -> list:
    conn = get_conn()
    if threats_only:
        rows = conn.execute("""
            SELECT * FROM motion WHERE is_threat=1
            ORDER BY detected_at DESC LIMIT ?
        """, (limit,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM motion ORDER BY detected_at DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════

DEFAULT_SETTINGS = {
    "gps"            : "true",
    "deterrent"      : "true",
    "audio"          : "false",
    "social"         : "true",
    "history"        : "true",
    "home_lat"       : "0.0",
    "home_lon"       : "0.0",
    "away_threshold" : "0.5",
    "auto_delete_days": "7",
}

def get_setting(key: str, default=None):
    conn = get_conn()
    row  = conn.execute(
        "SELECT value FROM settings WHERE key=?", (key,)
    ).fetchone()
    conn.close()
    if row:
        return row[0]
    return DEFAULT_SETTINGS.get(key, default)

def set_setting(key: str, value):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO settings (key, value, updated_at)
        VALUES (?,?,?)
    """, (key, str(value), datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_all_settings() -> dict:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    settings = dict(DEFAULT_SETTINGS)
    settings.update({r['key']: r['value'] for r in rows})
    return settings

def set_settings(data: dict):
    conn = get_conn()
    now  = datetime.now().isoformat()
    for key, value in data.items():
        conn.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?,?,?)
        """, (key, str(value), now))
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════
# MIGRATION — import existing JSON data into SQLite
# ════════════════════════════════════════════════════════════════

def migrate_existing_data():
    """
    One-time migration of existing JSON files into SQLite.
    Safe to run multiple times — skips existing records.
    """
    import uuid

    print("[NxV DB] Running migration...")

    # Migrate flagged_persons.json
    persons_path = '/home/emmanuel/camera_project/datasets/flagged_persons.json'
    if os.path.exists(persons_path):
        with open(persons_path) as f:
            data = json.load(f)
        for p in data.get('persons', []):
            try:
                add_person(
                    id           = p.get('id', str(uuid.uuid4())[:8]),
                    name         = p.get('name', 'Unknown'),
                    threat_score = p.get('threat_score', 50),
                    reason       = p.get('reason', ''),
                )
                print(f"[NxV DB] Migrated person: {p.get('name')}")
            except Exception as e:
                print(f"[NxV DB] Skip person {p.get('name')}: {e}")

    # Migrate clip JSON files
    clips_dir = '/home/emmanuel/camera_project/evidence/clips'
    if os.path.exists(clips_dir):
        import glob
        for mf in glob.glob(f'{clips_dir}/*.json'):
            try:
                with open(mf) as f:
                    meta = json.load(f)
                save_clip(
                    id           = meta['id'],
                    path         = meta['path'],
                    recorded_at  = meta.get('timestamp', datetime.now().isoformat()),
                    duration     = meta.get('duration', 0),
                    escalation   = meta.get('escalation', 'NONE'),
                    flagged      = meta.get('flagged', False),
                    keep_forever = meta.get('keep_forever', False),
                    delete_after = meta.get('delete_after'),
                    score        = meta.get('score', 0),
                )
            except Exception:
                continue
        print("[NxV DB] Clips migrated")

    # Migrate .env contacts
    env_path = '/home/emmanuel/camera_project/.env'
    if os.path.exists(env_path):
        phones = {}
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('export NXV_'):
                    key, _, val = line.replace('export ', '').partition('=')
                    phones[key] = val.strip('"')

        contacts_to_add = []
        if phones.get('NXV_OWNER_PHONE'):
            contacts_to_add.append({
                'name': 'Owner', 'phone': phones['NXV_OWNER_PHONE'],
                'role': 'owner', 'country_code': '+1'
            })
        for i in range(1, 4):
            k = f'NXV_CONTACT_{i}_PHONE'
            if phones.get(k):
                contacts_to_add.append({
                    'name': f'Contact {i}', 'phone': phones[k],
                    'role': 'contact', 'country_code': '+1'
                })
        if contacts_to_add:
            save_contacts(contacts_to_add)
            print(f"[NxV DB] Migrated {len(contacts_to_add)} contacts")

    print("[NxV DB] Migration complete")


if __name__ == '__main__':
    init_db()
    migrate_existing_data()
    print("\n[NxV DB] Database ready!")
    print(f"  Persons  : {len(get_all_persons())}")
    print(f"  Clips    : {len(get_clips())}")
    print(f"  Contacts : {len(get_contacts())}")
    print(f"  Alerts   : {len(get_alerts())}")
