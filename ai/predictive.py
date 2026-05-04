"""
NxV - Predictive Threat Modeling (Option A — Time-Based)
ai/predictive.py

Learns YOUR home's normal activity patterns over time.
After 24 hours of data, flags activity that deviates from
your personal baseline — not just generic "night = dangerous".

How it works:
  1. Every motion event is logged with hour, day, duration
  2. After MIN_EVENTS, a baseline is built per hour of day
  3. When motion detected, checks if this hour is normally active
  4. If unusually quiet hour → score boost
  5. If person stays much longer than usual → earlier loitering flag

Stored in SQLite DB — survives restarts.
Improves accuracy the longer NxV runs.
"""

import os
import json
import time
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


DB_PATH = '/home/emmanuel/camera_project/nxv.db'

# Minimum events before predictions are made
MIN_EVENTS_FOR_BASELINE = 20

# How many days of history to use
HISTORY_DAYS = 30

# Score boost ranges
BOOST_QUIET_HOUR    = 20   # this hour is normally dead quiet
BOOST_UNUSUAL_TIME  = 10   # this hour has low activity
BOOST_LONG_DWELL    = 15   # person staying much longer than usual
BOOST_FREQUENT_NEW  = 10   # new face appearing repeatedly


class PredictiveModel:
    """
    Time-based predictive threat model for NxV.

    Usage:
        model = PredictiveModel()
        model.log_event(hour=2, duration=45, had_person=True)
        boost, reason = model.get_score_boost(hour=2, current_dwell=60)
    """

    def __init__(self):
        self._ensure_table()
        self._baseline    = {}    # hour → stats dict
        self._last_built  = 0
        self._event_count = 0
        self._build_baseline()
        print(f"[NxV Predict] Model ready — "
              f"{self._event_count} events in history")

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_event(self, hour: int = None, duration: float = 0,
                  had_person: bool = True, score: int = 0,
                  escalation: str = "NONE"):
        """
        Log a motion/detection event to build the baseline.
        Call this every time motion is detected.
        """
        if hour is None:
            hour = datetime.now().hour

        day_of_week = datetime.now().weekday()   # 0=Monday 6=Sunday

        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO predictive_events
            (logged_at, hour, day_of_week, duration, had_person, score, escalation)
            VALUES (?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            hour, day_of_week, duration,
            int(had_person), score, escalation
        ))
        conn.commit()
        conn.close()

        self._event_count += 1

        # Rebuild baseline every 100 new events
        if self._event_count % 100 == 0:
            self._build_baseline()

    def get_score_boost(self, hour: int = None,
                        current_dwell: float = 0,
                        is_new_face: bool = False) -> tuple:
        """
        Get a score boost based on whether this moment is anomalous.

        Returns (boost, reason) where boost is 0-35 and reason is a string.
        Returns (0, None) if not enough data yet.
        """
        if hour is None:
            hour = datetime.now().hour

        if not self._has_enough_data():
            return 0, None

        boost   = 0
        reasons = []

        stats = self._baseline.get(hour)

        if stats:
            avg_events = stats.get('avg_events_per_day', 0)
            avg_dwell  = stats.get('avg_duration', 0)

            # ── Quiet hour boost ──────────────────────────────────────────────
            if avg_events < 0.1:
                # This hour almost never has activity
                boost   += BOOST_QUIET_HOUR
                reasons.append(f"unusual_hour:{hour}:00_normally_empty")

            elif avg_events < 0.5:
                # Low activity hour
                boost   += BOOST_UNUSUAL_TIME
                reasons.append(f"low_activity_hour:{hour}:00")

            # ── Dwell time anomaly ────────────────────────────────────────────
            if avg_dwell > 0 and current_dwell > avg_dwell * 2.5:
                boost   += BOOST_LONG_DWELL
                reasons.append(
                    f"unusual_dwell:{current_dwell:.0f}s_vs_avg_{avg_dwell:.0f}s"
                )

        # ── New face boost ────────────────────────────────────────────────────
        if is_new_face:
            boost   += BOOST_FREQUENT_NEW
            reasons.append("new_unrecognized_face")

        return min(boost, 35), (reasons[0] if reasons else None)

    def get_baseline_summary(self) -> dict:
        """Return baseline stats for display in the app."""
        if not self._has_enough_data():
            return {
                "ready"      : False,
                "event_count": self._event_count,
                "needed"     : MIN_EVENTS_FOR_BASELINE,
                "message"    : f"Learning... {self._event_count}/{MIN_EVENTS_FOR_BASELINE} events recorded",
            }

        # Find quiet and busy hours
        quiet_hours = []
        busy_hours  = []

        for hour, stats in self._baseline.items():
            avg = stats.get('avg_events_per_day', 0)
            if avg < 0.2:
                quiet_hours.append(hour)
            elif avg > 1.0:
                busy_hours.append(hour)

        return {
            "ready"       : True,
            "event_count" : self._event_count,
            "quiet_hours" : sorted(quiet_hours),
            "busy_hours"  : sorted(busy_hours),
            "history_days": HISTORY_DAYS,
            "last_built"  : datetime.fromtimestamp(self._last_built).isoformat()
                            if self._last_built else None,
        }

    def reset(self):
        """Clear all historical data and start fresh."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictive_events")
        conn.commit()
        conn.close()
        self._baseline    = {}
        self._event_count = 0
        print("[NxV Predict] Baseline reset")

    # ── Internals ──────────────────────────────────────────────────────────────

    def _ensure_table(self):
        """Create the predictive events table if it doesn't exist."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictive_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at   TEXT NOT NULL,
                hour        INTEGER NOT NULL,
                day_of_week INTEGER NOT NULL,
                duration    REAL DEFAULT 0,
                had_person  INTEGER DEFAULT 1,
                score       INTEGER DEFAULT 0,
                escalation  TEXT DEFAULT 'NONE'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_hour ON predictive_events(hour)"
        )
        conn.commit()
        conn.close()

    def _build_baseline(self):
        """
        Build per-hour baseline from historical events.
        Called on startup and every 100 new events.
        """
        conn   = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(days=HISTORY_DAYS)).isoformat()

        rows = conn.execute("""
            SELECT hour, duration, had_person, score
            FROM predictive_events
            WHERE logged_at > ?
        """, (cutoff,)).fetchall()
        conn.close()

        self._event_count = len(rows)

        if not rows:
            self._baseline = {}
            return

        # Group by hour
        by_hour = defaultdict(list)
        for hour, duration, had_person, score in rows:
            by_hour[hour].append({
                'duration'  : duration,
                'had_person': had_person,
                'score'     : score,
            })

        # Days in window for rate calculation
        days = min(HISTORY_DAYS,
                   max(1, (datetime.now() -
                           datetime.fromisoformat(
                               self._get_oldest_event() or datetime.now().isoformat()
                           )).days + 1))

        baseline = {}
        for hour in range(24):
            events = by_hour.get(hour, [])
            if events:
                durations = [e['duration'] for e in events if e['duration'] > 0]
                baseline[hour] = {
                    'event_count'       : len(events),
                    'avg_events_per_day': len(events) / max(days, 1),
                    'avg_duration'      : float(np.mean(durations)) if durations else 0,
                    'max_duration'      : float(np.max(durations))  if durations else 0,
                    'avg_score'         : float(np.mean([e['score'] for e in events])),
                }
            else:
                baseline[hour] = {
                    'event_count'       : 0,
                    'avg_events_per_day': 0,
                    'avg_duration'      : 0,
                    'max_duration'      : 0,
                    'avg_score'         : 0,
                }

        self._baseline   = baseline
        self._last_built = time.time()
        print(f"[NxV Predict] Baseline built — "
              f"{self._event_count} events · {days} days")

    def _get_oldest_event(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            row  = conn.execute(
                "SELECT MIN(logged_at) FROM predictive_events"
            ).fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    def _has_enough_data(self) -> bool:
        return self._event_count >= MIN_EVENTS_FOR_BASELINE
