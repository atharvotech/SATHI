"""
JARVIS Session Logger — Logs every interaction with timestamps.
"""

import json
import os
import datetime

SESSION_LOG_PATH = os.path.join(os.path.dirname(__file__), 'session_log.json')


def _load_log():
    if not os.path.exists(SESSION_LOG_PATH):
        return []
    try:
        with open(SESSION_LOG_PATH, 'r') as f:
            return json.load(f)
    except:
        return []


def _save_log(log):
    with open(SESSION_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, default=str)


def log_turn(user_input: str, speak: str, actions: list, action_results: list, think: str = ""):
    """Log a single conversation turn."""
    log = _load_log()
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user_input,
        "think": think,
        "speak": speak,
        "actions": actions,
        "action_results": action_results,
    }
    log.append(entry)
    
    # Keep last 200 entries
    if len(log) > 200:
        log = log[-200:]
    
    _save_log(log)


def get_recent_log(n: int = 10) -> list:
    """Get the last N log entries."""
    log = _load_log()
    return log[-n:]
