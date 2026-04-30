"""
JARVIS System Tools — Mouse, Keyboard & Screen Control
Safety-wrapped PyAutoGUI functions for the agentic autopilot.
"""

import pyautogui
import time
import threading

# ============================================================
# SAFETY CONFIGURATION
# ============================================================
pyautogui.FAILSAFE = True        # Move mouse to top-left corner to abort
pyautogui.PAUSE = 0.15           # Small delay between actions for stability

# Rate limiting — max actions per second
_action_timestamps = []
MAX_ACTIONS_PER_SECOND = 5
_lock = threading.Lock()

# Screen bounds
SCREEN_W, SCREEN_H = pyautogui.size()

def _rate_check():
    """Block if too many actions per second."""
    with _lock:
        now = time.time()
        _action_timestamps.append(now)
        # Keep only last second
        while _action_timestamps and _action_timestamps[0] < now - 1.0:
            _action_timestamps.pop(0)
        if len(_action_timestamps) > MAX_ACTIONS_PER_SECOND:
            time.sleep(0.3)

def _validate_coords(x, y):
    """Ensure coordinates are within screen bounds."""
    if not (0 <= x <= SCREEN_W and 0 <= y <= SCREEN_H):
        raise ValueError(f"Coordinates ({x}, {y}) out of screen bounds ({SCREEN_W}x{SCREEN_H})")
    return int(x), int(y)


# ============================================================
# MOUSE TOOLS
# ============================================================

def click(x: int, y: int) -> dict:
    """Left click at the given screen coordinates."""
    _rate_check()
    x, y = _validate_coords(x, y)
    pyautogui.click(x, y)
    return {"status": "ok", "result": f"Clicked at ({x}, {y})"}


def right_click(x: int, y: int) -> dict:
    """Right click at the given screen coordinates."""
    _rate_check()
    x, y = _validate_coords(x, y)
    pyautogui.rightClick(x, y)
    return {"status": "ok", "result": f"Right-clicked at ({x}, {y})"}


def double_click(x: int, y: int) -> dict:
    """Double click at the given screen coordinates."""
    _rate_check()
    x, y = _validate_coords(x, y)
    pyautogui.doubleClick(x, y)
    return {"status": "ok", "result": f"Double-clicked at ({x}, {y})"}


def move_mouse(x: int, y: int) -> dict:
    """Move mouse to the given coordinates without clicking."""
    _rate_check()
    x, y = _validate_coords(x, y)
    pyautogui.moveTo(x, y, duration=0.25)
    return {"status": "ok", "result": f"Mouse moved to ({x}, {y})"}


def drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> dict:
    """Drag from (x1,y1) to (x2,y2)."""
    _rate_check()
    x1, y1 = _validate_coords(x1, y1)
    x2, y2 = _validate_coords(x2, y2)
    duration = max(0.2, min(duration, 5.0))  # Clamp between 0.2s and 5s
    pyautogui.moveTo(x1, y1, duration=0.15)
    pyautogui.drag(x2 - x1, y2 - y1, duration=duration)
    return {"status": "ok", "result": f"Dragged from ({x1},{y1}) to ({x2},{y2})"}


def scroll(amount: int, x: int = None, y: int = None) -> dict:
    """Scroll at the current or given position. Positive = up, negative = down."""
    _rate_check()
    amount = max(-20, min(amount, 20))  # Clamp scroll amount
    if x is not None and y is not None:
        x, y = _validate_coords(x, y)
        pyautogui.scroll(amount, x, y)
    else:
        pyautogui.scroll(amount)
    return {"status": "ok", "result": f"Scrolled {amount} clicks"}


# ============================================================
# KEYBOARD TOOLS
# ============================================================

def type_text(text: str) -> dict:
    """Type the given text string. Max 500 chars for safety."""
    _rate_check()
    if len(text) > 500:
        return {"status": "error", "result": "Text too long (max 500 chars). Break into smaller chunks."}
    pyautogui.typewrite(text, interval=0.02) if text.isascii() else pyautogui.write(text)
    return {"status": "ok", "result": f"Typed {len(text)} characters"}


def press_key(key: str) -> dict:
    """Press a single key (enter, tab, escape, space, backspace, delete, etc.)."""
    _rate_check()
    ALLOWED_KEYS = {
        'enter', 'tab', 'escape', 'space', 'backspace', 'delete',
        'up', 'down', 'left', 'right', 'home', 'end', 'pageup', 'pagedown',
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
        'capslock', 'numlock', 'printscreen', 'insert', 'pause',
        'volumeup', 'volumedown', 'volumemute',
    }
    key = key.lower().strip()
    if key not in ALLOWED_KEYS:
        return {"status": "error", "result": f"Key '{key}' not in allowed list. Use hotkey() for combos."}
    pyautogui.press(key)
    return {"status": "ok", "result": f"Pressed '{key}'"}


def hotkey(*keys) -> dict:
    """Press a key combination (e.g., 'ctrl', 'c' for Ctrl+C)."""
    _rate_check()
    BLOCKED_COMBOS = [
        ('alt', 'f4'),         # Close window — too risky
        ('ctrl', 'alt', 'delete'),
    ]
    key_tuple = tuple(k.lower().strip() for k in keys)
    if key_tuple in BLOCKED_COMBOS:
        return {"status": "blocked", "result": f"Hotkey {'+'.join(keys)} is blocked for safety."}
    pyautogui.hotkey(*key_tuple)
    return {"status": "ok", "result": f"Pressed {'+'.join(keys)}"}


# ============================================================
# SCREEN INFO
# ============================================================

def get_screen_size() -> dict:
    """Get the screen resolution."""
    w, h = pyautogui.size()
    return {"status": "ok", "result": {"width": w, "height": h}}


def get_mouse_position() -> dict:
    """Get current mouse position."""
    x, y = pyautogui.position()
    return {"status": "ok", "result": {"x": x, "y": y}}


def screenshot() -> dict:
    """Take a screenshot and save it. Returns the file path."""
    import os
    from PIL import Image
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'memory', 'latest_screenshot.png')
    path = os.path.abspath(path)
    img = pyautogui.screenshot()
    img.save(path)
    return {"status": "ok", "result": f"Screenshot saved to {path}", "image_path": path}
