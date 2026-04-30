"""
JARVIS System Info Tools — DateTime, Apps, Clipboard, Battery
"""

import datetime
import subprocess
import os
import platform


def get_datetime() -> dict:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return {
        "status": "ok",
        "result": {
            "iso": now.isoformat(),
            "human": now.strftime("%A, %B %d, %Y at %I:%M %p"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
        }
    }


def get_active_window() -> dict:
    """Get the title of the currently focused window."""
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win:
            return {"status": "ok", "result": {"title": win.title, "x": win.left, "y": win.top, "width": win.width, "height": win.height}}
        return {"status": "ok", "result": {"title": "No active window"}}
    except Exception as e:
        return {"status": "error", "result": f"Could not get active window: {e}"}


def list_running_apps() -> dict:
    """List all visible running applications (Windows)."""
    try:
        import pygetwindow as gw
        windows = gw.getAllWindows()
        apps = []
        seen = set()
        for w in windows:
            title = w.title.strip()
            if title and title not in seen and w.visible:
                apps.append(title)
                seen.add(title)
        return {"status": "ok", "result": apps[:30]}  # Cap at 30
    except Exception as e:
        return {"status": "error", "result": f"Could not list apps: {e}"}


# Known apps map for Windows
APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "vscode": "code",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "spotify": "spotify.exe",
    "discord": "discord.exe",
}


def open_app(app_name: str) -> dict:
    """Open an application by common name."""
    key = app_name.lower().strip()
    cmd = APP_MAP.get(key, key)  # Fallback to raw name
    try:
        if cmd.startswith("ms-"):
            os.startfile(cmd)
        else:
            subprocess.Popen(cmd, shell=True)
        return {"status": "ok", "result": f"Opened {app_name}"}
    except Exception as e:
        return {"status": "error", "result": f"Could not open {app_name}: {e}"}


def get_clipboard() -> dict:
    """Get current clipboard text content."""
    try:
        import pyperclip
        text = pyperclip.paste()
        return {"status": "ok", "result": text[:2000]}  # Cap
    except Exception as e:
        return {"status": "error", "result": f"Clipboard read failed: {e}"}


def set_clipboard(text: str) -> dict:
    """Set clipboard content."""
    try:
        import pyperclip
        if len(text) > 5000:
            return {"status": "error", "result": "Text too long for clipboard (max 5000 chars)"}
        pyperclip.copy(text)
        return {"status": "ok", "result": "Clipboard updated"}
    except Exception as e:
        return {"status": "error", "result": f"Clipboard write failed: {e}"}


def get_battery_status() -> dict:
    """Get battery percentage and charging state."""
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery:
            return {"status": "ok", "result": {
                "percent": battery.percent,
                "charging": battery.power_plugged,
                "time_left_mins": int(battery.secsleft / 60) if battery.secsleft > 0 else None
            }}
        return {"status": "ok", "result": "No battery detected (desktop PC)"}
    except Exception as e:
        return {"status": "error", "result": f"Battery check failed: {e}"}


def get_system_info() -> dict:
    """Get basic system info — OS, CPU, RAM."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {"status": "ok", "result": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "processor": platform.processor()[:60],
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "ram_used_percent": mem.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.5),
        }}
    except Exception as e:
        return {"status": "error", "result": f"System info failed: {e}"}
