"""
JARVIS File Tools — Read, Write, List directories
All paths are sandboxed to user home directory for safety.
"""

import os
import sys
import io
import contextlib

# Safety: only allow access within these root directories
ALLOWED_ROOTS = [
    os.path.expanduser("~"),  # User home
]


def _is_safe_path(path: str) -> bool:
    """Check if the path is within allowed directories."""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(os.path.abspath(root)) for root in ALLOWED_ROOTS)


def read_file(path: str, max_chars: int = 5000) -> dict:
    """Read a text file's contents. Capped at max_chars."""
    try:
        if not _is_safe_path(path):
            return {"status": "blocked", "result": f"Path '{path}' is outside allowed directories."}
        if not os.path.exists(path):
            return {"status": "error", "result": f"File not found: {path}"}
        if os.path.getsize(path) > 1_000_000:  # 1MB limit
            return {"status": "error", "result": "File too large (max 1MB)"}
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(max_chars)
        return {"status": "ok", "result": content}
    except Exception as e:
        return {"status": "error", "result": f"Read failed: {e}"}


def write_file(path: str, content: str) -> dict:
    """Write text content to a file. Max 10KB."""
    try:
        if not _is_safe_path(path):
            return {"status": "blocked", "result": f"Path '{path}' is outside allowed directories."}
        if len(content) > 10_000:
            return {"status": "error", "result": "Content too large (max 10KB)"}
        
        # Blocked extensions
        _, ext = os.path.splitext(path)
        BLOCKED_EXTS = {'.exe', '.bat', '.cmd', '.ps1', '.vbs', '.reg', '.msi', '.dll', '.sys'}
        if ext.lower() in BLOCKED_EXTS:
            return {"status": "blocked", "result": f"Cannot write {ext} files for safety."}
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "ok", "result": f"Written {len(content)} chars to {path}"}
    except Exception as e:
        return {"status": "error", "result": f"Write failed: {e}"}


def list_directory(path: str) -> dict:
    """List files and folders in a directory."""
    try:
        if not _is_safe_path(path):
            return {"status": "blocked", "result": f"Path '{path}' is outside allowed directories."}
        if not os.path.isdir(path):
            return {"status": "error", "result": f"Not a directory: {path}"}
        entries = []
        for entry in os.scandir(path):
            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else None
            })
            if len(entries) >= 50:
                break
        return {"status": "ok", "result": entries}
    except Exception as e:
        return {"status": "error", "result": f"List failed: {e}"}


def run_python_code(code: str) -> dict:
    """Run a small Python snippet in a sandboxed exec. Timeout: 10s, stdout captured."""
    import threading

    if len(code) > 3000:
        return {"status": "error", "result": "Code too long (max 3000 chars)"}
    
    # Block dangerous imports/calls
    BLOCKED_PATTERNS = ['import os', 'import sys', 'import subprocess', 'os.system', 
                        'subprocess.', 'shutil.rmtree', '__import__', 'eval(', 'exec(',
                        'open(', 'import shutil']
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return {"status": "blocked", "result": f"Blocked: code contains '{pattern}' which is not allowed in sandbox."}

    output_capture = io.StringIO()
    result = {"status": "ok", "result": ""}
    error_occurred = threading.Event()

    def _exec():
        try:
            with contextlib.redirect_stdout(output_capture):
                exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                              "str": str, "int": int, "float": float,
                                              "list": list, "dict": dict, "tuple": tuple,
                                              "sorted": sorted, "sum": sum, "min": min, 
                                              "max": max, "abs": abs, "round": round,
                                              "enumerate": enumerate, "zip": zip, "map": map}})
            result["result"] = output_capture.getvalue()[:2000]
        except Exception as e:
            result["status"] = "error"
            result["result"] = str(e)
            error_occurred.set()

    thread = threading.Thread(target=_exec)
    thread.start()
    thread.join(timeout=10)
    
    if thread.is_alive():
        return {"status": "error", "result": "Code execution timed out (10s limit)"}
    
    return result
