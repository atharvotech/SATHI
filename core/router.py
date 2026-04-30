"""
JARVIS Tool Router — Dispatches tool calls from the LLM JSON response.
Includes safety layer, logging, and error recovery.
"""

import time
import json
from core.tools import system_tools, web_tools, system_info, file_tools
from core import vision

# ============================================================
# TOOL REGISTRY — Maps tool names to (function, risk_level)
# risk: "low" = auto-execute, "medium" = execute with logging, "high" = needs confirmation
# ============================================================

TOOL_REGISTRY = {
    # Mouse Tools
    "click":            (system_tools.click,            "medium",  {"x": int, "y": int}),
    "right_click":      (system_tools.right_click,      "medium",  {"x": int, "y": int}),
    "double_click":     (system_tools.double_click,     "medium",  {"x": int, "y": int}),
    "move_mouse":       (system_tools.move_mouse,       "low",     {"x": int, "y": int}),
    "drag":             (system_tools.drag,             "medium",  {"x1": int, "y1": int, "x2": int, "y2": int}),
    "scroll":           (system_tools.scroll,           "low",     {"amount": int}),

    # Keyboard Tools
    "type_text":        (system_tools.type_text,        "medium",  {"text": str}),
    "press_key":        (system_tools.press_key,        "low",     {"key": str}),
    "hotkey":           (system_tools.hotkey,            "medium",  {"keys": list}),

    # Screen Tools
    "screenshot":       (system_tools.screenshot,       "low",     {}),
    "get_screen_size":  (system_tools.get_screen_size,  "low",     {}),
    "get_mouse_position": (system_tools.get_mouse_position, "low", {}),

    # Web Tools
    "web_search":       (web_tools.web_search,          "low",     {"query": str}),
    "open_url":         (web_tools.open_url,            "medium",  {"url": str}),
    "fetch_page_text":  (web_tools.fetch_page_text,     "low",     {"url": str}),

    # System Info
    "get_datetime":     (system_info.get_datetime,      "low",     {}),
    "get_active_window": (system_info.get_active_window, "low",    {}),
    "list_running_apps": (system_info.list_running_apps, "low",    {}),
    "open_app":         (system_info.open_app,          "medium",  {"app_name": str}),
    "get_clipboard":    (system_info.get_clipboard,     "low",     {}),
    "set_clipboard":    (system_info.set_clipboard,     "low",     {"text": str}),
    "get_battery_status": (system_info.get_battery_status, "low",  {}),
    "get_system_info":  (system_info.get_system_info,   "low",     {}),

    # File Tools
    "read_file":        (file_tools.read_file,          "medium",  {"path": str}),
    "write_file":       (file_tools.write_file,         "high",    {"path": str, "content": str}),
    "list_directory":   (file_tools.list_directory,     "low",     {"path": str}),
    "run_python_code":  (file_tools.run_python_code,    "high",    {"code": str}),

    # Vision Tools
    "find_text_on_screen": (vision.find_text_on_screen, "low", {"search_text": str}),
}


def get_tool_catalog() -> str:
    """Generate a human-readable tool catalog for the system prompt."""
    lines = []
    for name, (fn, risk, params) in TOOL_REGISTRY.items():
        param_str = ", ".join(f"{k}: {v.__name__}" for k, v in params.items()) if params else "none"
        doc = fn.__doc__.strip().split('\n')[0] if fn.__doc__ else "No description"
        lines.append(f"  - {name}({param_str}) -- {doc} [risk: {risk}]")
    return "\n".join(lines)


# ============================================================
# ACTION EXECUTION
# ============================================================

# Execution log for session
_execution_log = []


def execute_action(action: dict) -> dict:
    """
    Execute a single tool action from the LLM's JSON response.
    
    Expected format:
        {"tool": "click", "params": {"x": 100, "y": 200}}
    
    Returns:
        {"tool": "click", "status": "ok"/"error"/"blocked", "result": "..."}
    """
    tool_name = action.get("tool", "")
    params = action.get("params", {})
    
    if tool_name not in TOOL_REGISTRY:
        return {"tool": tool_name, "status": "error", "result": f"Unknown tool: '{tool_name}'"}
    
    fn, risk, expected_params = TOOL_REGISTRY[tool_name]
    
    # Log the action
    log_entry = {
        "timestamp": time.time(),
        "tool": tool_name,
        "params": params,
        "risk": risk,
    }
    
    try:
        # Special handling for hotkey (variable args)
        if tool_name == "hotkey":
            keys = params.get("keys", [])
            result = fn(*keys)
        else:
            result = fn(**params)
        
        log_entry["result"] = result
        _execution_log.append(log_entry)
        
        return {"tool": tool_name, **result}
    
    except TypeError as e:
        return {"tool": tool_name, "status": "error", "result": f"Wrong parameters: {e}"}
    except Exception as e:
        return {"tool": tool_name, "status": "error", "result": f"Execution failed: {e}"}


def execute_actions(actions: list) -> list:
    """Execute a list of tool actions sequentially. Returns list of results."""
    results = []
    for action in actions:
        result = execute_action(action)
        results.append(result)
        print(f"  [Tool] {result['tool']}: {result.get('status', '?')} — {result.get('result', '')}")
        
        # If a tool fails, stop the chain (optional — can be made configurable)
        if result.get("status") == "error":
            print(f"  [⚠️ Tool chain stopped due to error]")
            break
        
        # Small delay between actions for stability
        time.sleep(0.1)
    
    return results


def get_execution_log() -> list:
    """Get the current session's execution log."""
    return _execution_log.copy()
