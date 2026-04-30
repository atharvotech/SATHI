"""
JARVIS — Agentic Computer Autopilot
Main server: serves Web UI + WebSocket for real-time chat + LLM inference + tool execution.
"""

import asyncio
import json
import os
import re
import datetime
import traceback

from aiohttp import web

from llama_cpp import Llama
from core.tts_engine import sathi_speak
from core.router import execute_actions, get_tool_catalog
from core.vision import find_text_on_screen, describe_screen
from memory.session_log import log_turn

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = './models/gemma-4-E4B-it-Q8_0.gguf'
MEMORY_PATH = 'memory/conversation_history.json'
RULEBOOK_PATH = 'prompts/rulebook.txt'
CONFIG_PATH = 'config/config.json'
UI_DIR = os.path.join(os.path.dirname(__file__), 'ui')
PORT = 8080

# ============================================================
# GLOBALS
# ============================================================
llm = None              # Llama model instance (loaded once)
chat_history = []       # Conversation history
system_prompt = ""      # Compiled system prompt
config = {}             # User config
active_ws_clients = set()  # Connected WebSocket clients

# ============================================================
# MEMORY
# ============================================================

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, 'r') as f:
            return json.load(f)
    except:
        return []

def save_memory(history):
    with open(MEMORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except:
        return {"bot": {"name": "SATHI"}, "user": {"name": "User"}, 
                "modes_config": {"casual": {"temperature": 0.6, "instruction": "Be casual."}}}

# ============================================================
# PROMPT BUILDING
# ============================================================

def compile_system_prompt(rulebook: str, cfg: dict) -> str:
    """Fill template variables in the rulebook."""
    import pyautogui
    screen_w, screen_h = pyautogui.size()
    
    bot = cfg.get("bot", {})
    user = cfg.get("user", {})
    mode = bot.get("active_mode", "casual")
    mode_config = cfg.get("modes_config", {}).get(mode, {})
    
    prompt = rulebook
    prompt = prompt.replace("{bot_name}", bot.get("name", "SATHI"))
    prompt = prompt.replace("{user_name}", user.get("name", "User"))
    prompt = prompt.replace("{active_mode}", mode)
    prompt = prompt.replace("{mode_instruction}", mode_config.get("instruction", "Be casual."))
    prompt = prompt.replace("{screen_w}", str(screen_w))
    prompt = prompt.replace("{screen_h}", str(screen_h))
    prompt = prompt.replace("{current_time}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    prompt = prompt.replace("{events_str}", "None")
    
    return prompt


def build_gemma_prompt(system_prompt: str, messages: list) -> str:
    """Build Gemma 4 formatted prompt with JSON response format."""
    prompt = ""
    
    # System prompt as first user turn
    prompt += f"<start_of_turn>user\n[System Instructions]\n{system_prompt}<end_of_turn>\n"
    prompt += '<start_of_turn>model\n{"speak": "Samajh gayi, SATHI hoon. JSON mein reply karungi.", "actions": []}<end_of_turn>\n'
    
    # Estimate tokens used by system prompt (~4 chars per token for English)
    system_tokens_est = len(prompt) // 3
    max_total_tokens = 3072  # Match n_ctx
    remaining = max_total_tokens - system_tokens_est - 300  # 300 reserved for generation
    
    # Build conversation history, trimming from oldest if too long
    history_parts = []
    recent = messages[-8:]  # Max 4 user + 4 assistant turns
    for msg in recent:
        role_tag = "user" if msg['role'] == 'user' else "model"
        # Truncate individual messages if too long
        content = msg['content'][:500]  # Max 500 chars per message
        part = f"<start_of_turn>{role_tag}\n{content}<end_of_turn>\n"
        history_parts.append(part)
    
    # Add history parts while within budget
    history_text = ""
    for part in history_parts:
        est_tokens = len(part) // 3
        if remaining - est_tokens < 0:
            break
        history_text += part
        remaining -= est_tokens
    
    prompt += history_text
    
    # Open model turn
    prompt += "<start_of_turn>model\n"
    
    print(f"[Prompt] ~{len(prompt)//3} tokens estimated", flush=True)
    return prompt

# ============================================================
# JSON RESPONSE PARSING
# ============================================================

def parse_llm_response(raw: str) -> dict:
    """
    Parse the LLM's raw text into a structured response.
    Handles: valid JSON, broken JSON, and plain text fallback.
    """
    raw = raw.strip()
    
    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "speak" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from surrounding text
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict) and "speak" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try to fix common issues: missing closing brace
    if raw.startswith('{') and not raw.endswith('}'):
        try:
            parsed = json.loads(raw + '}')
            if isinstance(parsed, dict) and "speak" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = json.loads(raw + ']}')
            if isinstance(parsed, dict) and "speak" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Fallback: treat entire response as speak text
    clean = raw.strip('{}[]"\'').strip()
    if not clean:
        clean = "Hmm... kuch samajh nahi aaya, fir se bol?"
    return {"speak": clean, "actions": []}

# ============================================================
# CORE PROCESSING
# ============================================================

async def process_message(user_input: str, ws: web.WebSocketResponse) -> dict:
    """Process a user message through the full agentic pipeline."""
    global chat_history, system_prompt
    
    # 1. Notify UI: thinking
    await ws.send_json({"type": "thinking"})
    
    # 2. Build prompt
    messages_copy = list(chat_history)
    messages_copy.append({"role": "user", "content": user_input})
    
    # Refresh time in system prompt
    sys_prompt = re.sub(
        r'Time: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
        f'Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        system_prompt
    )
    
    full_prompt = build_gemma_prompt(sys_prompt, messages_copy)
    
    # 3. Run LLM (in thread to not block event loop)
    def run_llm():
        output = llm(
            full_prompt,
            max_tokens=256,
            stop=["<start_of_turn>", "<end_of_turn>"],
            temperature=0.4,
            top_p=0.85,
        )
        return output["choices"][0]["text"].strip()
    
    raw_response = await asyncio.to_thread(run_llm)
    print(f"\n[LLM Raw]: {raw_response}")
    
    # 4. Parse JSON
    parsed = parse_llm_response(raw_response)
    speak_text = parsed.get("speak", "")
    actions = parsed.get("actions", [])
    think = parsed.get("think", "")
    
    if think:
        print(f"[Think]: {think}")
    
    # 5. Execute actions
    action_results = []
    if actions:
        for action in actions:
            tool_name = action.get("tool", "unknown")
            await ws.send_json({"type": "executing", "tool": tool_name})
            
            # Execute in thread (some tools use blocking I/O)
            result = await asyncio.to_thread(execute_actions, [action])
            action_results.extend(result)
            
            # Send individual tool result to UI
            for r in result:
                await ws.send_json({"type": "tool_result", "tool": r.get("tool", ""), "result": r})
    
    # 6. TTS — speak the response (in background, don't block)
    if speak_text:
        await ws.send_json({"type": "speaking"})
        try:
            await sathi_speak(speak_text)
        except Exception as e:
            print(f"[TTS Error]: {e}")
    
    # 7. Save to memory
    chat_history.append({"role": "user", "content": user_input})
    # Store the raw JSON as assistant content so model sees its own format
    chat_history.append({"role": "assistant", "content": raw_response})
    save_memory([{"user": m["content"], "bot": chat_history[i+1]["content"]} 
                  for i, m in enumerate(chat_history) if m["role"] == "user" and i+1 < len(chat_history)])
    
    # 8. Session log
    log_turn(user_input, speak_text, actions, action_results, think)
    
    # 9. Send final response to UI
    response_data = {
        "type": "response",
        "speak": speak_text,
        "think": think,
        "actions_results": [{"tool": r.get("tool", ""), "status": r.get("status", ""), 
                              "result": r.get("result", "")} for r in action_results],
    }
    await ws.send_json(response_data)
    
    return response_data

# ============================================================
# WEB SERVER & WEBSOCKET
# ============================================================

async def websocket_handler(request):
    """Handle WebSocket connections from the UI."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    active_ws_clients.add(ws)
    
    print(f"[WS] Client connected. Total: {len(active_ws_clients)}")
    
    # Send startup message
    await ws.send_json({
        "type": "startup",
        "speak": "JARVIS online. Bol Cuto, kya karna hai?"
    })
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("type") == "message":
                        content = data.get("content", "").strip()
                        if content:
                            await process_message(content, ws)
                except Exception as e:
                    traceback.print_exc()
                    await ws.send_json({"type": "error", "message": str(e)})
            elif msg.type == web.WSMsgType.ERROR:
                print(f"[WS] Error: {ws.exception()}")
    finally:
        active_ws_clients.discard(ws)
        print(f"[WS] Client disconnected. Total: {len(active_ws_clients)}")
    
    return ws


async def index_handler(request):
    """Serve the main index.html."""
    return web.FileResponse(os.path.join(UI_DIR, 'index.html'))


# ============================================================
# APP INITIALIZATION
# ============================================================

def setup():
    """Load model and config synchronously at startup."""
    global llm, system_prompt, config, chat_history
    
    # 1. Load config
    config = load_config()
    bot_name = config.get("bot", {}).get("name", "SATHI")
    print(f"\n{'='*50}", flush=True)
    print(f"  JARVIS — Computer Autopilot", flush=True)
    print(f"  Personality: {bot_name}", flush=True)
    print(f"{'='*50}\n", flush=True)
    
    # 2. Load LLM
    print("Loading Gemma 4 model into RAM... (this takes a minute)", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=3072,
        n_threads=6,
        n_batch=512,
        echo=False,
        verbose=False,
    )
    print("Model loaded!\n", flush=True)
    
    # 3. Load system prompt
    try:
        with open(RULEBOOK_PATH, 'r', encoding='utf-8') as f:
            rulebook = f.read()
        system_prompt = compile_system_prompt(rulebook, config)
    except FileNotFoundError:
        system_prompt = "You are SATHI, an intelligent computer autopilot. Respond in JSON format."
    
    # 4. Load memory
    old_memory = load_memory()
    for msg in old_memory[-6:]:
        if isinstance(msg, dict) and 'user' in msg and 'bot' in msg:
            chat_history.append({'role': 'user', 'content': msg['user']})
            chat_history.append({'role': 'assistant', 'content': msg['bot']})


def create_app() -> web.Application:
    """Create the aiohttp web application."""
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_static('/static/', path=UI_DIR, name='static')
    return app


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    setup()
    app = create_app()
    print(f"Starting server on http://localhost:{PORT}", flush=True)
    print(f"Open your browser to http://localhost:{PORT}\n", flush=True)
    web.run_app(app, host='localhost', port=PORT)