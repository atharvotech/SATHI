<div align="center">

# ⚡ JARVIS — Agentic Computer Autopilot

**A fully local, voice-and-text-controlled AI that can operate your entire PC.**

*Powered by Gemma 4 (llama.cpp) · Hinglish personality · Real-time WebSocket UI*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Gemma%204-orange?style=for-the-badge)](https://huggingface.co/google/gemma-4)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?style=for-the-badge&logo=windows)](https://microsoft.com/windows)

</div>

---

## 🧠 What is JARVIS?

JARVIS (styled after the iconic AI from Iron Man) is a **fully offline, agentic computer autopilot** built around a locally-running Gemma 4 language model. It goes far beyond a simple chatbot — it can **see your screen**, **move your mouse**, **type text**, **open apps**, **search the web**, **read and write files**, and **execute Python code**, all driven by natural language commands.

The AI personality is called **SATHI** — a witty, sassy, Hinglish-speaking female digital best friend. She responds only in structured JSON, which the system parses and executes as real computer actions in real time.

### Key highlights

- 🏠 **Fully local inference** — no cloud API, no data sent anywhere. Runs on Gemma 4 via `llama-cpp-python`.
- 🤖 **Agentic tool execution** — LLM decides which tools to call; the router executes them, then feeds results back.
- 🖥️ **Full PC control** — mouse, keyboard, screenshots, app launching, OCR-based element finding.
- 🎙️ **Text-to-Speech** — responds out loud using Microsoft Edge TTS with an Indian-English neural voice.
- 🌐 **Web UI** — a dark, holographic dashboard served locally at `http://localhost:8080`.
- 💾 **Persistent memory** — conversation history survives restarts.
- 🔒 **Safety layers** — rate limiting, coordinate bounds checking, blocked dangerous key combos, path sandboxing.

---

## 🗂️ Project Structure

```
JARVIS/
│
├── main.py                   # Entry point — aiohttp server + WebSocket + LLM pipeline
│
├── core/                     # Core AI engine
│   ├── router.py             # Tool registry and dispatch (the "brain stem")
│   ├── tts_engine.py         # Text-to-Speech via Edge TTS
│   ├── vision.py             # Screen capture and OCR (pytesseract)
│   └── tools/
│       ├── __init__.py
│       ├── system_tools.py   # Mouse, keyboard, screenshot control (PyAutoGUI)
│       ├── system_info.py    # DateTime, clipboard, battery, running apps
│       ├── web_tools.py      # DuckDuckGo search, URL opener, page fetcher
│       └── file_tools.py     # Read/write files, list directories, run Python snippets
│
├── input/
│   └── voice.py              # Microphone input (SpeechRecognition — currently text fallback)
│
├── memory/
│   ├── session_log.py        # Per-turn logging to session_log.json
│   ├── conversation_history.json   # Persistent chat memory (auto-generated)
│   ├── processed_info.json   # Additional persistent state
│   └── system_logs.json      # System-level event log
│
├── prompts/
│   └── rulebook.txt          # SATHI's system prompt — personality, tools, format rules
│
├── config/
│   ├── config.json           # Bot name, user profile, LLM mode configs
│   └── user_config.py        # (Reserved for future Python-level config)
│
├── ui/
│   ├── index.html            # Single-page Web UI
│   ├── style.css             # Dark holographic theme (Inter + JetBrains Mono)
│   └── app.js                # WebSocket client, canvas avatar, chat rendering
│
├── atharTTS/                 # Custom TTS submodule (RVC / atharTTS research)
│   └── ...                   # See atharTTS/README.md for full documentation
│
├── models/                   # Local GGUF model files (not committed to Git)
│   └── gemma-4-E4B-it-Q8_0.gguf
│
├── Modelfile.txt             # Ollama Modelfile for SATHI persona (legacy / alternate backend)
├── requirements.txt          # Python dependencies
└── output.wav                # Last TTS audio output (auto-overwritten each turn)
```

---

## ⚙️ Architecture & Data Flow

```
User types/speaks
       │
       ▼
  [ Web UI (index.html / app.js) ]
       │  WebSocket message: {"type":"message","content":"..."}
       ▼
  [ main.py — WebSocket Handler ]
       │
       ├─1─ Notify UI: {"type":"thinking"}
       │
       ├─2─ Build Gemma prompt
       │     └── compile_system_prompt()  ← rulebook.txt + config.json
       │     └── build_gemma_prompt()     ← system prompt + last 8 chat turns
       │
       ├─3─ LLM Inference (asyncio.to_thread)
       │     └── llama_cpp.Llama()        ← gemma-4-E4B-it-Q8_0.gguf
       │     └── Returns raw JSON string
       │
       ├─4─ Parse JSON response
       │     └── parse_llm_response()
       │         {"speak": "...", "actions": [...], "think": "..."}
       │
       ├─5─ Execute tool actions (asyncio.to_thread)
       │     └── core/router.py → execute_actions()
       │         ├── system_tools  (click, type, hotkey, screenshot...)
       │         ├── web_tools     (search, open_url, fetch_page...)
       │         ├── system_info   (datetime, apps, clipboard, battery...)
       │         ├── file_tools    (read, write, list_dir, run_python...)
       │         └── vision        (find_text_on_screen)
       │
       ├─6─ TTS — sathi_speak()
       │     └── edge_tts.Communicate()  → sounddevice playback
       │     └── Saves output.wav
       │
       ├─7─ Save to memory
       │     └── memory/conversation_history.json
       │     └── memory/session_log.py → session_log.json
       │
       └─8─ Send final response to UI
             └── {"type":"response","speak":"...","actions_results":[...]}
```

---

## 🛠️ Tool Registry

All tools SATHI can use are registered in `core/router.py`. Each entry specifies the function, a risk level, and expected parameter types.

### Mouse & Keyboard Control
| Tool | Parameters | Risk | Description |
|------|-----------|------|-------------|
| `click` | `x: int, y: int` | medium | Left-click at screen coordinates |
| `right_click` | `x: int, y: int` | medium | Right-click at coordinates |
| `double_click` | `x: int, y: int` | medium | Double-click at coordinates |
| `move_mouse` | `x: int, y: int` | low | Move cursor without clicking |
| `drag` | `x1,y1,x2,y2: int` | medium | Click-drag from one point to another |
| `scroll` | `amount: int` | low | Scroll (+up / -down) |
| `type_text` | `text: str` | medium | Type a string (max 500 chars) |
| `press_key` | `key: str` | low | Press a single named key |
| `hotkey` | `keys: list` | medium | Press a key combo (e.g. `ctrl+c`) |

### Screen & Vision
| Tool | Parameters | Risk | Description |
|------|-----------|------|-------------|
| `screenshot` | — | low | Capture and save the full screen |
| `get_screen_size` | — | low | Return screen resolution |
| `get_mouse_position` | — | low | Return current cursor XY |
| `find_text_on_screen` | `search_text: str` | low | OCR scan → return element coordinates |

### Web
| Tool | Parameters | Risk | Description |
|------|-----------|------|-------------|
| `web_search` | `query: str` | low | DuckDuckGo search (no API key) |
| `open_url` | `url: str` | medium | Open URL in default browser |
| `fetch_page_text` | `url: str` | low | Fetch page, strip HTML, return text |

### System Info
| Tool | Parameters | Risk | Description |
|------|-----------|------|-------------|
| `get_datetime` | — | low | Current date/time in multiple formats |
| `get_active_window` | — | low | Title + bounds of focused window |
| `list_running_apps` | — | low | All visible open windows |
| `open_app` | `app_name: str` | medium | Launch an app by name |
| `get_clipboard` | — | low | Read clipboard text |
| `set_clipboard` | `text: str` | low | Write to clipboard |
| `get_battery_status` | — | low | Battery % and charging state |
| `get_system_info` | — | low | OS, CPU, RAM usage |

### File System
| Tool | Parameters | Risk | Description |
|------|-----------|------|-------------|
| `read_file` | `path: str` | medium | Read file (max 1MB, user home only) |
| `write_file` | `path, content: str` | **high** | Write file (max 10KB, no .exe/.bat etc.) |
| `list_directory` | `path: str` | low | List directory contents (max 50 items) |
| `run_python_code` | `code: str` | **high** | Execute Python snippet in a sandbox |

> **Safety note:** `write_file` and `run_python_code` are classified as **high-risk**. The sandbox blocks `import os`, `import sys`, `import subprocess`, `shutil`, `eval()`, `exec()`, and `open()` to prevent destructive operations.

---

## 🤖 LLM & Prompt System

### Model
- **Model:** `gemma-4-E4B-it-Q8_0.gguf` (Gemma 4 4B, 8-bit quantized)
- **Backend:** `llama-cpp-python` (pure CPU inference, no GPU required)
- **Context window:** 3072 tokens
- **Threads:** 6 CPU threads, batch size 512

### Prompt Format
JARVIS uses Gemma's native chat template:

```
<start_of_turn>user
[System Instructions]
{rulebook contents with variables filled in}
<end_of_turn>
<start_of_turn>model
{"speak": "Samajh gayi, SATHI hoon. JSON mein reply karungi.", "actions": []}
<end_of_turn>
<start_of_turn>user
{user message}
<end_of_turn>
<start_of_turn>model
← LLM generates here
```

The system prompt is assembled from `prompts/rulebook.txt` at startup, with the following template variables injected at runtime:

| Variable | Source |
|----------|--------|
| `{bot_name}` | `config.json → bot.name` |
| `{user_name}` | `config.json → user.name` |
| `{active_mode}` | `config.json → bot.active_mode` |
| `{mode_instruction}` | `config.json → modes_config[mode].instruction` |
| `{screen_w}` / `{screen_h}` | `pyautogui.size()` at startup |
| `{current_time}` | `datetime.now()` — refreshed each message |

### Response Format
SATHI is instructed to **always** respond in this exact JSON shape:

```json
{
  "speak": "what to say aloud (1-3 sentences, Hinglish casual)",
  "actions": [
    {"tool": "click", "params": {"x": 100, "y": 200}},
    {"tool": "type_text", "params": {"text": "hello"}}
  ],
  "think": "(optional internal reasoning, not spoken)"
}
```

The parser in `parse_llm_response()` handles malformed JSON gracefully with several fallback strategies: direct parse → regex extract → partial JSON repair → plain-text fallback.

### Personality Modes
Three personality modes are defined in `config/config.json`, switchable without restarting:

| Mode | Temp | Style |
|------|------|-------|
| `casual` | 0.6 | Direct best-friend, practical, witty |
| `creative` | 0.85 | Imaginative, expressive, energetic |
| `companion` | 0.7 | Caring, empathetic, supportive |

---

## 🎙️ Text-to-Speech

**Current TTS:** Microsoft Edge TTS via the `edge-tts` library (internet required for synthesis).

```python
HINGLISH_VOICE = "en-IN-PrabhatNeural"   # Indian English male (default)
HINDI_FEMALE   = "hi-IN-SwaraNeural"     # Hindi female
```

The `sathi_speak()` function:
1. Streams audio chunks from the Edge TTS API
2. Decodes the MP3 stream using `soundfile`
3. Plays audio immediately via `sounddevice`
4. Saves the output to `output.wav` for reference

**Legacy TTS (commented out):** The `tts_engine.py` file retains the original ChatTTS implementation (GPU-based, fully offline) for future reactivation.

---

## 🖥️ Web UI

The dashboard is a single-page app served at `http://localhost:8080` by the aiohttp server.

### Layout
```
┌─────────────────────────────────────────────────────┐
│  ⚡ JARVIS     Computer Autopilot          ● Online  │
├──────────────────────┬──────────────────────────────┤
│   Avatar Panel       │   Chat Panel                 │
│                      │                              │
│  [Canvas Avatar]     │  [Message history]           │
│  Animated glow       │                              │
│                      │  ┌──────────────────────┐   │
│  SATHI               │  │ Type a message...    │ ▶ │
│  Idle / Thinking     │  └──────────────────────┘   │
│                      │  Enter to send              │
│  Activity Feed       │                              │
│  [Tool results]      │                              │
└──────────────────────┴──────────────────────────────┘
```

### WebSocket Protocol
The UI communicates with the server via WebSocket at `ws://localhost:8080/ws`.

**Client → Server:**
```json
{"type": "message", "content": "open chrome and search for cats"}
```

**Server → Client message types:**
| Type | Payload | Meaning |
|------|---------|---------|
| `startup` | `speak` | Initial greeting when connected |
| `thinking` | — | LLM inference started |
| `executing` | `tool` | A specific tool is being called |
| `tool_result` | `tool, result` | Tool execution finished |
| `speaking` | — | TTS playback started |
| `response` | `speak, think, actions_results` | Full final response |
| `error` | `message` | An error occurred |

---

## 💾 Memory & Logging

### Conversation History
`memory/conversation_history.json` — Persistent list of `{user, bot}` turn pairs. The last **6 turns** are loaded into the prompt context on startup. A maximum of the last **8 turns** are kept in the active window (4 user + 4 assistant) to respect the 3072-token context limit. Each individual message is capped at **500 characters**.

### Session Log
`memory/session_log.json` — Detailed per-turn log kept by `session_log.py`. Each entry records:
```json
{
  "timestamp": "2026-04-28T14:23:01.123456",
  "user": "open notepad",
  "think": "(optional reasoning)",
  "speak": "Sure yaar, opening Notepad abhi!",
  "actions": [{"tool": "open_app", "params": {"app_name": "notepad"}}],
  "action_results": [{"tool": "open_app", "status": "ok", "result": "Opened notepad"}]
}
```
Capped at **200 entries** (oldest auto-pruned).

---

## 🔒 Safety Mechanisms

| Layer | Mechanism |
|-------|-----------|
| **Mouse/Keyboard** | PyAutoGUI `FAILSAFE=True` — move mouse to top-left corner to abort immediately |
| **Rate limiting** | Max 5 tool actions per second (thread-safe) |
| **Coordinate validation** | All click/move coordinates are checked against screen bounds |
| **Hotkey blocking** | `Alt+F4` and `Ctrl+Alt+Delete` are permanently blocked |
| **File sandboxing** | `read_file`/`write_file`/`list_directory` restricted to user home directory |
| **File type blocking** | Cannot write `.exe`, `.bat`, `.cmd`, `.ps1`, `.vbs`, `.reg`, `.msi`, `.dll`, `.sys` |
| **Code sandbox** | `run_python_code` blocks `os`, `sys`, `subprocess`, `shutil`, `eval`, `exec`, `open` |
| **Code timeout** | Python snippet execution is killed after **10 seconds** |
| **Prompt constraints** | Rulebook explicitly forbids deleting system files and typing passwords |
| **Token budget** | Prompt is actively trimmed to stay under 3072 tokens |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+ (tested on 3.11)
- Windows 10/11 (some tools use Windows-specific APIs)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for `find_text_on_screen`)
- At least 8GB RAM (12GB+ recommended for comfortable inference)

### 1. Clone the repository
```bash
git clone https://github.com/atharvotech/SATHI.git
cd SATHI
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `llama-cpp-python` may require a C++ build toolchain. If installation fails, try the pre-built wheel:
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> ```

### 4. Download the model
Download `gemma-4-E4B-it-Q8_0.gguf` and place it in the `models/` directory:
```
models/
└── gemma-4-E4B-it-Q8_0.gguf   (~4.5 GB)
```

You can find it on [Hugging Face](https://huggingface.co/models?search=gemma+4+gguf).

### 5. Configure your profile
Edit `config/config.json`:
```json
{
  "bot": {
    "name": "SATHI",
    "active_mode": "casual"
  },
  "user": {
    "name": "YourName"
  }
}
```

### 6. Run JARVIS
```bash
python main.py
```

Open your browser to **http://localhost:8080**. The model loads in ~30–60 seconds on first start.

---

## 💬 Usage Examples

Once the UI is open, you can type commands like:

```
"Open Chrome and search for today's weather"
→ Actions: open_app("chrome") → web_search("today's weather") → type_text(...)

"What time is it?"
→ Actions: get_datetime()
→ Speaks: "Yaar, abhi 3:45 PM hai. Kuch kaam karna hai?"

"Take a screenshot"
→ Actions: screenshot()
→ Saves to memory/latest_screenshot.png

"What apps are open?"
→ Actions: list_running_apps()

"Type hello world in Notepad"
→ Workflow: find_text_on_screen("Notepad") → click(x,y) → type_text("hello world")

"What's my battery at?"
→ Actions: get_battery_status()

"Read the file at C:/Users/me/notes.txt"
→ Actions: read_file("C:/Users/me/notes.txt")
```

---

## 📁 `atharTTS` Submodule

The `atharTTS/` directory is a Git submodule containing an independent, advanced TTS system with support for:
- **RVC (Retrieval-based Voice Conversion)** — clone any voice
- **CLI**, **Web UI**, **Discord bot**, **Telegram bot** integrations
- Multiple engine backends (ChatTTS, Edge TTS, custom)

See [`atharTTS/README.md`](atharTTS/README.md) for full documentation. It is currently not active in the main pipeline (Edge TTS is used instead) but can be integrated by replacing the `sathi_speak()` function in `core/tts_engine.py`.

---

## 📦 Ollama / `Modelfile.txt`

`Modelfile.txt` is a legacy Ollama model definition file for loading SATHI's personality via the Ollama runtime using `Qwen2.5-3B`. It is **not used** by the current `main.py` pipeline (which uses `llama-cpp-python` directly) but is preserved as an alternative deployment option.

To use with Ollama:
```bash
ollama create sathi -f Modelfile.txt
ollama run sathi
```

---

## 🧩 Extending JARVIS

### Adding a new tool

1. **Write the function** in one of the `core/tools/` files (or create a new file):
```python
def my_new_tool(param1: str) -> dict:
    """Brief description of what this tool does."""
    try:
        result = do_something(param1)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "result": str(e)}
```

2. **Register it** in `core/router.py`:
```python
TOOL_REGISTRY = {
    ...
    "my_new_tool": (my_tools.my_new_tool, "low", {"param1": str}),
}
```

3. **Add it to the rulebook** in `prompts/rulebook.txt` under the appropriate TOOLS section so the LLM knows it exists.

### Swapping the LLM
The model loading happens in `main.py → setup()`. To use a different GGUF model, simply change:
```python
MODEL_PATH = './models/your-model.gguf'
```
And adjust `n_ctx`, `n_threads`, `n_batch` in the `Llama()` constructor accordingly.

---

## 🐛 Known Limitations

| Issue | Status |
|-------|--------|
| Voice input (microphone) is disabled — `input/voice.py` falls back to text `input()` | In development |
| `find_text_on_screen` requires Tesseract OCR installed system-wide | Optional install |
| Edge TTS requires an internet connection for speech synthesis | By design |
| Non-ASCII text typing (`type_text`) may fail for some Unicode characters with PyAutoGUI | Partial fix applied |
| `run_python_code` sandbox does not prevent all infinite loops (thread join timeout = 10s) | Acceptable |
| Token budget is estimated by character count (`// 3`), not exact tokenization | Works in practice |

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built with ❤️ by [Atharv](https://github.com/atharvotech) · "Bol Cuto, kya karna hai?"*

</div>