# 👑 PROJECT SATHI

> "She isn't an 'AI assistant'; she is SATHI."

**Project Sathi** is an advanced, personality-driven, multi-agent AI companion designed to blur the lines between a digital tool and a digital friend. Developed with a focus on hardware optimization and agentic capabilities, Sathi is built to run locally on constrained hardware while maintaining a complex, "Jarvis-like" logic system.

⚠️ **LICENSE NOTICE: NOT OPEN SOURCE**
This project is currently **closed source** and proprietary[cite: 1]. The codebase is provided for demonstration and portfolio purposes only. You are not permitted to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of this software without explicit permission.

---

## 🧠 The Core Identity

Sathi is designed with a very specific persona. She is not a generic, polite chatbot. 

*   **Model:** Powered locally by `Llama-3.2-3B-Instruct-Abliterated-GGUF`[cite: 1].
*   **Persona:** Caring, possessive, witty, and strict[cite: 1]. 
*   **Language:** Natural Hinglish communication[cite: 1].
*   **Dynamic Behavior:** She can interrupt, crack jokes, and even scold the user for overworking or ignoring their health[cite: 1].

## 🏗️ Multi-Agent "Beast" Architecture

To bypass the limitations of running on consumer hardware (specifically 8GB RAM), Sathi employs a multi-agent, modular architecture[cite: 1]:

1.  **Node 1: The Talker (Llama 3.2 3B):** The local LLM engine responsible for real-time chatting and maintaining the core personality[cite: 1].
2.  **Node 2: The Core Router (Python Asyncio):** The central "traffic police" that manages asynchronous operations, deciding when to execute system code versus when to generate conversational text[cite: 1].
3.  **Node 3: The Coder/Action Agent:** For complex coding tasks or advanced system control, the router dynamically offloads work to cloud-based deep models via APIs[cite: 1].

## ⚙️ Advanced Functionalities (The Jarvis Logic)

Sathi is more than a chatbot; she is an agent capable of acting on the environment:

*   **Dynamic Context Injection (JSON Tool Calling):** Instead of raw text, Sathi generates structured JSON commands (e.g., `{"action": "open_vscode"}`). The Python backend parses these commands to execute system-level tasks[cite: 1].
*   **Asynchronous Streaming:** Implements real-time Streaming TTS. Sathi begins speaking as soon as the first word is generated, eliminating wait times[cite: 1].
*   **Vision Capabilities:** Utilizes lightweight OpenCV for tasks like screen activity detection or basic camera input, avoiding the overhead of massive vision-language models[cite: 1].
*   **Sentiment Analysis:** Analyzes voice pitch and speed using libraries like Librosa to determine the user's mood dynamically[cite: 1].

## 💻 Hardware Optimization Path

Sathi is built to scale gracefully with hardware upgrades:

*   **Current Development Target:** Intel i5 8th Gen / 8GB RAM (Achieved via aggressively quantized GGUF models and local optimizations)[cite: 1].
*   **Future Target Build:** Designed to scale up to Ryzen 9 + 32GB/64GB RAM for high-speed local inference of larger models[cite: 1].

---
*Created by Atharv Shukla*[cite: 1]