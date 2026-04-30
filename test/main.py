
import asyncio
import json
import os
from ollama import AsyncClient
# SATHI ka aawaz module import kar rahe hain
from core.tts_engine import sathi_speak

from core.tts_engine import sathi_speak
m = 'sathi-A'  # custom model name
# --- Helper Functions ---
def load_json(filepath, default_data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_data, f, indent=4)
        return default_data
    with open(filepath, 'r') as f:
        return json.load(f)

async def main():
    client = AsyncClient()
    
    config_path = 'config/config.json'
    memory_path = 'memory/processed_info.json'

    # Load Data
    config = load_json(config_path, {"bot": {"name": "SATHI", "gender": 1, "active_mode": "casual"}, "user": {"name": "Atharv"}, "modes_config": {}})
    info = load_json(memory_path, {"recent_events": [], "learned_facts": []})
    
    user_name = config.get('user', {}).get('name', 'unknown')
    user_dob = config.get('user', {}).get('dob', 'unknown')
    user_gender = config.get('user', {}).get('gender', 'unknown')
    hobbies = ", ".join(config.get('user', {}).get('hobbies', [])) or "none"

    bot_name = config.get('bot', {}).get('name', 'SATHI')
    bot_gender = config.get('bot', {}).get('gender', 1)  # 1 for female
    active_mode = config.get('bot', {}).get('active_mode', 'casual').lower()
    

    events_str = " ".join(info.get('recent_events', []))
    
    # Get specific parameters for the active mode
    mode_settings = config.get('modes_config', {}).get(active_mode, {})
    current_temp = mode_settings.get('temperature', 0.7)
    current_top_p = mode_settings.get('top_p', 0.9)
    current_top_k = mode_settings.get('top_k', 40)
    mode_instruction = mode_settings.get('instruction', 'Act as a friend.')
    # PROMPT OPTIMIZATION: Rulebook se data padhna
    with open('prompts/rulebook.txt', 'r', encoding='utf-8') as f:
        raw_prompt = f.read()
    
    # Variables inject karna (Bulletproof method)
    system_context = raw_prompt.replace("{bot_name}", bot_name) \
                               .replace("{bot_gender}", bot_gender) \
                               .replace("{user_name}", user_name) \
                               .replace("{active_mode}", active_mode.upper()) \
                               .replace("{mode_instruction}", mode_instruction) \
                               .replace("{events_str}", events_str)

    messages = [{'role': 'system', 'content': system_context}]
    # ---------------------------------------------------------
    
    print(f"\n[{bot_name} booting in {active_mode.upper()} mode... (Temp: {current_temp})]\n")
    
    # Proactive Start
    messages.append({'role': 'user', 'content': f"Start the chat casually. Call me {user_name} that my name. No greetings. Ask me what I am doing right now in Hinglish."})
    
    print(f"{bot_name}: ", end="")
    bot_reply = ""
    async for chunk in await client.chat(
        model=m, 
        messages=messages, 
        stream=True,
        options={
            'temperature': current_temp,
            'top_p': current_top_p,
            'repeat_penalty': 1.15
        }
    ):
        print(chunk['message']['content'], end='', flush=True)
        bot_reply += chunk['message']['content']
    print("\n")
    
    messages.append({'role': 'assistant', 'content': bot_reply})

            
    # Main Chat Loop
    while True:
        try:
            user_input = input(f"\n{user_name}: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{bot_name}: Chal theek hai bhai, nikal ab. Bye!")
                break
                
            messages.append({'role': 'user', 'content': user_input})
            
            print(f"{bot_name}: ", end="")
            bot_reply = ""
            
            # Request with dynamic parameters
            async for chunk in await client.chat(
                model=m, 
                messages=messages, 
                stream=True,
                options={
                    'temperature': current_temp,
                    'top_p': current_top_p,
                    'repeat_penalty': 1.15
                }
            ):
                print(chunk['message']['content'], end='', flush=True)
                bot_reply += chunk['message']['content']
            print("\n")
            # SATHI ka response aane ke baad print karo

# Aur phir usko aawaz do!
            await sathi_speak(bot_reply)
    # save to conversation history
            try:
                with open('memory/conversation_history.json', 'r') as f:
                    chat_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                chat_data = [] 
            
            chat_data.append({"user": user_input, "bot": bot_reply})
            
            with open('memory/conversation_history.json', 'w') as f:
                json.dump(chat_data, f, indent=4)
            
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n[System forcefully shutdown.] \n")
            break

if __name__ == "__main__":
    asyncio.run(main())