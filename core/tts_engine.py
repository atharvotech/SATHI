# import os
# import pygame
# import ChatTTS
# import soundfile as sf
# import uuid
# import asyncio
# import torch
# import re

# # ==========================================
# # ⚙️ INITIALIZATION (Pehli baar mein time lega)
# # ==========================================
# print("Loading ChatTTS... (Tera i5 thoda royega, wait kar)")
# chat = ChatTTS.Chat()
# chat.load(compile=False) 

# # Seed 422 standard aur saaf female voice ke liye hai
# torch.manual_seed(762)
# rand_spk = chat.sample_random_speaker()

# # ==========================================
# # 🎤 THE SPEAK FUNCTION (With Frequency Fix)
# # ==========================================
# async def sathi_speak(text):
#     if len(text.strip()) < 2: return
    
#     # 1. Emoji aur junk characters saaf karo
#     clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?\[\]\']', '', text)
#     print(f"\n[Processing]: {clean_text}")

#     audio_file = f"sathi_test_{uuid.uuid4().hex[:8]}.wav"
    
#     try:
#         # 2. Audio Generate karo
#         wavs = chat.infer(
#             [clean_text], 
#             use_decoder=True,
#             params_refine_text=ChatTTS.Chat.RefineTextParams(
#                 prompt='[oral_2][laugh_0][break_4]', # [break_4] speed slow karega
#             ),
#             params_infer_code=ChatTTS.Chat.InferCodeParams(
#                 spk_emb=rand_spk, 
#                 temperature=0.1, # Temperature kam karne se aawaz "stable" aur saaf aati hai
#             )
#         )
        
#         # 3. Audio extraction fix (tuple index out of range error ke liye)
#         audio_data = wavs[0]
#         if isinstance(audio_data, (list, tuple)):
#             audio_data = audio_data[0]
            
#         # 4. Save Audio (24000Hz)
#         sf.write(audio_file, audio_data, 24000)
        
#         # 5. Play Audio with CORRECT FREQUENCY (Isse aawaz saaf aayegi)
#         if pygame.mixer.get_init():
#             pygame.mixer.quit()
        
#         pygame.mixer.init(frequency=24000) 
#         pygame.mixer.music.load(audio_file)
#         pygame.mixer.music.play()
        
#         print("SATHI bol rahi hai... Sunn dhyan se!")
#         while pygame.mixer.music.get_busy():
#             await asyncio.sleep(0.1)
            
#         pygame.mixer.music.unload()
#         if os.path.exists(audio_file):
#             os.remove(audio_file)
            
#     except Exception as e:
#         print(f"\n[TTS Error]: {e}")

# # ==========================================
# # 🧪 TEST RUN (Yahan apni line likh de)
# # ==========================================
# async def main():
#     # Maine "Phonetic" Hinglish likhi hai taaki aawaz real lage
#     # 1. Manual pauses add karo (commas aur [uv_break] slow kar denge)
#     test_line = "Abbe Cuto... [uv_break] moojhey pehchaana [laugh] Main online aa gayee hoon bhaai. Ab so jaa... varna kal tera dimaag phat jayega  kya tum mere se baat karoge"
#     await sathi_speak(test_line)

# if __name__ == "__main__":
#     asyncio.run(main())



import edge_tts
import asyncio
import sounddevice as sd
import soundfile as sf
import io

# Best voices for Hinglish
HINDI_VOICE = "hi-IN-MadhurNeural"        # Hindi male - very natural
HINDI_FEMALE = "hi-IN-SwaraNeural"        # Hindi female
HINGLISH_VOICE = "en-IN-PrabhatNeural"    # Indian English male - great for Hinglish
HINGLISH_FEMALE = "en-IN-NeerjaNeural"    # Indian English female

async def sathi_speak(text, voice=HINGLISH_VOICE):
    print(f"Speaking: {text}")
    communicate = edge_tts.Communicate(text, voice)

    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    audio_io = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_io)
    sd.play(data, samplerate)
    sd.wait()

    sf.write("output.wav", data, samplerate)
    print("Saved to output.wav")


# Test Hinglish sentences
async def main():

    # Pure Hindi
    await speak_hinglish(
        "नमस्ते। मैं JARVIS हूं। आपका personal AI assistant। मैं हमेशा आपकी सेवा में हाजिर हूं।",
        voice=HINDI_VOICE
    )

    # Hinglish - mixed
    await speak_hinglish(
        "Sir aapka schedule check kar liya. Aaj aapke paas teen important meetings hain. Pehli meeting 10 baje hai. Kya aap ready hain.",
        voice=HINGLISH_VOICE
    )

    # Hinglish casual funny
    await speak_hinglish(
        "Arre sir, aap phir late ho gaye. Main kab se wait kar raha hoon. Chalo koi baat nahi, abhi bhi time hai. Let us get started.",
        voice=HINGLISH_VOICE
    )

    # Hinglish excited
    await speak_hinglish(
        "Wah sir wah. Aapne aaj kya kaam kiya. Bilkul fantastic. Main bahut impressed hoon. Seriously, you are on fire today.",
        voice=HINGLISH_VOICE
    )

# asyncio.run(main())