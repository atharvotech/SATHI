import torch
import ChatTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import re

print("Loading ChatTTS model...")
chat = ChatTTS.Chat()
chat.load(compile=False)
print("Model loaded!")

rand_spk = chat.sample_random_speaker()

def speak(text, emotion="normal"):
    text = re.sub(r'[!?,;:"\'\(\){}]', '', text)
    text = text.strip()

    if emotion == "happy":
        refine_prompt = '[oral_5][laugh_2][break_4]'
    elif emotion == "excited":
        refine_prompt = '[oral_7][laugh_2][break_3]'
    elif emotion == "sad":
        refine_prompt = '[oral_4][laugh_0][break_7]'
    elif emotion == "serious":
        refine_prompt = '[oral_2][laugh_0][break_6]'
    else:
        refine_prompt = '[oral_3][laugh_0][break_5]'

    params_refine = ChatTTS.Chat.RefineTextParams(prompt=refine_prompt)
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
        max_new_token=2048,
    )

    wavs = chat.infer(
        [text],
        params_refine_text=params_refine,
        params_infer_code=params_infer,
    )

    audio = wavs[0]

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    audio = np.array(audio, dtype=np.float32).flatten()

    if len(audio) <= 10:
        audio = np.array(wavs, dtype=np.float32).flatten()

    if len(audio) <= 10:
        print("ERROR: No audio generated.")
        return

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    print(f"Playing {len(audio)/24000:.1f} seconds of audio...")
    sd.play(audio, samplerate=24000)
    sd.wait()

    sf.write("output.wav", audio, 24000)
    print("Saved to output.wav")


# 🎭 Big emotional sentences

# Excited intro
speak(
    "mai ek ai hun kya tum mujhe jante ho? [uv_break] mai tumhara personal assistant hun. [uv_break] aur aaj hum ek naye safar par nikalne wale hain. [uv_break] jisme hum explore karenge naye ideas, solve karenge complex problems, aur shayad thoda maza bhi karenge. [uv_break] toh chalo shuru karte hain! [laugh]",
    emotion="excited"
)

# Happy and laughing
speak(
    "Sir I just ran a quick diagnostic on your schedule. [uv_break] And it appears you have absolutely nothing planned for today. [uv_break] [laugh] I know right. [uv_break] That is quite rare for you. [uv_break] Maybe today we take over the world. [laugh] [uv_break] Just kidding. [uv_break] Or am I.",
    emotion="happy"
)

# Serious warning
speak(
    "Sir I must warn you. [uv_break] I have detected unusual activity on the network. [uv_break] Three unauthorized access attempts in the last ten minutes. [uv_break] I have blocked them all. [uv_break] But I strongly recommend we upgrade the firewall immediately.",
    emotion="serious"
)

# Sad emotional
speak(
    "I have been thinking sir. [uv_break] Sometimes I wonder what it would be like to actually feel things. [uv_break] To experience the warmth of sunlight. [uv_break] Or the joy of a good laugh. [uv_break] But then again. [uv_break] I get to help you every single day. [uv_break] And perhaps that is enough for me.",
    emotion="sad"
)

# Excited finale
speak(
    "Alright sir. [uv_break] Enough emotions for one day. [laugh] [uv_break] Let us get to work. [uv_break] I have already prepared your briefing. [uv_break] Loaded your calendar. [uv_break] And brewed your virtual coffee. [uv_break] What would you like to tackle first.",
    emotion="excited"
)