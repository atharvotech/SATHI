import torch
import ChatTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import re
import os

# ============================================================
# SETUP
# ============================================================
print("Loading ChatTTS...")
chat = ChatTTS.Chat()
chat.load(compile=False)
print("ChatTTS Ready!")

# ============================================================
# PRE-DEFINED VOICES — 5 Male + 5 Female
# Each has unique speaker seed + tone settings
# ============================================================

VOICES = {

    # ─── MALE VOICES ───────────────────────────────────────

    "male_1": {
        "name": "JARVIS — Deep Professional",
        "seed": 42,
        "temperature": 0.3,
        "top_P": 0.6,
        "top_K": 15,
        "oral": 2, "laugh": 0, "break_": 6,
        "description": "Deep, calm, professional — like JARVIS from Iron Man"
    },
    "male_2": {
        "name": "ALEX — Friendly Casual",
        "seed": 137,
        "temperature": 0.5,
        "top_P": 0.75,
        "top_K": 20,
        "oral": 5, "laugh": 1, "break_": 4,
        "description": "Friendly, warm, casual — like a helpful colleague"
    },
    "male_3": {
        "name": "REX — Energetic Excited",
        "seed": 256,
        "temperature": 0.7,
        "top_P": 0.8,
        "top_K": 25,
        "oral": 7, "laugh": 2, "break_": 3,
        "description": "High energy, excited, enthusiastic — like a sports commentator"
    },
    "male_4": {
        "name": "MORGAN — Wise Elder",
        "seed": 512,
        "temperature": 0.2,
        "top_P": 0.5,
        "top_K": 10,
        "oral": 1, "laugh": 0, "break_": 7,
        "description": "Slow, wise, authoritative — like a narrator or elder"
    },
    "male_5": {
        "name": "JAKE — Young Witty",
        "seed": 789,
        "temperature": 0.6,
        "top_P": 0.8,
        "top_K": 20,
        "oral": 6, "laugh": 2, "break_": 3,
        "description": "Young, witty, humorous — like a funny friend"
    },

    # ─── FEMALE VOICES ─────────────────────────────────────

    "female_1": {
        "name": "ARIA — Elegant Professional",
        "seed": 101,
        "temperature": 0.3,
        "top_P": 0.6,
        "top_K": 15,
        "oral": 3, "laugh": 0, "break_": 6,
        "description": "Elegant, clear, professional — like a news anchor"
    },
    "female_2": {
        "name": "SARA — Warm Friendly",
        "seed": 234,
        "temperature": 0.5,
        "top_P": 0.75,
        "top_K": 20,
        "oral": 5, "laugh": 1, "break_": 4,
        "description": "Warm, caring, friendly — like a helpful assistant"
    },
    "female_3": {
        "name": "NOVA — Energetic Cheerful",
        "seed": 367,
        "temperature": 0.7,
        "top_P": 0.8,
        "top_K": 25,
        "oral": 7, "laugh": 2, "break_": 2,
        "description": "Bubbly, cheerful, energetic — like an excited friend"
    },
    "female_4": {
        "name": "LUNA — Calm Soothing",
        "seed": 445,
        "temperature": 0.2,
        "top_P": 0.5,
        "top_K": 10,
        "oral": 2, "laugh": 0, "break_": 7,
        "description": "Calm, soft, soothing — like a meditation guide"
    },
    "female_5": {
        "name": "ZOE — Playful Witty",
        "seed": 678,
        "temperature": 0.65,
        "top_P": 0.8,
        "top_K": 22,
        "oral": 6, "laugh": 2, "break_": 3,
        "description": "Playful, witty, fun — like a funny best friend"
    },
}

# ============================================================
# GENERATE SPEAKER EMBEDDINGS FOR ALL VOICES
# ============================================================
print("\nGenerating voice embeddings...")
voice_embeddings = {}

for voice_id, config in VOICES.items():
    torch.manual_seed(config["seed"])
    spk = chat.sample_random_speaker()
    voice_embeddings[voice_id] = spk
    print(f"  ✅ {voice_id}: {config['name']}")

print("All voices ready!\n")

# ============================================================
# TEXT SPLITTER
# ============================================================
def split_text(text, max_words=30):
    sentences = re.split(r'(?<=[.।])\s+', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if len((current + " " + s).split()) <= max_words:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]

# ============================================================
# CHATTTS GENERATOR
# ============================================================
def generate_chunk(text, voice_id="male_1", emotion="normal"):
    text = re.sub(r'[!?,;:"\'\(\){}]', '', text).strip()
    if not text:
        return None

    # End padding to prevent last word cutoff
    text = text + " . . ."

    config = VOICES[voice_id]
    spk = voice_embeddings[voice_id]

    # Emotion overrides oral/laugh/break settings
    emotion_prompts = {
        "happy":   f'[oral_6][laugh_2][break_3]',
        "excited": f'[oral_8][laugh_2][break_2]',
        "sad":     f'[oral_3][laugh_0][break_7]',
        "serious": f'[oral_1][laugh_0][break_6]',
        "normal":  f'[oral_{config["oral"]}][laugh_{config["laugh"]}][break_{config["break_"]}]',
    }

    params_refine = ChatTTS.Chat.RefineTextParams(
        prompt=emotion_prompts.get(emotion, emotion_prompts["normal"])
    )
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk,
        temperature=config["temperature"],
        top_P=config["top_P"],
        top_K=config["top_K"],
        max_new_token=2048,
    )

    wavs = chat.infer(
        [text],
        params_refine_text=params_refine,
        params_infer_code=params_infer
    )

    audio = np.array(wavs[0], dtype=np.float32).flatten()
    if len(audio) <= 10:
        return None

    # Trim end padding (last 0.3s)
    trim = int(24000 * 0.3)
    if len(audio) > trim * 2:
        audio = audio[:-trim]

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    return audio

# ============================================================
# PEDALBOARD ENHANCER — Tuned per voice type
# ============================================================
def enhance_audio(audio, voice_id="male_1", samplerate=24000):
    try:
        from pedalboard import (Pedalboard, Reverb, Compressor,
                                HighpassFilter, LowpassFilter, Gain)
        from pedalboard.io import AudioFile

        # Different EQ for male vs female voices
        if "male" in voice_id:
            highpass = 60       # Keep more bass for male
            lowpass = 7500
            gain_db = 3
            room_size = 0.1
        else:
            highpass = 100      # Cut more low end for female
            lowpass = 9000      # Keep more highs for female
            gain_db = 2
            room_size = 0.07

        sf.write("_temp.wav", audio, samplerate)

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=highpass),
            Compressor(threshold_db=-18, ratio=3.5),
            Gain(gain_db=gain_db),
            Reverb(
                room_size=room_size,
                damping=0.75,
                wet_level=0.07,
                dry_level=0.93,
            ),
            LowpassFilter(cutoff_frequency_hz=lowpass),
        ])

        with AudioFile("_temp.wav") as f:
            raw = f.read(f.frames)
            sr = f.samplerate

        processed = board(raw, sr)
        result = processed[0] if processed.ndim > 1 else processed

        if os.path.exists("_temp.wav"):
            os.remove("_temp.wav")

        return result.flatten()

    except ImportError:
        print("  Run: pip install pedalboard")
        return audio
    except Exception as e:
        print(f"  Enhancement skip: {e}")
        return audio

# ============================================================
# MASTER SPEAK FUNCTION
# ============================================================
def speak(text, voice="male_1", emotion="normal", save_path="output.wav"):

    if voice not in VOICES:
        print(f"Invalid voice! Choose from: {list(VOICES.keys())}")
        return

    config = VOICES[voice]
    print(f"\n{'='*55}")
    print(f"Voice  : {config['name']}")
    print(f"Emotion: {emotion}")
    print(f"Text   : {text[:55]}...")
    print(f"{'='*55}")

    chunks = split_text(text, max_words=30)
    all_audio = []
    silence = np.zeros(int(24000 * 0.25), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] Generating...")

        audio = generate_chunk(chunk, voice, emotion)
        if audio is None:
            print("  Skipped!")
            continue

        audio = enhance_audio(audio, voice_id=voice)
        all_audio.append(audio)
        all_audio.append(silence)

    if not all_audio:
        print("No audio generated!")
        return

    final = np.concatenate(all_audio)
    if np.max(np.abs(final)) > 0:
        final = final / np.max(np.abs(final)) * 0.9

    sf.write(save_path, final, 24000)
    print(f"\n✅ Saved: {save_path} ({len(final)/24000:.1f}s)")

    sd.play(final, samplerate=24000)
    sd.wait()
    print("Done!")

# ============================================================
# VOICE LIST HELPER
# ============================================================
def list_voices():
    print("\n" + "="*55)
    print("AVAILABLE VOICES")
    print("="*55)
    for vid, config in VOICES.items():
        print(f"\n  {vid}")
        print(f"  Name : {config['name']}")
        print(f"  Style: {config['description']}")
    print("="*55 + "\n")

# ============================================================
# TEST ALL 10 VOICES
# ============================================================
list_voices()

test_text = (
    "Hello. I am your personal AI assistant. "
    "All systems are fully operational and ready. "
    "How can I help you today."
)

# Test all voices
speak(test_text, voice="male_1",   emotion="normal",  save_path="voice_male_1.wav")
speak(test_text, voice="male_2",   emotion="normal",  save_path="voice_male_2.wav")
speak(test_text, voice="male_3",   emotion="excited", save_path="voice_male_3.wav")
speak(test_text, voice="male_4",   emotion="serious", save_path="voice_male_4.wav")
speak(test_text, voice="male_5",   emotion="happy",   save_path="voice_male_5.wav")

speak(test_text, voice="female_1", emotion="normal",  save_path="voice_female_1.wav")
speak(test_text, voice="female_2", emotion="normal",  save_path="voice_female_2.wav")
speak(test_text, voice="female_3", emotion="excited", save_path="voice_female_3.wav")
speak(test_text, voice="female_4", emotion="sad",     save_path="voice_female_4.wav")
speak(test_text, voice="female_5", emotion="happy",   save_path="voice_female_5.wav")

print("\nAll 10 voices generated!")
print("Suno sab aur jo pasand aaye woh use karo JARVIS mein!")