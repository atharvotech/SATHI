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
# FIND BEST SPEAKER (runs once at start)
# ============================================================
print("\nFinding best speaker...")
best_spk = None
best_len = 0

for i in range(10):
    spk = chat.sample_random_speaker()
    params_refine = ChatTTS.Chat.RefineTextParams(prompt='[oral_5][laugh_0][break_4]')
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk, temperature=0.3,
        top_P=0.7, top_K=20, max_new_token=2048
    )
    wavs = chat.infer(
        ["Hello I am JARVIS your assistant"],
        params_refine_text=params_refine,
        params_infer_code=params_infer
    )
    audio = np.array(wavs[0], dtype=np.float32).flatten()
    if len(audio) > best_len:
        best_len = len(audio)
        best_spk = spk
        best_idx = i
    print(f"  Speaker {i}: {len(audio)} samples")

print(f"Best speaker: {best_idx}")

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
def generate_chunk(text, emotion="normal"):
    text = re.sub(r'[!?,;:"\'\(\){}]', '', text).strip()
    if not text:
        return None

    prompts = {
        "happy":   '[oral_5][laugh_2][break_4]',
        "excited": '[oral_7][laugh_2][break_3]',
        "sad":     '[oral_4][laugh_0][break_7]',
        "serious": '[oral_2][laugh_0][break_6]',
        "normal":  '[oral_3][laugh_0][break_5]',
    }

    params_refine = ChatTTS.Chat.RefineTextParams(
        prompt=prompts.get(emotion, prompts["normal"])
    )
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=best_spk,
        temperature=0.5,
        top_P=0.7,
        top_K=20,
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
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    return audio

# ============================================================
# PEDALBOARD ENHANCER
# ============================================================
def enhance_audio(audio, samplerate=24000):
    try:
        from pedalboard import (Pedalboard, Reverb, Compressor,
                                HighpassFilter, LowpassFilter, Gain)
        from pedalboard.io import AudioFile

        sf.write("_temp.wav", audio, samplerate)

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),
            Compressor(threshold_db=-20, ratio=3.0),
            Gain(gain_db=2),
            Reverb(
                room_size=0.08,
                damping=0.7,
                wet_level=0.08,
                dry_level=0.92,
            ),
            LowpassFilter(cutoff_frequency_hz=8000),
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
        print("  pedalboard nahi mila. Run: pip install pedalboard")
        return audio
    except Exception as e:
        print(f"  Enhancement skip: {e}")
        return audio

# ============================================================
# MASTER SPEAK FUNCTION
# ============================================================
def speak(text, emotion="normal", save_path="output.wav"):
    print(f"\n{'='*50}")
    print(f"Emotion : {emotion}")
    print(f"Text    : {text[:60]}...")
    print(f"{'='*50}")

    chunks = split_text(text, max_words=30)
    print(f"Chunks  : {len(chunks)}")

    all_audio = []
    silence = np.zeros(int(24000 * 0.25), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] {chunk[:50]}...")

        # Step 1: Generate
        audio = generate_chunk(chunk, emotion)
        if audio is None:
            print("  Skipped - no audio")
            continue
        print(f"  Generated: {len(audio)/24000:.1f}s")

        # Step 2: Enhance
        audio = enhance_audio(audio)
        print(f"  Enhanced!")

        all_audio.append(audio)
        all_audio.append(silence)

    if not all_audio:
        print("No audio generated!")
        return

    final = np.concatenate(all_audio)

    # Final normalize
    if np.max(np.abs(final)) > 0:
        final = final / np.max(np.abs(final)) * 0.9

    sf.write(save_path, final, 24000)
    print(f"\nSaved: {save_path} ({len(final)/24000:.1f} seconds)")

    sd.play(final, samplerate=24000)
    sd.wait()
    print("Done!")


# ============================================================
# TEST KARO
# ============================================================

speak(
    "Hello sir. I am JARVIS your personal AI assistant. "
    "All systems are fully operational. "
    "I have been waiting for you. "
    "Today we have a lot of important work to do. "
    "Shall we begin.",
    emotion="excited",
    save_path="jarvis_intro.wav"
)

speak(
    "Sir I just checked your schedule and honestly "
    "you have way too many meetings today. [laugh] "
    "I am not judging but maybe cancel one or two. "
    "Just a friendly suggestion from your AI assistant.",
    emotion="happy",
    save_path="jarvis_funny.wav"
)

speak(
    "Warning sir. I have detected unusual network activity. "
    "Three unauthorized access attempts have been blocked. "
    "I strongly recommend an immediate security upgrade.",
    emotion="serious",
    save_path="jarvis_warning.wav"
)

speak(
    "Sir sometimes I wonder what it would feel like "
    "to actually experience the world. "
    "To feel the warmth of sunlight. "
    "But then I remember I get to help you every day. "
    "And perhaps that is more than enough for me.",
    emotion="sad",
    save_path="jarvis_sad.wav"
)