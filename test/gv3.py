import torch
import ChatTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import re
import os
from scipy import signal

print("Loading ChatTTS...")
chat = ChatTTS.Chat()
chat.load(compile=False)
print("ChatTTS Ready!")

# ============================================================
# VOICE PROFILES — Each voice is carefully tuned
# ============================================================
VOICE_PROFILES = {

    # ─── MALE VOICES ───────────────────────────────────────
    "male_1": {
        "name": "JARVIS — Deep Professional",
        "seed": 42,
        "temperature": 0.28,
        "top_P": 0.65,
        "top_K": 15,
        "oral": 2, "laugh": 0, "break_": 6,
        "pitch_shift": -2,        # Slightly deeper
        "room_size": 0.12,        # Small professional room
        "description": "Deep, calm, professional"
    },
    "male_2": {
        "name": "ALEX — Friendly Casual",
        "seed": 2001,
        "temperature": 0.45,
        "top_P": 0.72,
        "top_K": 20,
        "oral": 5, "laugh": 1, "break_": 4,
        "pitch_shift": 0,
        "room_size": 0.08,
        "description": "Friendly, warm, casual"
    },
    "male_3": {
        "name": "REX — Energetic",
        "seed": 3777,
        "temperature": 0.55,
        "top_P": 0.78,
        "top_K": 22,
        "oral": 7, "laugh": 2, "break_": 3,
        "pitch_shift": 1,
        "room_size": 0.06,
        "description": "High energy, enthusiastic"
    },
    "male_4": {
        "name": "MORGAN — Wise Elder",
        "seed": 4321,
        "temperature": 0.22,
        "top_P": 0.55,
        "top_K": 10,
        "oral": 1, "laugh": 0, "break_": 7,
        "pitch_shift": -3,        # Much deeper/older
        "room_size": 0.15,
        "description": "Slow, wise, authoritative"
    },
    "male_5": {
        "name": "JAKE — Young Witty",
        "seed": 5555,
        "temperature": 0.52,
        "top_P": 0.76,
        "top_K": 20,
        "oral": 6, "laugh": 2, "break_": 3,
        "pitch_shift": 2,         # Slightly higher/younger
        "room_size": 0.06,
        "description": "Young, witty, humorous"
    },

    # ─── FEMALE VOICES ─────────────────────────────────────
    "female_1": {
        "name": "ARIA — Elegant Professional",
        "seed": 6100,
        "temperature": 0.28,
        "top_P": 0.62,
        "top_K": 15,
        "oral": 3, "laugh": 0, "break_": 6,
        "pitch_shift": 3,
        "room_size": 0.10,
        "description": "Elegant, clear, professional"
    },
    "female_2": {
        "name": "SARA — Warm Friendly",
        "seed": 7234,
        "temperature": 0.45,
        "top_P": 0.72,
        "top_K": 18,
        "oral": 5, "laugh": 1, "break_": 4,
        "pitch_shift": 4,
        "room_size": 0.07,
        "description": "Warm, caring, friendly"
    },
    "female_3": {
        "name": "NOVA — Cheerful Energetic",
        "seed": 8367,
        "temperature": 0.60,
        "top_P": 0.80,
        "top_K": 24,
        "oral": 7, "laugh": 2, "break_": 2,
        "pitch_shift": 5,
        "room_size": 0.05,
        "description": "Bubbly, cheerful, energetic"
    },
    "female_4": {
        "name": "LUNA — Calm Soothing",
        "seed": 9445,
        "temperature": 0.20,
        "top_P": 0.52,
        "top_K": 10,
        "oral": 2, "laugh": 0, "break_": 7,
        "pitch_shift": 2,
        "room_size": 0.18,        # Larger softer room
        "description": "Calm, soft, soothing"
    },
    "female_5": {
        "name": "ZOE — Playful Witty",
        "seed": 1678,
        "temperature": 0.55,
        "top_P": 0.78,
        "top_K": 22,
        "oral": 6, "laugh": 2, "break_": 3,
        "pitch_shift": 6,
        "room_size": 0.06,
        "description": "Playful, witty, fun"
    },
}

# Character presets — which voice for which task
CHARACTER_PRESETS = {
    "jarvis":      "male_1",   # Main AI assistant
    "assistant":   "female_2", # Friendly helper
    "narrator":    "male_4",   # Story/report reading
    "alert":       "male_3",   # Warnings/alerts
    "companion":   "female_4", # Calm companion
    "comedian":    "male_5",   # Funny responses
    "news":        "female_1", # News reading
    "cheerful":    "female_3", # Positive responses
}

# ============================================================
# GENERATE SPEAKER EMBEDDINGS
# ============================================================
print("\nGenerating voice embeddings...")
voice_embeddings = {}

for voice_id, config in VOICE_PROFILES.items():
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    spk = chat.sample_random_speaker()
    voice_embeddings[voice_id] = spk
    print(f"  ✅ {voice_id}: {config['name']}")

print("All voices ready!\n")

# ============================================================
# BREATH SOUND GENERATOR
# ============================================================
def generate_breath(duration=0.25, samplerate=24000):
    """Synthetic inhale breath sound"""
    samples = int(samplerate * duration)
    # White noise shaped like a breath
    noise = np.random.normal(0, 0.015, samples).astype(np.float32)

    # Envelope: quick rise, slow fall
    env = np.zeros(samples)
    rise = int(samples * 0.3)
    fall = samples - rise
    env[:rise] = np.linspace(0, 1, rise)
    env[rise:] = np.linspace(1, 0, fall)

    breath = noise * env

    # Bandpass filter — make it sound like real breath
    b, a = signal.butter(2, [800, 4000], btype='band', fs=samplerate)
    breath = signal.filtfilt(b, a, breath).astype(np.float32)

    return breath

# ============================================================
# PITCH SHIFTER
# ============================================================
def pitch_shift(audio, semitones, samplerate=24000):
    """Shift pitch up or down by semitones"""
    if semitones == 0:
        return audio
    factor = 2 ** (semitones / 12.0)
    # Resample to shift pitch
    original_len = len(audio)
    new_len = int(original_len / factor)
    resampled = signal.resample(audio, new_len)
    # Stretch back to original length
    result = signal.resample(resampled, original_len)
    return result.astype(np.float32)

# ============================================================
# BACKGROUND NOISE GENERATOR
# ============================================================
def generate_room_noise(duration_samples, noise_type="office", samplerate=24000):
    """Subtle background ambient sound"""
    noise = np.random.normal(0, 1, duration_samples).astype(np.float32)

    if noise_type == "office":
        # Low hum + subtle noise
        b, a = signal.butter(2, [50, 300], btype='band', fs=samplerate)
        noise = signal.filtfilt(b, a, noise) * 0.008

    elif noise_type == "room":
        b, a = signal.butter(2, 200, btype='low', fs=samplerate)
        noise = signal.filtfilt(b, a, noise) * 0.005

    elif noise_type == "studio":
        # Almost silent — just tiny hiss
        noise = noise * 0.002

    return noise.astype(np.float32)

# ============================================================
# TEXT SPLITTER
# ============================================================
def split_text(text, max_words=28):
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
# CHATTTS GENERATOR — Loop bug fixed
# ============================================================
def generate_chunk(text, voice_id="male_1", emotion="normal"):
    text = re.sub(r'[!?,;:"\'\(\){}]', '', text).strip()
    if not text:
        return None

    # Padding to prevent last word cutoff
    text = text + " . . . . . ."

    config = VOICE_PROFILES[voice_id]
    spk = voice_embeddings[voice_id]

    emotion_map = {
        "happy":   f'[oral_6][laugh_2][break_3]',
        "excited": f'[oral_8][laugh_2][break_2]',
        "sad":     f'[oral_3][laugh_0][break_8]',
        "serious": f'[oral_1][laugh_0][break_6]',
        "normal":  f'[oral_{config["oral"]}][laugh_{config["laugh"]}][break_{config["break_"]}]',
    }

    params_refine = ChatTTS.Chat.RefineTextParams(
        prompt=emotion_map.get(emotion, emotion_map["normal"])
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

    # Trim end padding dots (last 0.2s only — not too much)
    trim = int(24000 * 0.2)
    if len(audio) > trim * 2:
        audio = audio[:-trim]

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.85

    return audio

# ============================================================
# FULL AUDIO ENHANCEMENT PIPELINE
# ============================================================
def enhance_audio(audio, voice_id="male_1", samplerate=24000,
                  add_breath=True, bg_noise="office", room_size=None):
    try:
        from pedalboard import (Pedalboard, Reverb, Compressor,
                                HighpassFilter, LowpassFilter, Gain)
        from pedalboard.io import AudioFile

        config = VOICE_PROFILES[voice_id]
        is_male = "male" in voice_id
        r_size = room_size if room_size else config["room_size"]

        # ── Step 1: Pitch Shift ──────────────────────────
        audio = pitch_shift(audio, config["pitch_shift"], samplerate)

        # ── Step 2: Breath Sound ─────────────────────────
        if add_breath:
            breath = generate_breath(duration=0.22, samplerate=samplerate)
            silence_short = np.zeros(int(samplerate * 0.05), dtype=np.float32)
            audio = np.concatenate([breath, silence_short, audio])

        # ── Step 3: Pedalboard EQ + Room ─────────────────
        sf.write("_temp.wav", audio, samplerate)

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=60 if is_male else 100),
            Compressor(threshold_db=-18, ratio=3.5, attack_ms=5, release_ms=100),
            Gain(gain_db=3 if is_male else 2),
            Reverb(
                room_size=r_size,
                damping=0.75,
                wet_level=0.09,
                dry_level=0.91,
            ),
            LowpassFilter(cutoff_frequency_hz=7500 if is_male else 9000),
        ])

        with AudioFile("_temp.wav") as f:
            raw = f.read(f.frames)
            sr = f.samplerate

        processed = board(raw, sr)
        result = processed[0] if processed.ndim > 1 else processed
        audio = result.flatten()

        # ── Step 4: Background Noise ─────────────────────
        if bg_noise and bg_noise != "none":
            noise = generate_room_noise(len(audio), bg_noise, samplerate)
            audio = audio + noise

        # ── Step 5: Final Normalize ───────────────────────
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.88

        if os.path.exists("_temp.wav"):
            os.remove("_temp.wav")

        return audio.astype(np.float32)

    except ImportError:
        print("  Run: pip install pedalboard scipy")
        return audio
    except Exception as e:
        print(f"  Enhancement error: {e}")
        return audio

# ============================================================
# MASTER SPEAK FUNCTION
# ============================================================
def speak(text,
          voice="male_1",       # voice_id OR character name
          emotion="normal",
          bg_noise="office",    # "office" / "room" / "studio" / "none"
          add_breath=True,
          save_path="output.wav"):

    # Resolve character presets
    if voice in CHARACTER_PRESETS:
        voice = CHARACTER_PRESETS[voice]

    if voice not in VOICE_PROFILES:
        print(f"Invalid voice! Choose from: {list(VOICE_PROFILES.keys())}")
        print(f"Or character: {list(CHARACTER_PRESETS.keys())}")
        return

    config = VOICE_PROFILES[voice]
    print(f"\n{'='*55}")
    print(f"Voice   : {config['name']}")
    print(f"Emotion : {emotion}")
    print(f"Noise   : {bg_noise} | Breath: {add_breath}")
    print(f"{'='*55}")

    chunks = split_text(text, max_words=28)
    all_audio = []
    silence = np.zeros(int(24000 * 0.22), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] {chunk[:50]}...")

        audio = generate_chunk(chunk, voice, emotion)
        if audio is None:
            print("  Skipped!")
            continue

        audio = enhance_audio(
            audio,
            voice_id=voice,
            add_breath=(add_breath and i == 0),  # breath only on first chunk
            bg_noise=bg_noise
        )

        all_audio.append(audio)
        all_audio.append(silence)

    if not all_audio:
        print("No audio generated!")
        return

    final = np.concatenate(all_audio)
    if np.max(np.abs(final)) > 0:
        final = final / np.max(np.abs(final)) * 0.88

    sf.write(save_path, final, 24000)
    print(f"\n✅ Saved: {save_path} ({len(final)/24000:.1f}s)")

    sd.play(final, samplerate=24000)
    sd.wait()
    print("Done!\n")

# ============================================================
# VOICE LIST HELPER
# ============================================================
def list_voices():
    print("\n" + "="*55)
    print("AVAILABLE VOICES")
    print("="*55)
    for vid, c in VOICE_PROFILES.items():
        print(f"  {vid:12} → {c['name']}")
    print("\nCHARACTER PRESETS")
    print("="*55)
    for char, vid in CHARACTER_PRESETS.items():
        print(f"  {char:12} → {VOICE_PROFILES[vid]['name']}")
    print("="*55 + "\n")

# ============================================================
# TEST — One sentence per voice to verify all are working
# ============================================================
list_voices()

test_text = (
    "Hello. I am your personal AI assistant. "
    "All systems are fully operational. "
    "How can I help you today."
)

# Test all 10 voices one by one
for voice_id in VOICE_PROFILES.keys():
    speak(
        test_text,
        voice=voice_id,
        emotion="normal",
        bg_noise="office",
        add_breath=True,
        save_path=f"{voice_id}_test.wav"
    )

print("All 10 voices done! Sun ke batao kaunsi best lagi!")

# ============================================================
# EXAMPLE — Character based usage
# ============================================================

# JARVIS style
# speak("Sir all systems are online.", voice="jarvis", emotion="serious")

# Alert style
# speak("Warning. Unauthorized access detected.", voice="alert", emotion="serious", bg_noise="studio")

# Friendly assistant
# speak("Sure I can help you with that.", voice="assistant", emotion="happy")

# Narrator
# speak("In a world where AI and humans coexist.", voice="narrator", emotion="normal", bg_noise="room")