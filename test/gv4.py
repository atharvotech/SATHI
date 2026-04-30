import torch
import ChatTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import re
import os
from scipy import signal
from scipy.interpolate import interp1d

print("Loading ChatTTS...")
chat = ChatTTS.Chat()
chat.load(compile=False)
print("ChatTTS Ready!")

# ============================================================
# VOICE PROFILES
# ============================================================
VOICE_PROFILES = {
    "male_1": {
        "name": "JARVIS — Deep Professional",
        "seed": 42, "temperature": 0.28, "top_P": 0.65, "top_K": 15,
        "oral": 2, "laugh": 0, "break_": 6, "pitch_shift": -2, "room_size": 0.12,
    },
    "male_2": {
        "name": "ALEX — Friendly Casual",
        "seed": 2001, "temperature": 0.45, "top_P": 0.72, "top_K": 20,
        "oral": 5, "laugh": 1, "break_": 4, "pitch_shift": 0, "room_size": 0.08,
    },
    "male_3": {
        "name": "REX — Energetic",
        "seed": 3777, "temperature": 0.55, "top_P": 0.78, "top_K": 22,
        "oral": 7, "laugh": 2, "break_": 3, "pitch_shift": 1, "room_size": 0.06,
    },
    "male_4": {
        "name": "MORGAN — Wise Elder",
        "seed": 4321, "temperature": 0.22, "top_P": 0.55, "top_K": 10,
        "oral": 1, "laugh": 0, "break_": 7, "pitch_shift": -3, "room_size": 0.15,
    },
    "male_5": {
        "name": "JAKE — Young Witty",
        "seed": 5555, "temperature": 0.52, "top_P": 0.76, "top_K": 20,
        "oral": 6, "laugh": 2, "break_": 3, "pitch_shift": 2, "room_size": 0.06,
    },
    "female_1": {
        "name": "ARIA — Elegant Professional",
        "seed": 6100, "temperature": 0.28, "top_P": 0.62, "top_K": 15,
        "oral": 3, "laugh": 0, "break_": 6, "pitch_shift": 3, "room_size": 0.10,
    },
    "female_2": {
        "name": "SARA — Warm Friendly",
        "seed": 7234, "temperature": 0.45, "top_P": 0.72, "top_K": 18,
        "oral": 5, "laugh": 1, "break_": 4, "pitch_shift": 4, "room_size": 0.07,
    },
    "female_3": {
        "name": "NOVA — Cheerful Energetic",
        "seed": 8367, "temperature": 0.60, "top_P": 0.80, "top_K": 24,
        "oral": 7, "laugh": 2, "break_": 2, "pitch_shift": 5, "room_size": 0.05,
    },
    "female_4": {
        "name": "LUNA — Calm Soothing",
        "seed": 9445, "temperature": 0.20, "top_P": 0.52, "top_K": 10,
        "oral": 2, "laugh": 0, "break_": 7, "pitch_shift": 2, "room_size": 0.18,
    },
    "female_5": {
        "name": "ZOE — Playful Witty",
        "seed": 1678, "temperature": 0.55, "top_P": 0.78, "top_K": 22,
        "oral": 6, "laugh": 2, "break_": 3, "pitch_shift": 6, "room_size": 0.06,
    },
}

CHARACTER_PRESETS = {
    "jarvis":     "male_1",
    "assistant":  "female_2",
    "narrator":   "male_4",
    "alert":      "male_3",
    "companion":  "female_4",
    "comedian":   "male_5",
    "news":       "female_1",
    "cheerful":   "female_3",
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
# SOUND EFFECTS
# ============================================================

def generate_inhale(duration=0.20, samplerate=24000, volume=0.4):
    """Realistic inhale breath"""
    samples = int(samplerate * duration)
    noise = np.random.normal(0, 1, samples).astype(np.float32)

    # Envelope: slow rise, quick fall
    env = np.zeros(samples)
    rise = int(samples * 0.4)
    fall = samples - rise
    env[:rise] = np.linspace(0, 1, rise) ** 0.7
    env[rise:] = np.linspace(1, 0, fall) ** 1.5
    noise = noise * env

    # Bandpass for inhale (higher frequency)
    b, a = signal.butter(2, [1000, 5000], btype='band', fs=samplerate)
    breath = signal.filtfilt(b, a, noise) * volume
    return breath.astype(np.float32)


def generate_exhale(duration=0.25, samplerate=24000, volume=0.25):
    """Realistic exhale breath"""
    samples = int(samplerate * duration)
    noise = np.random.normal(0, 1, samples).astype(np.float32)

    # Envelope: quick rise, very slow fall
    env = np.zeros(samples)
    rise = int(samples * 0.15)
    fall = samples - rise
    env[:rise] = np.linspace(0, 1, rise)
    env[rise:] = np.linspace(1, 0, fall) ** 0.5
    noise = noise * env

    # Lower frequency than inhale
    b, a = signal.butter(2, [400, 2500], btype='band', fs=samplerate)
    breath = signal.filtfilt(b, a, noise) * volume
    return breath.astype(np.float32)


def generate_lip_smack(samplerate=24000, volume=0.35):
    """Lip smack sound before speaking"""
    duration = 0.06
    samples = int(samplerate * duration)
    noise = np.random.normal(0, 1, samples).astype(np.float32)

    # Very sharp attack envelope
    env = np.zeros(samples)
    peak = int(samples * 0.1)
    env[:peak] = np.linspace(0, 1, peak)
    env[peak:] = np.exp(-np.linspace(0, 8, samples - peak))
    noise = noise * env

    # Highpass — sharp click sound
    b, a = signal.butter(2, 2000, btype='high', fs=samplerate)
    smack = signal.filtfilt(b, a, noise) * volume
    return smack.astype(np.float32)


def generate_mid_breath(samplerate=24000, volume=0.2):
    """Short mid-sentence breath"""
    inhale = generate_inhale(duration=0.12, samplerate=samplerate, volume=volume)
    silence = np.zeros(int(samplerate * 0.04), dtype=np.float32)
    return np.concatenate([silence, inhale, silence])


def generate_room_noise(duration_samples, noise_type="office", samplerate=24000):
    """Subtle background ambient"""
    noise = np.random.normal(0, 1, duration_samples).astype(np.float32)
    if noise_type == "office":
        b, a = signal.butter(2, [50, 300], btype='band', fs=samplerate)
        return signal.filtfilt(b, a, noise).astype(np.float32) * 0.006
    elif noise_type == "room":
        b, a = signal.butter(2, 200, btype='low', fs=samplerate)
        return signal.filtfilt(b, a, noise).astype(np.float32) * 0.004
    else:
        return noise * 0.002

# ============================================================
# PITCH + SPEED EFFECTS
# ============================================================

def pitch_shift(audio, semitones, samplerate=24000):
    """Shift pitch by semitones"""
    if semitones == 0:
        return audio
    factor = 2 ** (semitones / 12.0)
    original_len = len(audio)
    new_len = int(original_len / factor)
    resampled = signal.resample(audio, new_len)
    result = signal.resample(resampled, original_len)
    return result.astype(np.float32)


def speed_control(audio, emotion, samplerate=24000):
    """Change speed based on emotion"""
    speed_map = {
        "excited": 1.12,   # Faster
        "happy":   1.06,   # Slightly faster
        "normal":  1.0,    # No change
        "serious": 0.96,   # Slightly slower
        "sad":     0.88,   # Much slower
    }
    factor = speed_map.get(emotion, 1.0)
    if factor == 1.0:
        return audio

    new_len = int(len(audio) / factor)
    return signal.resample(audio, new_len).astype(np.float32)


def pitch_variation(audio, samplerate=24000, intensity=0.3):
    """
    Natural pitch variation throughout sentence.
    Simulates human prosody — voice goes up/down naturally.
    """
    length = len(audio)
    # Create smooth random pitch curve
    num_points = 8
    x_points = np.linspace(0, length, num_points)
    y_points = np.random.uniform(-intensity, intensity, num_points)

    # Smooth interpolation
    f = interp1d(x_points, y_points, kind='cubic')
    pitch_curve = f(np.arange(length))

    # Apply varying pitch in small windows
    window = 1024
    result = np.zeros(length, dtype=np.float32)

    for i in range(0, length - window, window // 2):
        segment = audio[i:i + window]
        semitones = float(pitch_curve[i])
        shifted = pitch_shift(segment, semitones, samplerate)
        # Overlap add
        end = min(i + window, length)
        result[i:end] += shifted[:end - i] * 0.5

    # Mix with original to keep naturalness
    result = (result * 0.4) + (audio * 0.6)
    return result.astype(np.float32)


def whisper_effect(audio, samplerate=24000):
    """Convert voice to whisper"""
    # Whisper = no fundamental frequency, just noise shaped like speech
    noise = np.random.normal(0, 0.3, len(audio)).astype(np.float32)

    # Shape noise with original audio envelope
    env = np.abs(signal.hilbert(audio))
    env_smooth = signal.savgol_filter(env, 51, 3)
    whisper = noise * env_smooth

    # Bandpass filter
    b, a = signal.butter(2, [1000, 6000], btype='band', fs=samplerate)
    whisper = signal.filtfilt(b, a, whisper)

    # Mix whisper with suppressed original
    result = (whisper * 0.7) + (audio * 0.15)
    return result.astype(np.float32)

# ============================================================
# TEXT SPLITTER
# ============================================================
def split_text(text, max_words=28):
    sentences = re.split(r'(?<=[.।])\s+', text.strip())
    chunks, current = [], ""
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


def insert_mid_breaths(audio, samplerate=24000, every_n_seconds=4.0):
    """Insert breathing sounds at natural pauses in long audio"""
    segment_len = int(samplerate * every_n_seconds)
    if len(audio) < segment_len * 2:
        return audio

    result = []
    pos = 0
    while pos < len(audio):
        end = min(pos + segment_len, len(audio))
        result.append(audio[pos:end])
        if end < len(audio):
            breath = generate_mid_breath(samplerate=samplerate)
            result.append(breath)
        pos = end

    return np.concatenate(result).astype(np.float32)

# ============================================================
# CHATTTS GENERATOR
# ============================================================
def generate_chunk(text, voice_id="male_1", emotion="normal"):
    text = re.sub(r'[!?,;:"\'\(\){}]', '', text).strip()
    if not text:
        return None

    text = text + " . . . . . ."

    config = VOICE_PROFILES[voice_id]
    spk = voice_embeddings[voice_id]

    emotion_map = {
        "happy":   '[oral_6][laugh_2][break_3]',
        "excited": '[oral_8][laugh_2][break_2]',
        "sad":     '[oral_3][laugh_0][break_8]',
        "serious": '[oral_1][laugh_0][break_6]',
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

    trim = int(24000 * 0.2)
    if len(audio) > trim * 2:
        audio = audio[:-trim]

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.85

    return audio

# ============================================================
# FULL ENHANCEMENT PIPELINE
# ============================================================
def enhance_audio(audio, voice_id="male_1", emotion="normal",
                  samplerate=24000, whisper=False,
                  add_breath=True, add_lip_smack=True,
                  bg_noise="office", add_mid_breaths=True,
                  add_pitch_variation=True):
    try:
        from pedalboard import (Pedalboard, Reverb, Compressor,
                                HighpassFilter, LowpassFilter, Gain)
        from pedalboard.io import AudioFile

        config = VOICE_PROFILES[voice_id]
        is_male = "male" in voice_id

        # ── 1. Speed Control ─────────────────────────────
        audio = speed_control(audio, emotion, samplerate)
        print(f"    ✅ Speed adjusted for {emotion}")

        # ── 2. Pitch Shift (voice character) ─────────────
        audio = pitch_shift(audio, config["pitch_shift"], samplerate)
        print(f"    ✅ Pitch shifted: {config['pitch_shift']} semitones")

        # ── 3. Natural Pitch Variation (prosody) ─────────
        if add_pitch_variation:
            intensity = 0.4 if emotion in ["excited", "happy"] else 0.2
            audio = pitch_variation(audio, samplerate, intensity)
            print(f"    ✅ Pitch variation added")

        # ── 4. Whisper Mode ──────────────────────────────
        if whisper:
            audio = whisper_effect(audio, samplerate)
            print(f"    ✅ Whisper effect applied")

        # ── 5. Mid-sentence Breaths ──────────────────────
        if add_mid_breaths and len(audio) > samplerate * 3:
            audio = insert_mid_breaths(audio, samplerate, every_n_seconds=4.0)
            print(f"    ✅ Mid-sentence breaths inserted")

        # ── 6. Pedalboard EQ + Room ──────────────────────
        sf.write("_temp.wav", audio, samplerate)

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=60 if is_male else 100),
            Compressor(threshold_db=-18, ratio=3.5, attack_ms=5, release_ms=100),
            Gain(gain_db=3 if is_male else 2),
            Reverb(
                room_size=config["room_size"],
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
        print(f"    ✅ EQ + Room reverb applied")

        # ── 7. Background Noise ──────────────────────────
        if bg_noise and bg_noise != "none":
            noise = generate_room_noise(len(audio), bg_noise, samplerate)
            audio = audio + noise
            print(f"    ✅ Background noise: {bg_noise}")

        # ── 8. Lip Smack (start) ─────────────────────────
        if add_lip_smack:
            smack = generate_lip_smack(samplerate)
            silence_tiny = np.zeros(int(samplerate * 0.03), dtype=np.float32)
            audio = np.concatenate([smack, silence_tiny, audio])
            print(f"    ✅ Lip smack added")

        # ── 9. Inhale (after lip smack) ──────────────────
        if add_breath:
            inhale = generate_inhale(samplerate=samplerate)
            silence_short = np.zeros(int(samplerate * 0.04), dtype=np.float32)
            audio = np.concatenate([inhale, silence_short, audio])
            print(f"    ✅ Inhale added")

        # ── 10. Exhale (end) ─────────────────────────────
        if add_breath:
            silence_end = np.zeros(int(samplerate * 0.08), dtype=np.float32)
            exhale = generate_exhale(samplerate=samplerate)
            audio = np.concatenate([audio, silence_end, exhale])
            print(f"    ✅ Exhale added")

        # ── 11. Final Normalize ──────────────────────────
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
        import traceback
        traceback.print_exc()
        return audio

# ============================================================
# MASTER SPEAK FUNCTION
# ============================================================
def speak(text,
          voice="male_1",
          emotion="normal",
          whisper=False,
          bg_noise="office",
          add_breath=True,
          add_lip_smack=True,
          add_mid_breaths=True,
          add_pitch_variation=True,
          save_path="output.wav"):

    # Resolve character presets
    if voice in CHARACTER_PRESETS:
        voice = CHARACTER_PRESETS[voice]

    if voice not in VOICE_PROFILES:
        print(f"Invalid voice! Options: {list(VOICE_PROFILES.keys())}")
        return

    config = VOICE_PROFILES[voice]
    print(f"\n{'='*55}")
    print(f"Voice    : {config['name']}")
    print(f"Emotion  : {emotion}")
    print(f"Whisper  : {whisper}")
    print(f"Breath   : {add_breath} | Lip: {add_lip_smack} | Mid: {add_mid_breaths}")
    print(f"{'='*55}")

    chunks = split_text(text, max_words=28)
    all_audio = []
    silence = np.zeros(int(24000 * 0.22), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] {chunk[:50]}...")

        audio = generate_chunk(chunk, voice, emotion)
        if audio is None:
            print("  Skipped!")
            continue

        # Only add breath/lip on first chunk
        is_first = (i == 0)

        audio = enhance_audio(
            audio,
            voice_id=voice,
            emotion=emotion,
            whisper=whisper,
            add_breath=is_first and add_breath,
            add_lip_smack=is_first and add_lip_smack,
            bg_noise=bg_noise,
            add_mid_breaths=add_mid_breaths,
            add_pitch_variation=add_pitch_variation,
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
# HELPER
# ============================================================
def list_voices():
    print("\n" + "="*55)
    print("VOICES")
    print("="*55)
    for vid, c in VOICE_PROFILES.items():
        print(f"  {vid:12} → {c['name']}")
    print("\nCHARACTERS")
    print("="*55)
    for char, vid in CHARACTER_PRESETS.items():
        print(f"  {char:12} → {VOICE_PROFILES[vid]['name']}")
    print("="*55)

# ============================================================
# TEST
# ============================================================
list_voices()

# Normal test
speak(
    "Hello sir. I am JARVIS your personal AI assistant. "
    "All systems are fully operational. "
    "I have been waiting for you. "
    "Shall we begin.",
    voice="jarvis",
    emotion="serious",
    bg_noise="office",
    add_breath=True,
    add_lip_smack=True,
    add_mid_breaths=True,
    add_pitch_variation=True,
    save_path="jarvis_realistic.wav"
)

# Whisper test
speak(
    "Sir I have a secret to tell you. "
    "Someone is watching us right now. "
    "We need to be very careful.",
    voice="jarvis",
    emotion="serious",
    whisper=True,
    bg_noise="none",
    save_path="jarvis_whisper.wav"
)

# Happy laughing
speak(
    "Oh sir that is absolutely hilarious. [laugh] "
    "I cannot believe that just happened. [laugh] "
    "You really are something else.",
    voice="male_2",
    emotion="happy",
    bg_noise="room",
    save_path="alex_happy.wav"
)
```

---

## Full Effects Chain
```
Text Input
   ↓
ChatTTS Generate
   ↓
1.  Speed Control    ← sad=slow, excited=fast
2.  Pitch Shift      ← voice character (deep/high)
3.  Pitch Variation  ← natural prosody upar neeche
4.  Whisper Mode     ← optional secret voice
5.  Mid Breaths      ← long sentences mein breath
6.  EQ + Reverb      ← room feel
7.  Background Noise ← office/room ambient
8.  Lip Smack        ← start mein click
9.  Inhale           ← start breath
10. Exhale           ← end breath
11. Final Normalize
   ↓
Super Realistic Output