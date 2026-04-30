import torch
import ChatTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import re
import os

# ============================================================
# LAYER 1: CHATTTS - Generate base speech with emotions
# ============================================================
print("="*50)
print("LAYER 1: Loading ChatTTS model...")
print("="*50)
chat = ChatTTS.Chat()
chat.load(compile=False)
print("ChatTTS Ready!")

# Find best speaker from 10 options
print("\nFinding best speaker voice...")
best_spk = None
best_idx = 0

test_text = "Hello I am JARVIS your personal assistant"
best_audio_len = 0

for i in range(10):
    spk = chat.sample_random_speaker()
    params_refine = ChatTTS.Chat.RefineTextParams(prompt='[oral_5][laugh_0][break_4]')
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk, temperature=0.3, top_P=0.7, top_K=20, max_new_token=2048
    )
    wavs = chat.infer([test_text], params_refine_text=params_refine, params_infer_code=params_infer)
    audio = np.array(wavs[0], dtype=np.float32).flatten()
    
    if len(audio) > best_audio_len:
        best_audio_len = len(audio)
        best_spk = spk
        best_idx = i
    print(f"  Speaker {i}: {len(audio)} samples")

print(f"Best speaker: {best_idx} — using this voice!")

# ============================================================
# LAYER 2: TEXT CHUNKING - Handle long texts
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
# LAYER 3: CHATTTS GENERATION - With emotion control
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

    params_refine = ChatTTS.Chat.RefineTextParams(prompt=prompts.get(emotion, prompts["normal"]))
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=best_spk,
        temperature=0.5,
        top_P=0.7,
        top_K=20,
        max_new_token=2048,
    )

    wavs = chat.infer([text], params_refine_text=params_refine, params_infer_code=params_infer)
    audio = np.array(wavs[0], dtype=np.float32).flatten()

    if len(audio) <= 10:
        return None
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    return audio

# ============================================================
# LAYER 4: AUDIO ENHANCEMENT - Pedalboard post processing
# ============================================================
def enhance_audio(audio, samplerate=24000):
    try:
        from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, LowpassFilter, Gain
        from pedalboard.io import AudioFile

        sf.write("_temp_raw.wav", audio, samplerate)

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),       # Remove low rumble
            Compressor(threshold_db=-20, ratio=3.0),      # Even out volume
            Gain(gain_db=2),                              # Slight boost
            Reverb(
                room_size=0.08,
                damping=0.7,
                wet_level=0.08,
                dry_level=0.92,
            ),                                            # Small room feel
            LowpassFilter(cutoff_frequency_hz=8000),      # Remove harsh highs
        ])

        with AudioFile("_temp_raw.wav") as f:
            raw = f.read(f.frames)
            sr = f.samplerate

        processed = board(raw, sr)
        result = processed[0] if processed.ndim > 1 else processed
        
        # Cleanup temp file
        os.remove("_temp_raw.wav")
        
        print("  ✅ Layer 4: Audio enhanced with pedalboard")
        return result.flatten()

    except ImportError:
        print("  ⚠️ pedalboard not installed. Run: pip install pedalboard")
        print("  Skipping enhancement layer...")
        return audio

# ============================================================
# LAYER 5: RVC VOICE CONVERSION - Ultra realistic voice
# ============================================================
def apply_rvc(audio, samplerate=24000, model_path=None, index_path=None, pitch=0):
    if not model_path or not os.path.exists(model_path):
        print("  ⚠️ Layer 5: No RVC model found - skipping")
        print("  Tip: Download .pth model from weights.gg and set RVC_MODEL_PATH")
        return audio

    try:
        from rvc_python.infer import RVCInference

        sf.write("_temp_enhanced.wav", audio, samplerate)

        rvc = RVCInference(device="cpu")
        rvc.load_model(model_path, index_path=index_path)
        rvc.infer_file(
            input_path="_temp_enhanced.wav",
            output_path="_temp_rvc.wav",
            f0_method="rmvpe",
            f0_up_key=pitch,
            index_rate=0.75,
            filter_radius=3,
            resample_sr=24000,
            rms_mix_rate=0.25,
            protect=0.33,
        )

        rvc_audio, _ = sf.read("_temp_rvc.wav")
        rvc_audio = np.array(rvc_audio, dtype=np.float32).flatten()

        os.remove("_temp_enhanced.wav")
        os.remove("_temp_rvc.wav")

        print("  ✅ Layer 5: RVC voice conversion applied!")
        return rvc_audio

    except ImportError:
        print("  ⚠️ rvc_python not installed. Run: pip install rvc-python")
        return audio
    except Exception as e:
        print(f"  ⚠️ RVC failed: {e}")
        return audio

# ============================================================
# MASTER SPEAK FUNCTION - All layers combined
# ============================================================

# 🔧 CONFIG - Set your RVC model path here if you have one
RVC_MODEL_PATH = None        # Example: "models/jarvis.pth"
RVC_INDEX_PATH = None        # Example: "models/jarvis.index"
RVC_PITCH = 0                # Semitones: 0=same, +2=higher, -2=lower/deeper

def speak(text, emotion="normal", save_path="final_output.wav"):
    print(f"\n{'='*50}")
    print(f"Speaking ({emotion}): {text[:60]}...")
    print(f"{'='*50}")

    # LAYER 2: Split into chunks
    chunks = split_text(text, max_words=30)
    print(f"Layer 2: Split into {len(chunks)} chunks")

    all_audio = []
    silence = np.zeros(int(24000 * 0.25), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

        # LAYER 3: Generate with ChatTTS
        print("  🎤 Layer 3: Generating with ChatTTS...")
        audio = generate_chunk(chunk, emotion)

        if audio is None:
            print("  ❌ Chunk failed, skipping...")
            continue

        print(f"  ✅ Layer 3: Generated {len(audio)/24000:.1f}s of audio")

        # LAYER 4: Enhance audio
        print("  🎚️ Layer 4: Enhancing audio...")
        audio = enhance_audio(audio)

        # LAYER 5: Apply RVC
        print("  🔄 Layer 5: Applying RVC...")
        audio = apply_rvc(audio, model_path=RVC_MODEL_PATH, index_path=RVC_INDEX_PATH, pitch=RVC_PITCH)

        all_audio.append(audio)
        all_audio.append(silence)

    if not all_audio:
        print("No audio generated!")
        return

    # Join all chunks
    final = np.concatenate(all_audio)

    # Normalize final output
    if np.max(np.abs(final)) > 0:
        final = final / np.max(np.abs(final)) * 0.9

    sf.write(save_path, final, 24000)
    print(f"\n✅ ALL LAYERS DONE!")
    print(f"Total duration: {len(final)/24000:.1f} seconds")
    print(f"Saved to: {save_path}")

    # PLAY
    print("Playing...")
    sd.play(final, samplerate=24000)
    sd.wait()
    print("Done!")


# ============================================================
# TEST - Suno kitna realistic hai!
# ============================================================

speak(
    "Hello sir. I am JARVIS your personal AI assistant. [uv_break] All systems are fully operational. [uv_break] I have been waiting for you.",
    emotion="excited",
    save_path="jarvis_intro.wav"
)

speak(
    "Sir I just checked your schedule and honestly [uv_break] you have way too many meetings today. [laugh] [uv_break] I am not judging but maybe cancel one or two.",
    emotion="happy",
    save_path="jarvis_funny.wav"
)

speak(
    "Warning sir. I have detected unusual network activity. [uv_break] Three unauthorized access attempts blocked. [uv_break] Recommend immediate security upgrade.",
    emotion="serious",
    save_path="jarvis_warning.wav"
)