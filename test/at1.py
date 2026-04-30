# ============================================================
# AtharTTS v1.0 — Production Ready TTS Engine
# For Project JARVIS + Future Open Source Launch
# 
# PIPELINE:
# Text → Emotion Detection → ChatTTS → Effects Chain → Output
#
# Install:
# pip install ChatTTS pedalboard scipy transformers
# pip install torch torchaudio soundfile sounddevice
# pip install fastapi uvicorn python-multipart
# ============================================================

import torch
import ChatTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import re
import os
import json
import time
from scipy import signal
from scipy.interpolate import interp1d
from transformers import pipeline as hf_pipeline

# ============================================================
# EMOTION AUTO DETECTOR
# Uses a real sentiment/emotion model — no manual tagging needed
# ============================================================
class EmotionDetector:
    def __init__(self):
        print("Loading emotion detector...")
        try:
            # Lightweight emotion classifier
            self.classifier = hf_pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=1,
                device=-1  # CPU
            )
            self.ready = True
            print("✅ Emotion detector ready!")
        except Exception as e:
            print(f"⚠️ Emotion detector failed: {e}")
            print("   Falling back to keyword detection")
            self.ready = False

        # Keyword fallback mapping
        self.keyword_map = {
            "excited":  ["amazing", "incredible", "wow", "awesome", "fantastic",
                         "excited", "great", "excellent", "wonderful", "brilliant"],
            "happy":    ["happy", "glad", "pleased", "laugh", "haha", "fun",
                         "enjoy", "love", "nice", "good", "yay", "cheerful"],
            "sad":      ["sad", "sorry", "unfortunate", "miss", "lost", "failed",
                         "disappointed", "regret", "unhappy", "depressed", "cry"],
            "serious":  ["warning", "alert", "critical", "urgent", "danger",
                         "error", "failed", "unauthorized", "breach", "threat"],
            "normal":   []
        }

        # Model label → our emotion mapping
        self.model_map = {
            "joy":      "happy",
            "surprise": "excited",
            "anger":    "serious",
            "fear":     "serious",
            "sadness":  "sad",
            "disgust":  "serious",
            "neutral":  "normal",
        }

    def detect(self, text):
        """Auto detect emotion from text"""

        # Try ML model first
        if self.ready:
            try:
                result = self.classifier(text[:512])[0]
                label = result["label"].lower()
                score = result["score"]
                emotion = self.model_map.get(label, "normal")
                print(f"  🧠 Emotion detected: {emotion} ({label}: {score:.2f})")
                return emotion
            except:
                pass

        # Keyword fallback
        text_lower = text.lower()
        scores = {}
        for emotion, keywords in self.keyword_map.items():
            scores[emotion] = sum(1 for kw in keywords if kw in text_lower)

        best = max(scores, key=scores.get)
        detected = best if scores[best] > 0 else "normal"
        print(f"  🔍 Keyword emotion: {detected}")
        return detected


# ============================================================
# SSML PARSER
# Supports: <speed>, <pitch>, <pause>, <whisper>, <emotion>
# ============================================================
class SSMLParser:
    """
    Usage in text:
    <emotion value='happy'>Hello there</emotion>
    <speed value='0.8'>Slow down here</speed>
    <pitch value='+3'>Higher pitch</pitch>
    <pause duration='0.5'/>
    <whisper>Secret message</whisper>
    """

    def parse(self, text):
        segments = []

        # Remove outer whitespace
        text = text.strip()

        # Pattern to find SSML tags
        pattern = r'(<[^>]+>.*?</[^>]+>|<[^/][^>]*/?>|[^<]+)'
        parts = re.findall(pattern, text, re.DOTALL)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Pause tag
            if re.match(r'<pause', part):
                dur = re.search(r'duration=["\']([0-9.]+)["\']', part)
                duration = float(dur.group(1)) if dur else 0.3
                segments.append({
                    "type": "pause",
                    "duration": duration,
                    "text": ""
                })

            # Whisper tag
            elif re.match(r'<whisper>', part):
                inner = re.sub(r'</?whisper>', '', part).strip()
                segments.append({
                    "type": "speech",
                    "text": inner,
                    "whisper": True,
                    "speed": 1.0,
                    "pitch": 0,
                    "emotion": None
                })

            # Emotion tag
            elif re.match(r'<emotion', part):
                val = re.search(r'value=["\'](\w+)["\']', part)
                inner = re.sub(r'<emotion[^>]*>|</emotion>', '', part).strip()
                segments.append({
                    "type": "speech",
                    "text": inner,
                    "emotion": val.group(1) if val else None,
                    "whisper": False,
                    "speed": 1.0,
                    "pitch": 0
                })

            # Speed tag
            elif re.match(r'<speed', part):
                val = re.search(r'value=["\']([0-9.]+)["\']', part)
                inner = re.sub(r'<speed[^>]*>|</speed>', '', part).strip()
                segments.append({
                    "type": "speech",
                    "text": inner,
                    "speed": float(val.group(1)) if val else 1.0,
                    "emotion": None,
                    "whisper": False,
                    "pitch": 0
                })

            # Pitch tag
            elif re.match(r'<pitch', part):
                val = re.search(r'value=["\']([+-]?[0-9]+)["\']', part)
                inner = re.sub(r'<pitch[^>]*>|</pitch>', '', part).strip()
                segments.append({
                    "type": "speech",
                    "text": inner,
                    "pitch": int(val.group(1)) if val else 0,
                    "emotion": None,
                    "whisper": False,
                    "speed": 1.0
                })

            # Plain text
            else:
                clean = re.sub(r'<[^>]+>', '', part).strip()
                if clean:
                    segments.append({
                        "type": "speech",
                        "text": clean,
                        "emotion": None,
                        "whisper": False,
                        "speed": 1.0,
                        "pitch": 0
                    })

        return segments


# ============================================================
# VOICE PROFILES
# ============================================================
VOICE_PROFILES = {
    "male_1": {
        "name": "JARVIS — Deep Professional",
        "seed": 42, "temperature": 0.28, "top_P": 0.65, "top_K": 15,
        "oral": 2, "laugh": 0, "break_": 6,
        "pitch_shift": -2, "room_size": 0.12,
    },
    "male_2": {
        "name": "ALEX — Friendly Casual",
        "seed": 2001, "temperature": 0.45, "top_P": 0.72, "top_K": 20,
        "oral": 5, "laugh": 1, "break_": 4,
        "pitch_shift": 0, "room_size": 0.08,
    },
    "male_3": {
        "name": "REX — Energetic",
        "seed": 3777, "temperature": 0.55, "top_P": 0.78, "top_K": 22,
        "oral": 7, "laugh": 2, "break_": 3,
        "pitch_shift": 1, "room_size": 0.06,
    },
    "male_4": {
        "name": "MORGAN — Wise Elder",
        "seed": 4321, "temperature": 0.22, "top_P": 0.55, "top_K": 10,
        "oral": 1, "laugh": 0, "break_": 7,
        "pitch_shift": -3, "room_size": 0.15,
    },
    "male_5": {
        "name": "JAKE — Young Witty",
        "seed": 5555, "temperature": 0.52, "top_P": 0.76, "top_K": 20,
        "oral": 6, "laugh": 2, "break_": 3,
        "pitch_shift": 2, "room_size": 0.06,
    },
    "female_1": {
        "name": "ARIA — Elegant Professional",
        "seed": 6100, "temperature": 0.28, "top_P": 0.62, "top_K": 15,
        "oral": 3, "laugh": 0, "break_": 6,
        "pitch_shift": 3, "room_size": 0.10,
    },
    "female_2": {
        "name": "SARA — Warm Friendly",
        "seed": 7234, "temperature": 0.45, "top_P": 0.72, "top_K": 18,
        "oral": 5, "laugh": 1, "break_": 4,
        "pitch_shift": 4, "room_size": 0.07,
    },
    "female_3": {
        "name": "NOVA — Cheerful Energetic",
        "seed": 8367, "temperature": 0.60, "top_P": 0.80, "top_K": 24,
        "oral": 7, "laugh": 2, "break_": 2,
        "pitch_shift": 5, "room_size": 0.05,
    },
    "female_4": {
        "name": "LUNA — Calm Soothing",
        "seed": 9445, "temperature": 0.20, "top_P": 0.52, "top_K": 10,
        "oral": 2, "laugh": 0, "break_": 7,
        "pitch_shift": 2, "room_size": 0.18,
    },
    "female_5": {
        "name": "ZOE — Playful Witty",
        "seed": 1678, "temperature": 0.55, "top_P": 0.78, "top_K": 22,
        "oral": 6, "laugh": 2, "break_": 3,
        "pitch_shift": 6, "room_size": 0.06,
    },
}

CHARACTER_PRESETS = {
    "jarvis":    "male_1",
    "assistant": "female_2",
    "narrator":  "male_4",
    "alert":     "male_3",
    "companion": "female_4",
    "comedian":  "male_5",
    "news":      "female_1",
    "cheerful":  "female_3",
}

# ============================================================
# ATHARTTS ENGINE — Main Class
# ============================================================
class AtharTTS:
    def __init__(self):
        print("\n" + "="*55)
        print("  AtharTTS v1.0 — Initializing...")
        print("="*55)

        # Load ChatTTS
        print("\n[1/3] Loading ChatTTS...")
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)

        # Load emotion detector
        print("\n[2/3] Loading Emotion Detector...")
        self.emotion_detector = EmotionDetector()

        # Load SSML parser
        self.ssml_parser = SSMLParser()

        # Generate voice embeddings
        print("\n[3/3] Generating Voice Embeddings...")
        self.embeddings = {}
        for vid, config in VOICE_PROFILES.items():
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            self.embeddings[vid] = self.chat.sample_random_speaker()
            print(f"  ✅ {vid}: {config['name']}")

        self.samplerate = 24000
        print("\n✅ AtharTTS Ready!\n")

    # ── Sound Effects ───────────────────────────────────────

    def _inhale(self, duration=0.20, volume=0.4):
        samples = int(self.samplerate * duration)
        noise = np.random.normal(0, 1, samples).astype(np.float32)
        env = np.zeros(samples)
        rise = int(samples * 0.4)
        env[:rise] = np.linspace(0, 1, rise) ** 0.7
        env[rise:] = np.linspace(1, 0, samples - rise) ** 1.5
        noise *= env
        b, a = signal.butter(2, [1000, 5000], btype='band', fs=self.samplerate)
        return (signal.filtfilt(b, a, noise) * volume).astype(np.float32)

    def _exhale(self, duration=0.25, volume=0.25):
        samples = int(self.samplerate * duration)
        noise = np.random.normal(0, 1, samples).astype(np.float32)
        env = np.zeros(samples)
        rise = int(samples * 0.15)
        env[:rise] = np.linspace(0, 1, rise)
        env[rise:] = np.linspace(1, 0, samples - rise) ** 0.5
        noise *= env
        b, a = signal.butter(2, [400, 2500], btype='band', fs=self.samplerate)
        return (signal.filtfilt(b, a, noise) * volume).astype(np.float32)

    def _lip_smack(self, volume=0.35):
        samples = int(self.samplerate * 0.06)
        noise = np.random.normal(0, 1, samples).astype(np.float32)
        env = np.zeros(samples)
        peak = int(samples * 0.1)
        env[:peak] = np.linspace(0, 1, peak)
        env[peak:] = np.exp(-np.linspace(0, 8, samples - peak))
        noise *= env
        b, a = signal.butter(2, 2000, btype='high', fs=self.samplerate)
        return (signal.filtfilt(b, a, noise) * volume).astype(np.float32)

    def _mid_breath(self, volume=0.2):
        inhale = self._inhale(duration=0.12, volume=volume)
        sil = np.zeros(int(self.samplerate * 0.04), dtype=np.float32)
        return np.concatenate([sil, inhale, sil])

    def _room_noise(self, n_samples, noise_type="office"):
        noise = np.random.normal(0, 1, n_samples).astype(np.float32)
        if noise_type == "office":
            b, a = signal.butter(2, [50, 300], btype='band', fs=self.samplerate)
            return (signal.filtfilt(b, a, noise) * 0.006).astype(np.float32)
        elif noise_type == "room":
            b, a = signal.butter(2, 200, btype='low', fs=self.samplerate)
            return (signal.filtfilt(b, a, noise) * 0.004).astype(np.float32)
        return (noise * 0.002).astype(np.float32)

    # ── Audio Processing ────────────────────────────────────

    def _pitch_shift(self, audio, semitones):
        if semitones == 0:
            return audio
        factor = 2 ** (semitones / 12.0)
        orig_len = len(audio)
        resampled = signal.resample(audio, int(orig_len / factor))
        return signal.resample(resampled, orig_len).astype(np.float32)

    def _speed_control(self, audio, emotion=None, custom_speed=None):
        if custom_speed:
            factor = custom_speed
        else:
            speed_map = {
                "excited": 1.12, "happy": 1.06,
                "normal": 1.0, "serious": 0.96, "sad": 0.88
            }
            factor = speed_map.get(emotion, 1.0)
        if factor == 1.0:
            return audio
        return signal.resample(audio, int(len(audio) / factor)).astype(np.float32)

    def _pitch_variation(self, audio, intensity=0.3):
        length = len(audio)
        x_pts = np.linspace(0, length, 8)
        y_pts = np.random.uniform(-intensity, intensity, 8)
        f = interp1d(x_pts, y_pts, kind='cubic')
        curve = f(np.arange(length))
        window = 1024
        result = np.zeros(length, dtype=np.float32)
        for i in range(0, length - window, window // 2):
            seg = audio[i:i + window]
            shifted = self._pitch_shift(seg, float(curve[i]))
            end = min(i + window, length)
            result[i:end] += shifted[:end - i] * 0.5
        return ((result * 0.4) + (audio * 0.6)).astype(np.float32)

    def _whisper(self, audio):
        noise = np.random.normal(0, 0.3, len(audio)).astype(np.float32)
        env = np.abs(signal.hilbert(audio))
        env_smooth = signal.savgol_filter(env, 51, 3)
        whisper = noise * env_smooth
        b, a = signal.butter(2, [1000, 6000], btype='band', fs=self.samplerate)
        whisper = signal.filtfilt(b, a, whisper)
        return ((whisper * 0.7) + (audio * 0.15)).astype(np.float32)

    def _insert_mid_breaths(self, audio, every_n_sec=4.0):
        seg_len = int(self.samplerate * every_n_sec)
        if len(audio) < seg_len * 2:
            return audio
        result = []
        pos = 0
        while pos < len(audio):
            end = min(pos + seg_len, len(audio))
            result.append(audio[pos:end])
            if end < len(audio):
                result.append(self._mid_breath())
            pos = end
        return np.concatenate(result).astype(np.float32)

    # ── Text Processing ─────────────────────────────────────

    def _split_text(self, text, max_words=28):
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

    # ── ChatTTS Generation ──────────────────────────────────

    def _generate_chunk(self, text, voice_id, emotion):
        text = re.sub(r'[!?,;:"\'\(\){}]', '', text).strip()
        if not text:
            return None

        text = text + " . . . . . ."
        config = VOICE_PROFILES[voice_id]
        spk = self.embeddings[voice_id]

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

        wavs = self.chat.infer(
            [text],
            params_refine_text=params_refine,
            params_infer_code=params_infer
        )

        audio = np.array(wavs[0], dtype=np.float32).flatten()
        if len(audio) <= 10:
            return None

        trim = int(self.samplerate * 0.2)
        if len(audio) > trim * 2:
            audio = audio[:-trim]

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.85
        return audio

    # ── Full Enhancement ────────────────────────────────────

    def _enhance(self, audio, voice_id, emotion,
                 whisper=False, custom_speed=None,
                 custom_pitch=0, bg_noise="office",
                 add_breath=True, add_lip_smack=True,
                 add_mid_breaths=True, add_pitch_variation=True):
        try:
            from pedalboard import (Pedalboard, Reverb, Compressor,
                                    HighpassFilter, LowpassFilter, Gain)
            from pedalboard.io import AudioFile

            config = VOICE_PROFILES[voice_id]
            is_male = "male" in voice_id

            # 1. Speed
            audio = self._speed_control(audio, emotion, custom_speed)

            # 2. Voice pitch character
            total_pitch = config["pitch_shift"] + custom_pitch
            audio = self._pitch_shift(audio, total_pitch)

            # 3. Natural pitch variation
            if add_pitch_variation:
                intensity = 0.4 if emotion in ["excited", "happy"] else 0.2
                audio = self._pitch_variation(audio, intensity)

            # 4. Whisper
            if whisper:
                audio = self._whisper(audio)

            # 5. Mid breaths
            if add_mid_breaths and len(audio) > self.samplerate * 3:
                audio = self._insert_mid_breaths(audio)

            # 6. Pedalboard EQ + Room
            sf.write("_temp_athartts.wav", audio, self.samplerate)

            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=60 if is_male else 100),
                Compressor(threshold_db=-18, ratio=3.5,
                           attack_ms=5, release_ms=100),
                Gain(gain_db=3 if is_male else 2),
                Reverb(
                    room_size=config["room_size"],
                    damping=0.75,
                    wet_level=0.09,
                    dry_level=0.91,
                ),
                LowpassFilter(cutoff_frequency_hz=7500 if is_male else 9000),
            ])

            with AudioFile("_temp_athartts.wav") as f:
                raw = f.read(f.frames)
                sr = f.samplerate

            processed = board(raw, sr)
            result = processed[0] if processed.ndim > 1 else processed
            audio = result.flatten()

            # 7. Background noise
            if bg_noise and bg_noise != "none":
                audio = audio + self._room_noise(len(audio), bg_noise)

            # 8. Lip smack
            if add_lip_smack:
                sil = np.zeros(int(self.samplerate * 0.03), dtype=np.float32)
                audio = np.concatenate([self._lip_smack(), sil, audio])

            # 9. Inhale
            if add_breath:
                sil = np.zeros(int(self.samplerate * 0.04), dtype=np.float32)
                audio = np.concatenate([self._inhale(), sil, audio])

            # 10. Exhale
            if add_breath:
                sil = np.zeros(int(self.samplerate * 0.08), dtype=np.float32)
                audio = np.concatenate([audio, sil, self._exhale()])

            # 11. Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.88

            if os.path.exists("_temp_athartts.wav"):
                os.remove("_temp_athartts.wav")

            return audio.astype(np.float32)

        except Exception as e:
            print(f"  Enhancement error: {e}")
            return audio

    # ── MAIN SPEAK FUNCTION ─────────────────────────────────

    def speak(self, text,
              voice="male_1",
              emotion="auto",          # "auto" = detect automatically
              whisper=False,
              bg_noise="office",
              add_breath=True,
              add_lip_smack=True,
              add_mid_breaths=True,
              add_pitch_variation=True,
              custom_speed=None,
              custom_pitch=0,
              save_path="output.wav",
              play=True,
              use_ssml=False):         # Enable SSML tags

        start_time = time.time()

        # Resolve character preset
        if voice in CHARACTER_PRESETS:
            voice = CHARACTER_PRESETS[voice]

        if voice not in VOICE_PROFILES:
            print(f"Invalid voice! Options: {list(VOICE_PROFILES.keys())}")
            return None

        config = VOICE_PROFILES[voice]

        print(f"\n{'='*55}")
        print(f"AtharTTS v1.0")
        print(f"Voice   : {config['name']}")

        # SSML Mode
        if use_ssml:
            return self._speak_ssml(text, voice, bg_noise, save_path, play)

        # Auto emotion detection
        if emotion == "auto":
            emotion = self.emotion_detector.detect(text)
        print(f"Emotion : {emotion}")
        print(f"{'='*55}")

        # Split and generate
        chunks = self._split_text(text)
        all_audio = []
        silence = np.zeros(int(self.samplerate * 0.22), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            print(f"[{i+1}/{len(chunks)}] {chunk[:50]}...")

            audio = self._generate_chunk(chunk, voice, emotion)
            if audio is None:
                continue

            is_first = (i == 0)
            audio = self._enhance(
                audio, voice, emotion,
                whisper=whisper,
                custom_speed=custom_speed,
                custom_pitch=custom_pitch,
                bg_noise=bg_noise,
                add_breath=is_first and add_breath,
                add_lip_smack=is_first and add_lip_smack,
                add_mid_breaths=add_mid_breaths,
                add_pitch_variation=add_pitch_variation,
            )

            all_audio.append(audio)
            all_audio.append(silence)

        if not all_audio:
            print("No audio generated!")
            return None

        final = np.concatenate(all_audio)
        if np.max(np.abs(final)) > 0:
            final = final / np.max(np.abs(final)) * 0.88

        sf.write(save_path, final, self.samplerate)

        elapsed = time.time() - start_time
        duration = len(final) / self.samplerate
        print(f"\n✅ Generated {duration:.1f}s audio in {elapsed:.1f}s")
        print(f"   Saved: {save_path}")

        if play:
            sd.play(final, samplerate=self.samplerate)
            sd.wait()

        return final

    def _speak_ssml(self, text, voice, bg_noise, save_path, play):
        """Handle SSML tagged text"""
        print("SSML Mode enabled")
        segments = self.ssml_parser.parse(text)
        all_audio = []

        for seg in segments:
            if seg["type"] == "pause":
                pause = np.zeros(
                    int(self.samplerate * seg["duration"]),
                    dtype=np.float32
                )
                all_audio.append(pause)
                print(f"  ⏸️ Pause: {seg['duration']}s")

            elif seg["type"] == "speech" and seg["text"]:
                # Detect emotion if not specified
                emotion = seg.get("emotion") or \
                          self.emotion_detector.detect(seg["text"])

                chunks = self._split_text(seg["text"])
                for chunk in chunks:
                    audio = self._generate_chunk(chunk, voice, emotion)
                    if audio is None:
                        continue

                    audio = self._enhance(
                        audio, voice, emotion,
                        whisper=seg.get("whisper", False),
                        custom_speed=seg.get("speed", None),
                        custom_pitch=seg.get("pitch", 0),
                        bg_noise=bg_noise,
                        add_breath=False,
                        add_lip_smack=False,
                        add_mid_breaths=False,
                        add_pitch_variation=True,
                    )
                    all_audio.append(audio)

        if not all_audio:
            return None

        final = np.concatenate(all_audio)
        if np.max(np.abs(final)) > 0:
            final = final / np.max(np.abs(final)) * 0.88

        sf.write(save_path, final, self.samplerate)
        print(f"✅ SSML audio saved: {save_path}")

        if play:
            sd.play(final, samplerate=self.samplerate)
            sd.wait()

        return final

    def list_voices(self):
        print("\n" + "="*55)
        print("VOICE PROFILES")
        print("="*55)
        for vid, c in VOICE_PROFILES.items():
            print(f"  {vid:12} → {c['name']}")
        print("\nCHARACTER PRESETS")
        print("="*55)
        for char, vid in CHARACTER_PRESETS.items():
            print(f"  {char:12} → {VOICE_PROFILES[vid]['name']}")
        print("="*55)

    def save_voice_profile(self, path="athartts_config.json"):
        """Save config for sharing/open source"""
        config = {
            "version": "1.0",
            "voices": VOICE_PROFILES,
            "characters": CHARACTER_PRESETS,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved: {path}")


# ============================================================
# REST API — FastAPI (for JARVIS integration)
# Run: uvicorn tts:app --reload --port 8000
# ============================================================
def create_api(tts_engine):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel

        app = FastAPI(
            title="AtharTTS API",
            description="Production TTS API for Project JARVIS",
            version="1.0"
        )

        class TTSRequest(BaseModel):
            text: str
            voice: str = "jarvis"
            emotion: str = "auto"
            whisper: bool = False
            bg_noise: str = "office"
            add_breath: bool = True
            use_ssml: bool = False
            save_path: str = "api_output.wav"

        @app.post("/speak")
        async def speak(req: TTSRequest):
            try:
                tts_engine.speak(
                    text=req.text,
                    voice=req.voice,
                    emotion=req.emotion,
                    whisper=req.whisper,
                    bg_noise=req.bg_noise,
                    add_breath=req.add_breath,
                    use_ssml=req.use_ssml,
                    save_path=req.save_path,
                    play=False
                )
                return FileResponse(req.save_path, media_type="audio/wav")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/voices")
        async def voices():
            return {
                "voices": list(VOICE_PROFILES.keys()),
                "characters": list(CHARACTER_PRESETS.keys())
            }

        @app.get("/health")
        async def health():
            return {"status": "AtharTTS running", "version": "1.0"}

        return app

    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None


# ============================================================
# INITIALIZE + TEST
# ============================================================
tts = AtharTTS()
tts.list_voices()

# Test 1 — Auto emotion detection
tts.speak(
    "Hello sir. I am JARVIS your personal AI assistant. "
    "All systems are fully operational. Shall we begin.",
    voice="jarvis",
    emotion="auto",
    save_path="test_auto_emotion.wav"
)

# Test 2 — SSML mode
tts.speak(
    "<emotion value='serious'>Warning sir.</emotion>"
    "<pause duration='0.4'/>"
    "<speed value='0.9'>I have detected unauthorized access.</speed>"
    "<pause duration='0.3'/>"
    "<whisper>We need to act immediately.</whisper>",
    voice="jarvis",
    use_ssml=True,
    save_path="test_ssml.wav"
)

# Test 3 — Custom speed + pitch
tts.speak(
    "Sir that is absolutely hilarious. I cannot believe that just happened.",
    voice="male_5",
    emotion="happy",
    custom_speed=1.1,
    custom_pitch=1,
    save_path="test_custom.wav"
)

# Save config for open source
tts.save_voice_profile("athartts_config.json")

# Start API (uncomment to run)
# app = create_api(tts)
# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)