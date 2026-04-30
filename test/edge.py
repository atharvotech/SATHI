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

async def speak_hinglish(text, voice=HINGLISH_VOICE):
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

asyncio.run(main())