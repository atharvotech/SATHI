import speech_recognition as sr

def listen_to_cuto():
    recognizer = sr.Recognizer()
    
    # 🛑 YAHI HAI MAGIC SETTINGS JO TUJHE CUT NAHI HONE DENGI
    recognizer.pause_threshold = 2.5  # Ab tu 2.5 second bhi rukega toh mic cut nahi hoga
    recognizer.dynamic_energy_threshold = False # Mic ka auto-cut band
    recognizer.energy_threshold = 300 # Sannate ka level set kar diya
    
    with sr.Microphone() as source:
        print("\n[🎙️ SATHI sun rahi hai... aaram se bol, main nahi kaatungi!]")
        
        try:
            # timeout aur limit hata di hai. Jab tak tu chup nahi hota, ye sunti rahegi.
            # audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
            # print("[⚙️ Processing...]")
            
            # # en-IN (Indian English) se Hinglish thodi theek aati hai
            # text = recognizer.recognize_google(audio, language="en-IN")
            # print(f"Tu bola: '{text}'")
            text = input("Input ")
            return text
            
        except sr.UnknownValueError:
            print("[SATHI: Aawaz theek se nahi aayi, fir se bol!]")
            return ""
        except Exception as e:
            return ""