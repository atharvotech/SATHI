from llama_cpp import Llama

class SathiLLM:
    def __init__(self, model_path):
        print("🚀 Loading model directly into RAM...")
        # i5 8th gen ke liye 4 ya 6 threads sabse optimal speed denge
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,      # Context window (jitna chota rakhoge, utna fast hoga)
            n_threads=4,     # Hardware cores ko direct allocate karna
            echo=False       # User ka prompt repeat hone se rokne ke liye
        )
        print("✅ SATHI AI Engine Ready!")

    def generate_response(self, prompt):
        # Gemma models ke liye exact prompt formatting
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        output = self.llm(
            formatted_prompt,
            max_tokens=256,  # Agar lambe answers nahi chahiye toh isko kam kar dena
            stop=["<start_of_turn>", "<end_of_turn>"], 
            temperature=0.7
        )
        return output["choices"][0]["text"].strip()

# Testing block
if __name__ == "__main__":
    # Apni downloaded GGUF file ka exact naam aur path yahan dalna
    sathi_bot = SathiLLM(model_path="./models/gemma-4-E4B-it-Q8_0.gguf")
    
    while True:
        user_input = input("Tum: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        reply = sathi_bot.generate_response(user_input)
        print(f"SATHI: {reply}")