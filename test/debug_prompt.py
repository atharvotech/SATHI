from llama_cpp import Llama

llm = Llama(
    model_path='./models/gemma-4-E4B-it-Q8_0.gguf',
    n_ctx=512,
    n_threads=4,
    echo=False,
    verbose=False,
)

test_prompt = '<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n'
tokens = llm.tokenize(test_prompt.encode('utf-8'))
print(f"Simple prompt tokens: {len(tokens)}")

# Now test with system prompt
with open('prompts/rulebook.txt', 'r', encoding='utf-8') as f:
    rulebook = f.read()

full = f"<start_of_turn>user\n[System Instructions]\n{rulebook}<end_of_turn>\n"
full += '<start_of_turn>model\n{{"speak": "OK", "actions": []}}<end_of_turn>\n'
full += "<start_of_turn>user\nhi<end_of_turn>\n"
full += "<start_of_turn>model\n"

tokens = llm.tokenize(full.encode('utf-8'))
print(f"Full prompt tokens: {len(tokens)}")
print(f"Prompt chars: {len(full)}")
