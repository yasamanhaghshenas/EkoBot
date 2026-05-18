import json
import sys
import os

# Ensure repository root is on sys.path for local imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Lightweight stubs for optional LLM libraries so tests run offline.
import types
if 'groq' not in sys.modules:
    groq_mod = types.ModuleType('groq')
    class Groq:
        def __init__(self, api_key=None):
            class Chat:
                class Completions:
                    @staticmethod
                    def create(model=None, messages=None):
                        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='(stubbed groq response)'))])
            self.chat = Chat()
    groq_mod.Groq = Groq
    sys.modules['groq'] = groq_mod

# Stub heavy optional embedding backend to avoid importing sentence-transformers
if 'embeddings' not in sys.modules:
    emb = types.ModuleType('embeddings')
    class SentenceTransformerEmbeddingFunction:
        def __init__(self):
            pass
        def name(self):
            return "default"
        def __call__(self, input):
            # Return a dummy embedding vector per input
            if isinstance(input, str):
                items = [input]
            else:
                items = list(input)
            return [[0.0] for _ in items]
    emb.SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction
    sys.modules['embeddings'] = emb

try:
    import importlib
    importlib.import_module('google.generativeai')
except Exception:
    generativeai = types.ModuleType('google.generativeai')
    def configure(api_key=None):
        return None
    class GenerativeModel:
        def __init__(self, name=None):
            pass
        def generate_content(self, prompt):
            return types.SimpleNamespace(text='(stubbed gemini response)')
    generativeai.configure = configure
    generativeai.GenerativeModel = GenerativeModel
    sys.modules['google.generativeai'] = generativeai

import gemini


def run_case(name, question, history):
    print("==============================")
    print("CASE:", name)
    print("Question:", question)
    print("History:", json.dumps(history, ensure_ascii=False, indent=2))
    try:
        out = gemini.ask(question, history=history, client_type=None, session_language="en")
        print("Response:")
        print(out)
    except Exception as e:
        print("EXCEPTION:", e)
    print()


def main():
    # Case 1: interrupt during profiling - assistant asked dept, user asks unrelated web/info question
    hist1 = [{"role": "assistant", "content": "Which department are you in?"}]
    run_case("interrupt_profile_unrelated", "What are the library hours?", hist1)

    # Case 2: stall timeout - many recent messages and last assistant question is profiling
    hist2 = []
    for i in range(6):
        hist2.append({"role": "user", "content": f"placeholder user {i}"})
        hist2.append({"role": "assistant", "content": "Which department are you in?"})
    run_case("stall_timeout", "Is this still working?", hist2)

    # Case 3: valid profiling reply should NOT be reset
    hist3 = [{"role": "assistant", "content": "Which department are you in?"}]
    run_case("valid_profile_reply", "CE", hist3)


if __name__ == '__main__':
    main()
