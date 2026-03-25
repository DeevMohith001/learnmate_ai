from llama_cpp import Llama
import os

MODEL_PATH = os.path.join("models", "mistral-7b.Q4_K_M.gguf")

# ✅ Defining the model loader globally
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False
)

# ✅ This is the callable function
def generate_llm_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    output = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    return output["choices"][0]["text"].strip()