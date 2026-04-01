import json
import os

import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")


def generate_llm_response(prompt, max_tokens=400, temperature=0.4):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        body = response.json()
        return body.get("response", "").strip()
    except requests.RequestException as exc:
        return (
            "Ollama is not reachable. Start Ollama, pull the configured model, and try again.\n"
            f"Configured model: {OLLAMA_MODEL}\n"
            f"Configured URL: {OLLAMA_BASE_URL}\n"
            f"Error: {exc}"
        )


def generate_json_response(prompt, fallback):
    raw_response = generate_llm_response(prompt, max_tokens=600, temperature=0.3)
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return fallback
