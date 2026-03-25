from modules.llama_model import generate_llm_response

def summarize_text(content: str, mode: str = "brief") -> str:
    if mode == "brief":
        instruction = "Summarize the following text in 5 bullet points for quick review."
    else:
        instruction = "Give a detailed summary of the following text with headings and subpoints."

    prompt = f"""### Instruction: {instruction}\n\nText:\n{content}\n\n### Summary:"""
    return generate_llm_response(prompt)