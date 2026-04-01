from modules.llama_model import generate_llm_response


def summarize_text(text):
    if not text or not text.strip():
        return "No content available to summarize."

    prompt = f"""
    Summarize the following study material for a student.
    Keep it concise, accurate, and useful for revision.
    Use 4 to 6 bullet points.

    Study material:
    {text[:6000]}
    """

    return generate_llm_response(prompt, max_tokens=250, temperature=0.3)
