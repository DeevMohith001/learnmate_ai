from __future__ import annotations

import os
import re
from typing import BinaryIO

import pdfplumber


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_text_from_pdf(uploaded_file: BinaryIO) -> str:
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            if not pdf.pages:
                raise ValueError("The uploaded PDF has no pages.")

            extracted_pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)

            if not extracted_pages:
                raise ValueError("No readable text was found in the uploaded PDF.")
            return "\n".join(extracted_pages)
    except Exception as exc:
        raise ValueError(f"Could not extract text from the PDF: {exc}") from exc


def chunk_text(text: str, length: int = 1000) -> list[str]:
    return [text[i:i + length] for i in range(0, len(text), length)]


def clean_token(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", token).lower().strip()
