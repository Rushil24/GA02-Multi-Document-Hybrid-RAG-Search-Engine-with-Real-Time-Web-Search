import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_noise(text: str) -> bool:
    noise_terms = ["references", "acknowledgment", "biography", "editor", "©", "ieee"]
    t = text.lower()
    return any(n in t for n in noise_terms)