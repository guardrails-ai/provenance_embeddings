import nltk
from sentence_transformers import SentenceTransformer

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
print("NLTK stuff loaded successfully.")

# Load model for default embedding function
SentenceTransformer("paraphrase-MiniLM-L6-v2")