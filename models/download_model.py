# download_model.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
model.save("models/MiniLM")
