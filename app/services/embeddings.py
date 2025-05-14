import os
import pickle
import requests
import json
import torch
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Load Hugging Face model for embeddings
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings using Hugging Face model
def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy().astype("float32")  # Ensure float32 for FAISS


# === Load FAISS index and texts ===

# Go to parent of `services/` → `app/`
BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "faiss_store"
INDEX_PATH = FAISS_DIR / "index.faiss"
TEXTS_PATH = FAISS_DIR / "texts.pkl"

# Load index and corresponding text chunks
try:
    print(f"[FAISS] Loading index from: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

try:
    with open(TEXTS_PATH, "rb") as f:
        stored_texts = pickle.load(f)  # Should be a dict: {id: text}
except Exception as e:
    raise RuntimeError(f"Failed to load texts: {e}")


# === Query handler ===

def get_response_from_query(query: str, session_id: str = ""):
    query_embedding = get_embeddings(query).astype("float32")

    # Search in FAISS index
    D, I = index.search(query_embedding.reshape(1, -1), k=5)

    # Retrieve corresponding texts
    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        text = stored_texts.get(str(idx), "")
        if text:
            results.append(text)

    if not results:
        return "Sorry, I couldn't find any relevant information."

    # Combine found texts
    return " ".join(results)


# === Gemini API integration ===

GEMINI_API_KEY = "AIzaSyA6vlJPND_i1ZG6zsnqEkQJTe17dt907U4"  # Replace with your actual API key
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
def get_response_with_gemini(prompt: str) -> str:
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        response = requests.post(GEMINI_URL, headers=headers, json=data)
        response.raise_for_status()

        response_data = response.json()
        print("Response from Gemini API:", response_data)

        # ✅ Correct field path based on your logs
        return response_data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print("Error generating response with Gemini:", e)
        return "Sorry, I couldn't generate a response using Gemini."
