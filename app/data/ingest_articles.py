# app/data/ingest_articles.py

import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "faiss_store"
FAISS_DIR.mkdir(exist_ok=True)
INDEX_PATH = FAISS_DIR / "index.faiss"
TEXTS_PATH = FAISS_DIR / "texts.pkl"

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    emb = embeddings.numpy()
    emb /= np.linalg.norm(emb)
    return emb.astype("float32")

def fetch_articles():
    urls = [
        "https://www.bbc.com/news", 
        "https://edition.cnn.com/world", 
        "https://www.reuters.com/news/world"
    ]
    articles = []

    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            full_text = " ".join(p.text.strip() for p in paragraphs)
            if full_text:
                articles.append(full_text[:20000])
        except Exception as e:
            print(f"[Error] Failed to fetch {url}: {e}")

    return articles

def chunk_text(text, chunk_size=3000, overlap=500):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def ingest():
    articles = fetch_articles()
    print(f"Fetched {len(articles)} articles.")

    texts = []
    embeddings = []

    for idx, article in enumerate(articles):
        chunks = chunk_text(article)
        for i, chunk in enumerate(chunks):
            emb = get_embeddings(chunk)
            texts.append(chunk)
            embeddings.append(emb)

    if not embeddings:
        print("[Error] No embeddings generated. Aborting FAISS index creation.")
        return

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, str(INDEX_PATH))
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump({str(i): t for i, t in enumerate(texts)}, f)

    print(f"[Ingest] Ingested {len(texts)} chunks.")
    print(f"[Ingest] Saved FAISS index to: {INDEX_PATH}")
    print(f"[Ingest] Saved texts to: {TEXTS_PATH}")

if __name__ == "__main__":
    ingest()
