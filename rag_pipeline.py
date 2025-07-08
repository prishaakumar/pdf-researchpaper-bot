from dotenv import load_dotenv
load_dotenv()
import fitz  # PyMuPDF
import faiss
import requests
import numpy as np
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Hugging Face model
LLM_MODEL = 'openai/gpt-4o'

# --- PDF PARSING ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- CHUNKING ---
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- EMBEDDING (Hugging Face) ---
model = SentenceTransformer(EMBEDDING_MODEL)
def embed_texts(texts):
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)

# --- FAISS VECTOR STORE ---
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- RETRIEVAL ---
def retrieve(query, chunks, index, chunk_embeddings, top_k=5):
    query_emb = embed_texts([query])[0].reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]]

# --- LLM CALL (OpenRouter) ---
def call_llm_openrouter(context, query):
    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }
    prompt = f"Context from PDF:\n{context}\n\nQuestion: {query}\nAnswer:"
    data = {
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful academic research assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 512,
        'temperature': 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("OpenRouter error details:", response.text)
        raise
    return response.json()['choices'][0]['message']['content']

# --- END-TO-END PIPELINE ---
def process_pdf_and_answer(pdf_path, user_query):
    print('Extracting text from PDF...')
    text = extract_text_from_pdf(pdf_path)
    print('Chunking text...')
    chunks = chunk_text(text)
    print(f'Number of chunks: {len(chunks)}')
    print('Embedding chunks...')
    chunk_embeddings = embed_texts(chunks)
    print('Building FAISS index...')
    index = build_faiss_index(chunk_embeddings)
    print('Retrieving relevant chunks...')
    top_chunks = retrieve(user_query, chunks, index, chunk_embeddings)
    context = '\n---\n'.join(top_chunks)
    print('Calling LLM via OpenRouter...')
    answer = call_llm_openrouter(context, user_query)
    return answer 