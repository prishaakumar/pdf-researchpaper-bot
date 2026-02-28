from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
import os
import requests

# --- CONFIG ---
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Hugging Face model
# Models tried in order; free ones first, paid last (requires credits)


# --- PDF PARSING ---
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file on disk."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_pdf_bytes(pdf_bytes: bytes):
    """Extract text from an in-memory PDF byte stream."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# --- SIMPLE CHUNKING (without FAISS/vector search) ---
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# --- SIMPLE RETRIEVAL (keyword-based instead of vector search) ---
def retrieve_simple(query, chunks, top_k=3):
    """Simple keyword-based retrieval instead of vector search."""
    scores = []
    query_lower = query.lower()

    for i, chunk in enumerate(chunks):
        score = 0
        chunk_lower = chunk.lower()
        for word in query_lower.split():
            if not word:
                continue
            score += chunk_lower.count(word)
        scores.append((score, i))

    scores.sort(reverse=True)
    top_indices = [idx for _, idx in scores[:top_k]]
    return [chunks[i] for i in top_indices]


# --- REAL LLM CALL (OpenRouter) ---
# Try these models in order; free models can be flaky (500), so we retry and fall back
LLM_MODELS = [
    "openrouter/free",
    "stepfun/step-3.5-flash:free",
    "openai/gpt-4o-mini",
]


def call_llm_openrouter(context, query):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file as OPENROUTER_API_KEY=..."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "PDF Research Paper Chatbot",
    }

    # Cap context to avoid token limits / 500 errors
    max_context_chars = 6000
    context_trimmed = context[:max_context_chars] + ("..." if len(context) > max_context_chars else "")

    prompt = (
        "You are a helpful academic research assistant. "
        "Use ONLY the provided PDF context to answer the user's question. "
        "If the answer is not in the context, say you are not sure.\n\n"
        f"Context from PDF:\n{context_trimmed}\n\nQuestion: {query}\nAnswer:"
    )

    last_error = None
    for model in LLM_MODELS:
        for attempt in range(2):
            try:
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.2,
                }
                resp = requests.post(url, headers=headers, json=data, timeout=60)
                if resp.status_code == 200:
                    body = resp.json()
                    return body["choices"][0]["message"]["content"]
                try:
                    err_body = resp.json()
                    msg = err_body.get("error", {}).get("message", resp.text[:300])
                except Exception:
                    msg = resp.text[:300] if resp.text else f"Status {resp.status_code}"
                last_error = f"OpenRouter API failed ({resp.status_code}): {msg}"
                print(f"OpenRouter ({model}) attempt {attempt + 1}:", last_error)
                if resp.status_code == 500 and attempt < 1:
                    import time
                    time.sleep(2)
                    continue
                break
            except requests.RequestException as e:
                last_error = str(e)
                print(f"OpenRouter ({model}) attempt {attempt + 1} request error:", last_error)
                if attempt < 1:
                    import time
                    time.sleep(2)
    raise RuntimeError(last_error or "OpenRouter API failed.")


# --- END-TO-END PIPELINE ---
def process_pdf_and_answer(pdf_path, user_query):
    try:
        print(f"process_pdf_and_answer called with pdf_path={pdf_path}")
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        print("Chunking text...")
        chunks = chunk_text(text)
        print(f"Number of chunks: {len(chunks)}")
        print("Retrieving relevant chunks...")
        top_chunks = retrieve_simple(user_query, chunks)
        context = "\n---\n".join(top_chunks)
        print("Calling LLM via OpenRouter...")
        answer = call_llm_openrouter(context, user_query)
        return answer
    except Exception as e:
        print(f"Error in process_pdf_and_answer: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"An error occurred while processing your request: {str(e)}"


def process_pdf_bytes_and_answer(pdf_bytes: bytes, user_query: str):
    """End-to-end pipeline for an uploaded PDF provided as bytes."""
    try:
        print("process_pdf_bytes_and_answer called")
        print("Extracting text from PDF (bytes)...")
        text = extract_text_from_pdf_bytes(pdf_bytes)
        print("Chunking text...")
        chunks = chunk_text(text)
        print(f"Number of chunks: {len(chunks)}")
        print("Retrieving relevant chunks...")
        top_chunks = retrieve_simple(user_query, chunks)
        context = "\n---\n".join(top_chunks)
        print("Calling LLM via OpenRouter...")
        answer = call_llm_openrouter(context, user_query)
        return answer
    except Exception as e:
        print(f"Error in process_pdf_bytes_and_answer: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"An error occurred while processing your request: {str(e)}"