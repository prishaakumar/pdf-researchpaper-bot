# PDF Research Paper Chatbot with RAG (OpenRouter + HuggingFace)

<p align="center">
  <a href="https://your-demo-link-here.com" target="_blank" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

A modern, AI-powered chatbot that lets you upload academic PDFs and ask questions about their content. Uses Retrieval-Augmented Generation (RAG) with local HuggingFace embeddings, FAISS vector search, and OpenRouter LLMs (e.g., GPT-4o) for accurate, context-aware answers.

---

## 🚀 Features

- **PDF Upload & Parsing:** Extracts text from uploaded PDFs using PyMuPDF.
- **Text Chunking & Embedding:** Splits content into overlapping chunks and embeds them using HuggingFace's `all-MiniLM-L6-v2`.
- **Vector Search:** Stores embeddings in FAISS for fast semantic retrieval.
- **RAG Pipeline:** Retrieves top-matching chunks as context for the LLM.
- **LLM Integration:** Uses OpenRouter API (e.g., GPT-4o) for answer generation.
- **Modern Web UI:** Clean, responsive interface with beautiful bullet-pointed answers.

---


## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **PDF Extraction:** PyMuPDF
- **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS
- **LLM API:** OpenRouter (e.g., GPT-4o)
- **Frontend:** HTML/CSS

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-research-paper-chatbot.git
   cd pdf-research-paper-chatbot
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   - Copy `.env.example` to `.env` and add your OpenRouter API key:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key_here
     ```

---

## ⚡ Usage

1. **Start the app:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```

3. **Upload a PDF and ask a question!**
   - Example PDF: [Attention Is All You Need (arXiv)](https://arxiv.org/pdf/1706.03762.pdf)
   - Example question:  
     > What problem do Transformers solve?

---

## 📝 Example Questions

- What problem do Transformers solve?
- How does self-attention work in the Transformer model?
- What are the main contributions of this paper?
- How does the Transformer architecture differ from RNNs?

---

## 🧪 API Usage

You can also use the `/ask` endpoint programmatically:

```bash
curl -X POST http://127.0.0.1:5000/ask \
  -F "pdf=@/path/to/your/research_paper.pdf" \
  -F "question=What problem do Transformers solve?"
```

---

## 🗂️ Project Structure

```
pdf-research-paper-chatbot/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
├── templates/
│   └── index.html
├── uploads/           # (auto-created, gitignored)
└── README.md
```

---

## 🔒 Security & Credits

- **API keys:** Never commit your real `.env` file. Use `.env.example` for sharing variable names.
- **OpenRouter credits:** Free accounts have token limits. Lower `max_tokens` in `rag_pipeline.py` if you hit quota errors.


