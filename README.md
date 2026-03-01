# PDF Research Paper Chatbot with RAG (OpenRouter)

<p align="center">
  <a href="https://pdf-researchpaper-bot-1.onrender.com" target="_blank" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

A modern, AI-powered chatbot that lets you upload academic PDFs and ask questions about their content. Uses Retrieval-Augmented Generation (RAG) with keyword-based retrieval and OpenRouter LLMs for accurate, context-aware answers.

---

## 🚀 Features

- **PDF Upload & Parsing:** Extracts text from uploaded PDFs using PyMuPDF (in-memory, no disk writes).
- **Text Chunking:** Splits content into overlapping chunks for efficient retrieval.
- **Keyword Retrieval:** Finds relevant chunks using simple keyword matching (no embeddings required).
- **RAG Pipeline:** Sends top-matching chunks as context to the LLM.
- **OpenRouter Integration:** Uses free models (`openrouter/free`, `stepfun/step-3.5-flash:free`) with automatic fallback; supports paid models (e.g., GPT-4o-mini) if you have credits.
- **Modern Web UI:** Clean, responsive interface with bullet-pointed answers.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **PDF Extraction:** PyMuPDF
- **Retrieval:** Keyword-based (no vector DB)
- **LLM API:** OpenRouter
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
   - Get a free API key at [OpenRouter](https://openrouter.ai/).

---

## ⚡ Usage

1. **Start the app:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:15000/
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
curl -X POST http://127.0.0.1:15000/ask \
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
├── Procfile
├── .env.example
├── .gitignore
├── templates/
│   └── index.html
└── README.md
```

---

## 🔒 Security & Credits

- **API keys:** Never commit your real `.env` file. Use `.env.example` for sharing variable names.
- **OpenRouter:** Free models work without credits. For paid models (e.g., GPT-4o-mini), add credits at [OpenRouter](https://openrouter.ai/). Lower `max_tokens` in `rag_pipeline.py` if you hit quota errors.
