# 🧠 Agentic RAG Toolkit

A modular, research-driven framework for building **Agentic Retrieval-Augmented Generation (RAG)** pipelines using tools such as **LangChain**, **FAISS**, **Hugging Face**, **OpenAI**, **Together.ai**, and **Groq**.

This toolkit is designed to:
- Enable **multi-source document retrieval**
- Integrate **custom vector databases**
- Empower agents with **domain-specific intelligence**
- Support both **notebook-based** experimentation and **app-based** deployment

---

## 📁 Project Structure

```plaintext
.
├── agents/
│   ├── agents.ipynb         # General Agentic RAG pipeline using Wikipedia, ArXiv, and LangSmith
│   └── ai_agents.ipynb      # Custom domain-specific agent for PCOS/PMS/PCOD
│
├── groq/
│   ├── app.py               # Streamlit app using Groq-hosted LLaMA 3.3 70B
│   └── groq.ipynb           # Notebook version of the same app
│
├── huggingface/
│   ├── tf_papers/           # Open-source Transformer research papers
│   └── hugging.ipynb        # RAG pipeline using Hugging Face models and academic sources
│
├── .env                     # API keys and environment variables
├── venv/                    # Python virtual environment
└── README.md                # You're here!
```

---

## 🧠 `agents/` – General & Domain-Specific RAG Agents

### `agents.ipynb` – General-Purpose Agentic RAG

This notebook demonstrates an **AI assistant pipeline** built using `LangChain`, designed to retrieve and synthesize information from:

- 🌐 [LangSmith Documentation](https://docs.smith.langchain.com/)
- 📚 Wikipedia
- 📄 ArXiv scientific papers

### 🔧 Core Components:
- **WebBaseLoader**: Scrapes and loads content from online sources
- **TextSplitter**: Splits text into manageable chunks
- **FAISS**: Vector store for embedding and fast retrieval
- **Retriever Tools**: Integrates Wikipedia, ArXiv, and LangSmith into the agent
- **Together.ai LLaMA 3.3-70B**: Generates coherent answers based on the retrieved context



## ⚙️ `groq/` – Real-Time RAG with Streamlit

### `app.py`

Interactive Streamlit application with the following capabilities:
- Loads content dynamically (e.g., *Virat Kohli* on Wikipedia)
- Chunks, embeds, and stores vectors in FAISS
- Queries Groq-hosted **LLaMA 3.3-70B** for context-aware responses

### ✅ Features
- Real-time LLM-powered querying
- Response context tracing
- Execution time monitoring

---

## 🤗 `huggingface/` – Research-Oriented RAG

### `hugging.ipynb`

- Embeds open-source research papers located in `tf_papers/`
- Supports document similarity and Q&A over papers like:
  - *Attention is All You Need*
  - *TransUNet*
  - *Masked Attention Transformers*

---

## 📦 Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag

# Activate virtual environment
.env\Scriptsctivate    # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 🔐 Environment Variables (.env)
```env
OPENAI_API_KEY=your_openai_or_together_api_key
GROQ_API_KEY=your_groq_api_key
HUGGING_FACE_API=your_hugging_face_api
```
