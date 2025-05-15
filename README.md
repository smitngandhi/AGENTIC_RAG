# Agentic RAG Toolkit 📚🧠

This repository contains modular and experimental implementations of **Agentic Retrieval-Augmented Generation (RAG)** pipelines using [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [HuggingFace](https://huggingface.co/), and models from **OpenAI**, **Together.ai**, and **GROQ**.

---

## 🔧 Project Structure

.
├── agents/
│ ├── agents.ipynb ← General Agentic RAG pipeline with Wikipedia, ArXiv, and LangSmith tools
│ └── ai_agents.ipynb ← Custom RAG agent for PCOS/PMS/PCOD with multiple vector sources
│
├── groq/
│ ├── app.py ← Streamlit app using Groq-hosted LLaMA model with vector retrieval
│ └── groq.ipynb ← Jupyter notebook version of the same
│
├── huggingface/
│ ├── tf_papers/ ← Open-source Transformer-related research PDFs
│ └── hugging.ipynb ← RAG pipeline using huggingface models and academic sources
│
├── .env ← Environment variables
├── venv/ ← Python virtual environment
└── README.md ← You're here!


---

## 🧠 agents/ - Agentic Retrieval-Augmented Generation

### `agents.ipynb`

An AI agent powered by LangChain that can:

- Search and retrieve documents from:
  - 🔗 [LangSmith documentation](https://docs.smith.langchain.com/)
  - 📚 Wikipedia
  - 📄 ArXiv research papers
- Use `FAISS` for vector storage and similarity search
- Generate responses with Together.ai’s **LLaMA 3.3 70B** model

**Sample Pipeline:**
```python
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

# Embedding and storing
vectordb = FAISS.from_documents(documents, HuggingFaceEmbeddings(...))

# Tools
retriever_tool = create_retriever_tool(vectordb.as_retriever(), ...)
arxiv = ArxivQueryRun(...)
wikipedia = WikipediaQueryRun(...)
tools = [retriever_tool, arxiv, wikipedia]

# Agent
llm = ChatOpenAI(...)
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

## 📁 Project Structure

### `ai_agents.ipynb`  
A **domain-specific agent** trained for PCOS/PMS/PCOD topics using:

🔍 **Multiple FAISS vector DBs** for:
- Clinical guidelines  
- Government health policies  
- Research publications  
- Blogs and public health education  

🔗 **External tools** integrated:
- ArXiv  
- Wikipedia  
- Google Search  

---

### 💻 `groq/` - Streamlit + Groq-Powered RAG

#### `app.py`  
An **interactive Streamlit app** that:
- Loads content (e.g., from Wikipedia: *Virat Kohli*)  
- Chunks and embeds using HuggingFace Transformers  
- Stores vectors in a FAISS database  
- Answers queries using Groq-hosted **LLaMA 3.3-70B** with a custom prompt  

✅ **Features:**
- Real-time query handling  
- Context-aware LLM responses  
- Execution time tracking  

---

### 🤗 `huggingface/` - Open Source RAG from Research

#### `hugging.ipynb`  
Implements the **same RAG architecture** to:
- Process and embed papers in `tf_papers/`  
- Perform **document similarity search**  
- Answer questions on Transformer-based literature  

📚 Uses research papers like:
- *Attention is All You Need*  
- *Masked Attention Transformers*  
- *TransUNet*, etc.  

This pipeline is **fully open-source**, ideal for academic and reproducible research.

---

## 📦 Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag

# Activate virtual environment
.\venv\Scripts\activate    # Windows PowerShell
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Inside .env
OPENAI_API_TOGETHER_API=your_openai_or_together_api_key
GROQ_API_KEY=your_groq_api_key
HUGGING_FACE_API = your_hugging_face_api
