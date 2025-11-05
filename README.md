---
title: RAG Q&A API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# 🤖 RAG Q&A API - Intelligent Document Query System

> A production-ready Retrieval-Augmented Generation (RAG) API that answers questions using custom knowledge bases. Built to demonstrate enterprise-grade AI/ML development skills.

[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](YOUR_HUGGINGFACE_SPACE_URL)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)

---

## 🎯 Overview

This project implements a RAG system that answers questions about custom documents using natural language. It retrieves relevant context from your documents before generating answers, ensuring responses are accurate and grounded in your data.

**Built for the WebMob Technologies AI/ML Developer Trainee position**

### What is RAG?

RAG (Retrieval-Augmented Generation) combines:
1. **Retrieval**: Finding relevant document chunks using semantic search
2. **Augmentation**: Adding retrieved context to the query
3. **Generation**: Creating accurate, source-backed answers

---

## ✨ Key Features

- 🧠 **Semantic Search**: FAISS vector database for intelligent context retrieval
- ⚡ **Fast Responses**: Optimized pipeline with <2s average response time
- 🔒 **Source Attribution**: Every answer includes source passages
- 🌐 **RESTful API**: Clean API with automatic interactive documentation
- 🐳 **Docker Ready**: One-command deployment

---

## 🛠️ Technology Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Google `text-embedding-004`
- **Vector DB**: FAISS (CPU)
- **Framework**: LangChain (LCEL)
- **API**: FastAPI + Uvicorn
- **Deployment**: Docker + Hugging Face Spaces

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google API Key ([Get one here](https://aistudio.google.com/))

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_project

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo 'GOOGLE_API_KEY="your-api-key-here"' > .env

# Create the knowledge base
python ingest.py

# Run the API
uvicorn main:app --reload
```

### Using Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 rag-api
```

---

## 📖 API Usage

### Interactive Documentation
Once running, visit: **http://localhost:8000/docs**

### Example Request

**Endpoint**: `POST /ask`

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?"
  }'
```

**Response**:
```json
{
  "question": "What is this document about?",
  "answer": "This document discusses...",
  "source_documents": [
    "Original text chunk 1...",
    "Original text chunk 2..."
  ]
}
```

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/ask` | Submit a question and get an answer |
| GET | `/docs` | Interactive API documentation |

---

## 📁 Project Structure

```
rag_project/
├── main.py              # FastAPI application & RAG chain
├── ingest.py            # Document processing & indexing
├── data.txt             # Your knowledge base document
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration
├── .env                 # API keys (not committed)
└── faiss_index/         # Vector database (generated)
```

---

## 🔧 Configuration

### Customize Retrieval
In `main.py`, adjust the retriever:
```python
retriever = db.as_retriever(search_kwargs={"k": 3})  # Return top 3 results
```

### Adjust Model Temperature
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,  # Lower = more focused, Higher = more creative
)
```

### Change Chunk Size
In `ingest.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=100   # Overlap between chunks
)
```

---

## 📊 Performance

- **Average Response Time**: <2 seconds
- **Embedding Model**: 768-dimensional vectors
- **Vector Search**: FAISS L2 similarity
- **Chunk Strategy**: 1000 chars with 100 char overlap

---

## 🤝 Skills Demonstrated

This project showcases:
- ✅ **Generative AI**: LLM integration and prompt engineering
- ✅ **Vector Databases**: Semantic search with FAISS
- ✅ **API Development**: RESTful design with FastAPI
- ✅ **ML Engineering**: Data preprocessing and pipeline optimization
- ✅ **DevOps**: Containerization and cloud deployment
- ✅ **Best Practices**: Code structure, documentation, version control

---

## 🐛 Troubleshooting

**Issue**: `API key not found`
- **Solution**: Ensure `.env` file exists with `GOOGLE_API_KEY="your-key"`

**Issue**: `faiss_index not found`
- **Solution**: Run `python ingest.py` first to create the index

**Issue**: `Module not found`
- **Solution**: Install all dependencies: `pip install -r requirements.txt`

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

**Built with ❤️ for WebMob Technologies AI/ML Developer Trainee Position**