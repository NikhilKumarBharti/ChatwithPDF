# ChatWithPDF: A Conversational PDF Assistant

ChatWithPDF is a lightweight, extensible Flask-based web application that allows users to **upload PDF documents** and **chat** with their contents using a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain, FAISS, HuggingFace Embeddings, and the OpenRouter API.

---

## ✨ Features

* ✉ Upload and parse PDF files.
* 🤖 Chat with the content using stateful, context-aware conversations.
* ⚙ Built with a modular architecture using LangChain components.
* ⚖ Uses **FAISS** for efficient vector search and **HuggingFace embeddings**.
* 🔑 Supports secure API access via environment variables.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
FLASK_SECRET_KEY=your_flask_secret
```

You may also customize these:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=3
EMBEDDING_DEVICE=cpu  # or cuda
```

### 4. Run the Application

```bash
python appy.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 🧹 API Endpoints

| Endpoint  | Method | Description                               |
| --------- | ------ | ----------------------------------------- |
| `/`       | GET    | Homepage with upload form                 |
| `/upload` | POST   | Upload a PDF file                         |
| `/chat`   | POST   | Ask questions about the uploaded document |
| `/health` | GET    | Health check and readiness status         |

---

## 🧱 Docker Setup

### 1. Create `Dockerfile`

```dockerfile
# Use an official Python base image
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "appy.py"]
```

### 2. Build and Run

```bash
docker build -t chat-with-pdf .
docker run -p 5000:5000 --env-file .env chat-with-pdf
```

---

## 🎓 Technologies Used

* **Flask**: Web framework
* **LangChain**: Conversational RAG pipeline
* **FAISS**: Vector search index
* **HuggingFace Transformers**: Embeddings
* **OpenRouter API**: LLM responses
* **dotenv**: Secure config management

---

## 📚 License

This project is licensed under the MIT License.

---

## ✨ Acknowledgments

* [LangChain](https://www.langchain.com/)
* [OpenRouter](https://openrouter.ai)
* [HuggingFace](https://huggingface.co/)
* [FAISS](https://github.com/facebookresearch/faiss)
