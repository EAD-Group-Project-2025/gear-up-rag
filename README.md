# 🤖 GearUp AI Chatbot Service

Advanced Python FastAPI service implementing RAG (Retrieval-Augmented Generation) architecture with Google Gemini LLM for intelligent vehicle service appointment assistance.

## 🏗️ Architecture Overview

This service implements a sophisticated RAG pipeline combining:
- **Vector Database**: Semantic search through FAISS/Pinecone
- **LLM Integration**: Google Gemini 2.0 Flash for response generation
- **Contextual Retrieval**: Dynamic document filtering and ranking
- **Streaming Interface**: Real-time SSE responses
- **Persistent Storage**: PostgreSQL for chat history and appointment data

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   ├────┤   RAG Service    ├────┤  Gemini LLM     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       
         │                       ▼                       
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Chat History   ├────┤ Vector Database  ├────┤   Embeddings    │
│   PostgreSQL    │    │  (FAISS/Pinecone)│    │  SentenceT5     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

### 🔍 RAG Implementation
- **Semantic Search**: Uses `all-MiniLM-L6-v2` embeddings (384-dim)
- **Context Retrieval**: Top-K document matching with relevance scoring
- **Dynamic Filtering**: Appointment date and service type filters
- **Confidence Scoring**: Automatic response confidence calculation

### 🧠 LLM Integration
- **Google Gemini 2.0 Flash**: Latest model with streaming support
- **Safety Controls**: Built-in content filtering and safety settings
- **Conversation Memory**: Multi-turn conversation handling
- **Prompt Engineering**: Optimized system instructions for vehicle services

### 📊 Vector Database Options
- **FAISS (Local)**: High-performance similarity search, persistent storage
- **Pinecone (Cloud)**: Managed vector database with serverless scaling
- **Dual Support**: Runtime switching between vector database providers

### 🌊 Streaming Architecture
- **Server-Sent Events**: Real-time response streaming
- **Chunked Processing**: Incremental content delivery
- **Session Management**: Persistent conversation tracking

## ⚙️ Environment Configuration

**IMPORTANT**: The application no longer uses hardcoded localhost URLs. All integration URLs must be configured via environment variables.

### Required Environment Variables

```bash
# AI Service - REQUIRED for chatbot functionality
GEMINI_API_KEY=your_gemini_api_key_here

# Spring Boot Integration - REQUIRED
SPRING_BOOT_BASE_URL=http://backend:8080/api/v1

# CORS Configuration - REQUIRED
CORS_ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:3000
```

### Production Deployment

For production environments, ensure all URLs use proper domain names:

```bash
# Production example
SPRING_BOOT_BASE_URL=https://api.yourdomain.com/api/v1
CORS_ALLOWED_ORIGINS=https://yourdomain.com
GEMINI_API_KEY=your_production_api_key
USE_PINECONE=true
PINECONE_API_KEY=your_pinecone_key
DEBUG=false
```

See [Environment Variables Guide](../docs/ENVIRONMENT_VARIABLES.md) for complete configuration details.

### 📈 Performance Features
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Caching Strategy**: Optimized embedding storage and retrieval
- **Error Handling**: Comprehensive exception management

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 12+
- CUDA (optional, for GPU acceleration)

### 1. Clone Repository
```bash
git clone https://github.com/EAD-Group-Project-2025/gear-up-rag.git
cd gear-up-rag
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and configure:

```env
# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost/gearup_db

# Vector Database (Choose one)
USE_PINECONE=false
PINECONE_API_KEY=your_pinecone_key  # If using Pinecone
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=gearup-chatbot

# FAISS Configuration (Local)
FAISS_INDEX_PATH=./data/faiss_index

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_CONTEXT_DOCS=5

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### 4. Database Setup
```bash
# Create PostgreSQL database
createdb gearup_db

# Run migrations (if using Alembic)
alembic upgrade head
```

### 5. Launch Service
```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

## 📡 API Reference

### Core Endpoints

#### Chat Completion
```http
POST /chat
Content-Type: application/json

{
  "question": "When is my next appointment?",
  "sessionId": "user_123",
  "conversationHistory": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ],
  "appointmentDate": "2024-12-01",
  "serviceType": "oil_change",
  "customerId": 12345,
  "customerEmail": "user@example.com",
  "authToken": "jwt_token_here"
}
```

**Response:**
```json
{
  "answer": "Your next appointment is scheduled for...",
  "session_id": "user_123",
  "from_cache": false,
  "processing_time_ms": 1250,
  "timestamp": "2024-11-01T10:30:00Z",
  "confidence": 0.92,
  "sources": ["appointment_api", "service_catalog"]
}
```

#### Streaming Chat
```http
POST /chat/stream
Content-Type: application/json

{
  "question": "Tell me about brake service options",
  "sessionId": "user_123"
}
```

**Response (SSE):**
```
data: {"content": "I'd be happy to", "is_final": false, "session_id": "user_123", "chunk_index": 0}

data: {"content": " help you with brake service options...", "is_final": false, "session_id": "user_123", "chunk_index": 1}

data: {"content": "", "is_final": true, "session_id": "user_123", "chunk_index": 2}
```

### Management Endpoints

#### Update Vector Embeddings
```http
POST /embeddings/update
```
Manually refresh vector database from PostgreSQL appointment data.

#### Chat History
```http
# Get history
GET /chat/history/{session_id}?limit=10

# Clear history
DELETE /chat/history/{session_id}
```

#### Service Statistics
```http
GET /stats
```
Returns RAG service performance metrics and availability status.

## 🔧 Technical Deep Dive

- Interactive API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
chatbot-service/
├── app/
│   ├── services/
│   │   ├── gemini_service.py      # Gemini LLM integration
│   │   ├── vector_db_service.py   # Vector database
│   │   └── rag_service.py         # RAG pipeline
│   ├── database/
│   │   └── db.py                  # Database utilities
│   └── models/
│       └── schemas.py             # Pydantic models
├── main.py                        # FastAPI application
├── requirements.txt               # Python dependencies
└── Dockerfile                     # Docker configuration
```

## Vector Database Options

### FAISS (Local - Default)
```env
USE_FAISS=true
FAISS_INDEX_PATH=./data/faiss_index
```

### Pinecone (Cloud)
```env
USE_PINECONE=true
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=gearup-chatbot
```

## Testing

```bash
# Test health
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What services do you offer?"}'
```

## Docker

```bash
# Build
docker build -t gearup-chatbot .

# Run
docker run -p 8000:8000 --env-file .env gearup-chatbot
```

## License

MIT
