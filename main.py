"""
FastAPI Chatbot Service with RAG (Retrieval-Augmented Generation)
Uses Google Gemini for LLM and Pinecone/FAISS for vector database
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Import custom modules (will be created)
from app.services.gemini_service import GeminiService
from app.services.vector_db_service import VectorDBService
from app.services.rag_service import RAGService
from app.database.db import get_db_connection
from app.database.chat_history_db import init_chat_history_table
from app.models.schemas import ChatRequest, ChatResponse, ChatStreamChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "false").lower() == "true" else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
gemini_service = None
vector_db_service = None
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global gemini_service, vector_db_service, rag_service
    
    logger.info("Initializing Chatbot Service...")
    
    # Initialize chat history table
    await init_chat_history_table()
    
    # Initialize services
    gemini_service = GeminiService()
    vector_db_service = VectorDBService()
    rag_service = RAGService(gemini_service, vector_db_service)
    
    # Load initial data into vector database
    await vector_db_service.initialize()
    
    logger.info("Chatbot Service initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Chatbot Service...")
    await vector_db_service.close()


# Create FastAPI app
app = FastAPI(
    title="GearUp AI Chatbot Service",
    description="RAG-based chatbot for appointment queries using Gemini LLM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "GearUp AI Chatbot",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gemini": gemini_service.is_available(),
        "vector_db": vector_db_service.is_available(),
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process chat request with RAG
    Returns complete response
    """
    try:
        logger.info(f"Chat request: {request.question[:50]}...")
        
        response = await rag_service.process_query(
            question=request.question,
            session_id=request.sessionId,
            conversation_history=request.conversationHistory,
            filters={
                "appointment_date": request.appointmentDate,
                "service_type": request.serviceType
            },
            customer_id=getattr(request, 'customerId', None),
            customer_email=getattr(request, 'customerEmail', None),
            auth_token=getattr(request, 'authToken', None)
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Process chat request with streaming response (SSE)
    Returns incremental chunks
    """
    try:
        logger.info(f"Stream chat request: {request.question[:50]}...")
        
        async def generate_stream():
            async for chunk in rag_service.process_query_stream(
                question=request.question,
                session_id=request.sessionId,
                conversation_history=request.conversationHistory,
                filters={
                    "appointment_date": request.appointmentDate,
                    "service_type": request.serviceType
                }
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Send final marker
            yield "data: {\"is_final\": true}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Error processing stream chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/update")
async def update_embeddings():
    """
    Manually trigger embeddings update from database
    Should be called when appointment data changes
    """
    try:
        logger.info("Manually updating embeddings...")
        await vector_db_service.update_from_database()
        return {"status": "success", "message": "Embeddings updated successfully"}
    
    except Exception as e:
        logger.error(f"Error updating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 10):
    """
    Get chat history for a session
    
    Args:
        session_id: The session ID to retrieve history for
        limit: Maximum number of messages to return (default: 10)
    
    Returns:
        List of chat messages with questions and answers
    """
    try:
        logger.info(f"Retrieving chat history for session: {session_id}")
        history = await rag_service.get_chat_history(session_id, limit)
        return history
    
    except Exception as e:
        logger.error(f"Error getting chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a session
    
    Args:
        session_id: The session ID to clear history for
    
    Returns:
        Success message
    """
    try:
        logger.info(f"Clearing chat history for session: {session_id}")
        await rag_service.clear_chat_history(session_id)
        return {"status": "success", "message": f"Chat history cleared for session {session_id}"}
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions")
async def get_chat_sessions(
    customer_email: Optional[str] = None,
    user_id: Optional[int] = None,
    limit: int = 20
):
    """
    Get chat sessions for a customer

    Args:
        customer_email: Customer email to filter sessions (for display purposes)
        user_id: User ID to filter sessions (for actual filtering)
        limit: Maximum number of sessions to return

    Returns:
        List of chat sessions
    """
    try:
        logger.info(f"Getting chat sessions for customer: {customer_email}, user_id: {user_id}")

        # Import here to avoid circular imports
        from app.database.chat_history_db import get_recent_sessions

        # Filter by user_id if provided for security
        sessions = await get_recent_sessions(limit=limit, user_id=user_id)
        
        # For now, return basic session info
        # In production, you'd want to include session metadata, titles, etc.
        session_list = []
        from app.database.chat_history_db import get_chat_history
        
        for session_id in sessions:
            # Get first message efficiently (limit=3 to handle potential empty messages)
            history = await get_chat_history(session_id, limit=3)
            
            title = "New Chat"
            created_at = None
            if history:
                # Find first message with actual content
                for msg in history:
                    question = msg.get('question', '').strip()
                    if question:  # Skip empty placeholder messages
                        title = question[:50] + ('...' if len(question) > 50 else '')
                        break
                
                # Use the created_at from the first message for chronological order
                created_at = history[0].get('created_at')
            
            session_list.append({
                "sessionId": session_id,
                "title": title,
                "createdAt": created_at,
                "customerEmail": customer_email
            })
        
        return {
            "sessions": session_list,
            "total": len(session_list)
        }
    
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/sessions")
async def create_chat_session(request: dict):
    """
    Create a new chat session
    
    Args:
        request: Dictionary with customerEmail and title
    
    Returns:
        New session information
    """
    try:
        import uuid
        from datetime import datetime
        
        customer_email = request.get("customerEmail")
        title = request.get("title", "New Chat")
        
        logger.info(f"Creating new chat session for customer: {customer_email}")
        
        # Generate new session ID
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Save initial session record to database so it appears in history
        # We'll save a placeholder message to create the session record
        from app.database.chat_history_db import save_chat_message
        await save_chat_message(
            session_id=session_id,
            question="",  # Empty question for session initialization
            answer="",    # Empty answer for session initialization
            confidence_score=1.0,
            from_cache=False,
            processing_time_ms=0,
            user_id=None
        )
        
        logger.info(f"Session created and saved: {session_id}")
        
        return {
            "sessionId": session_id,
            "title": title,
            "createdAt": created_at.isoformat(),
            "customerEmail": customer_email
        }
    
    except Exception as e:
        logger.error(f"Error creating chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """
    Delete a chat session and its history
    
    Args:
        session_id: Session ID to delete
    
    Returns:
        Success message
    """
    try:
        logger.info(f"Deleting chat session: {session_id}")
        
        from app.database.chat_history_db import clear_chat_history
        deleted_count = await clear_chat_history(session_id)
        
        return {
            "status": "success",
            "message": f"Chat session {session_id} deleted",
            "deletedMessages": deleted_count
        }
    
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    try:
        stats = await rag_service.get_statistics()
        return stats
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
