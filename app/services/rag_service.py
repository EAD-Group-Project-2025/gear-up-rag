"""RAG (Retrieval-Augmented Generation) Service"""

import logging
import time
import uuid
import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from app.services.gemini_service import GeminiService
from app.services.vector_db_service import VectorDBService
from app.services.appointment_service import appointment_service
from app.services.function_orchestrator import function_orchestrator
from app.models.schemas import ChatResponse, ChatStreamChunk, ChatMessage
from app.database.chat_history_db import (
    save_chat_message,
    get_chat_history as db_get_chat_history,
    clear_chat_history as db_clear_chat_history
)

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(self, gemini_service: GeminiService, vector_db_service: VectorDBService):
        self.gemini_service = gemini_service
        self.vector_db_service = vector_db_service
        self.max_context_docs = int(os.getenv("MAX_CONTEXT_DOCS", "5"))
    
    async def process_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        filters: Optional[Dict[str, Any]] = None,
        customer_id: Optional[int] = None,
        customer_email: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> ChatResponse:
        """
        Process user query with RAG
        
        Args:
            question: User question
            session_id: Session identifier
            conversation_history: Previous conversation messages
            filters: Additional filters for context retrieval
            customer_id: Customer ID for appointment queries
            customer_email: Customer email for appointment queries
            auth_token: JWT token for API authentication
        
        Returns:
            ChatResponse with answer and metadata
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())

        try:
            # Use Gemini with function calling to understand user intent
            logger.info(f"Processing query with function calling: {question[:50]}...")

            gemini_response = await self.gemini_service.generate_response_with_functions(
                prompt=question,
                conversation_history=self._format_history(conversation_history) if conversation_history else None
            )

            # Handle function calls from Gemini
            if gemini_response["type"] == "function_call":
                function_name = gemini_response["function_name"]
                function_args = gemini_response["function_args"]

                logger.info(f"Gemini requested function call: {function_name}")

                # Execute the function
                function_result = await function_orchestrator.execute_function(
                    function_name=function_name,
                    function_args=function_args,
                    customer_id=customer_id,
                    customer_email=customer_email,
                    auth_token=auth_token
                )

                # Format the result for the user
                if function_result.get("success"):
                    # Generate natural language response with the data
                    if function_name == "get_user_appointments":
                        appointments = function_result.get("data", [])
                        if appointments:
                            # Format appointments nicely
                            appointment_context = "USER'S APPOINTMENTS:\n\n"
                            for apt in appointments:
                                apt_date = apt.get("appointmentDate", "Unknown")
                                apt_time = apt.get("startTime", "")
                                apt_vehicle = apt.get("vehicleName", "Unknown vehicle")
                                apt_issue = apt.get("customerIssue", "No issue description")
                                apt_status = apt.get("status", "Unknown")

                                appointment_context += f"- Date: {apt_date}"
                                if apt_time:
                                    appointment_context += f" at {apt_time}"
                                appointment_context += f"\n  Vehicle: {apt_vehicle}\n"
                                appointment_context += f"  Issue: {apt_issue}\n"
                                appointment_context += f"  Status: {apt_status}\n\n"

                            answer = f"Here are your appointments:\n\n{appointment_context}"
                        else:
                            answer = function_result.get("message", "You don't have any appointments.")

                    elif function_name == "get_user_vehicles":
                        vehicles = function_result.get("data", [])
                        if vehicles:
                            vehicle_list = []
                            for i, v in enumerate(vehicles, 1):
                                year = v.get('year', '')
                                make = v.get('make', 'Unknown')
                                model = v.get('model', 'Unknown')
                                v_id = v.get('id', 'N/A')
                                license_plate = v.get('licensePlate', '')

                                vehicle_str = f"{i}. {year} {make} {model}"
                                if license_plate:
                                    vehicle_str += f" ({license_plate})"
                                vehicle_str += f" - ID: {v_id}"
                                vehicle_list.append(vehicle_str)

                            answer = f"Your vehicles:\n\n" + "\n".join(vehicle_list)
                        else:
                            answer = function_result.get("message", "You don't have any vehicles registered.")

                    elif function_name == "book_appointment":
                        appointment_data = function_result.get("data", {})
                        answer = f"✅ Appointment booked successfully!\n\n"
                        answer += f"Date: {appointment_data.get('appointmentDate')}\n"
                        answer += f"Time: {appointment_data.get('startTime')}\n"
                        answer += f"Vehicle: {appointment_data.get('vehicleName')}\n"
                        answer += f"Service: {appointment_data.get('consultationTypeLabel')}\n"
                        answer += f"Status: {appointment_data.get('status')}"

                    else:
                        answer = "Function executed successfully."

                else:
                    # Error handling with context-aware messages
                    error_msg = function_result.get("error", "Unknown error")

                    # Provide helpful guidance based on error type
                    if "vehicle" in error_msg.lower() or "vehicle_id" in error_msg.lower():
                        # Vehicle-related error
                        if "Invalid vehicle ID" in error_msg:
                            answer = f"❌ {error_msg}\n\nPlease select a vehicle from your list."
                        elif "No vehicles found" in error_msg:
                            answer = "❌ You don't have any vehicles registered yet.\n\nPlease add a vehicle to your account before booking an appointment."
                        else:
                            answer = f"❌ Vehicle error: {error_msg}"

                    elif "past date" in error_msg.lower():
                        answer = "❌ Cannot book appointments in the past.\n\nPlease select today or a future date."

                    elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        answer = "❌ Authentication issue. Please try logging out and logging back in."

                    else:
                        answer = f"❌ Error: {error_msg}\n\nPlease try again or contact support if the issue persists."

                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)

                # Store in chat history
                await self._save_to_history(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    confidence_score=0.95,
                    from_cache=False,
                    processing_time_ms=processing_time,
                    user_id=customer_id
                )

                return ChatResponse(
                    answer=answer,
                    session_id=session_id,
                    from_cache=False,
                    processing_time_ms=processing_time,
                    timestamp=datetime.utcnow(),
                    confidence=0.95,
                    sources=[f"function_{function_name}"]
                )

            # Regular text response from Gemini (no function call)
            elif gemini_response["type"] == "text":
                answer = gemini_response["text"]
                processing_time = int((time.time() - start_time) * 1000)

                await self._save_to_history(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    confidence_score=0.8,
                    from_cache=False,
                    processing_time_ms=processing_time,
                    user_id=customer_id
                )

                return ChatResponse(
                    answer=answer,
                    session_id=session_id,
                    from_cache=False,
                    processing_time_ms=processing_time,
                    timestamp=datetime.utcnow(),
                    confidence=0.8,
                    sources=["gemini_text"]
                )

            # If we reach here, Gemini didn't call a function or provide text
            # This shouldn't happen, but handle it gracefully
            logger.warning("No response type from Gemini")
            answer = "I'm sorry, I couldn't process that request. Could you please rephrase?"
            processing_time = int((time.time() - start_time) * 1000)

            return ChatResponse(
                answer=answer,
                session_id=session_id,
                from_cache=False,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow(),
                confidence=0.0,
                sources=[]
            )
        
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return ChatResponse(
                answer="I apologize, but I'm having trouble processing your request. Please try again.",
                session_id=session_id,
                from_cache=False,
                processing_time_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.utcnow(),
                confidence=0.0,
                sources=[]
            )
    
    async def process_query_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """
        Process user query with streaming RAG response
        
        Yields:
            ChatStreamChunk objects with incremental content
        """
        session_id = session_id or str(uuid.uuid4())
        chunk_index = 0
        
        try:
            # Step 1: Retrieve context (same as non-streaming) with customer isolation
            logger.info(f"Retrieving context for stream: {question[:50]}...")
            relevant_docs = await self.vector_db_service.search(
                query=question,
                top_k=self.max_context_docs,
                filters=self._build_filters(filters),
                customer_email=None  # Note: Streaming doesn't receive customer_email currently
            )
            
            context = self._build_context(relevant_docs)
            history = self._format_history(conversation_history) if conversation_history else None
            
            # Step 2: Stream response from Gemini
            logger.info("Streaming response with Gemini...")
            async for chunk_text in self.gemini_service.generate_response_stream(
                prompt=question,
                context=context,
                conversation_history=history
            ):
                yield ChatStreamChunk(
                    content=chunk_text,
                    is_final=False,
                    session_id=session_id,
                    chunk_index=chunk_index
                )
                chunk_index += 1
            
            # Send final marker
            yield ChatStreamChunk(
                content="",
                is_final=True,
                session_id=session_id,
                chunk_index=chunk_index
            )
        
        except Exception as e:
            logger.error(f"Error streaming query: {e}", exc_info=True)
            yield ChatStreamChunk(
                content="I apologize, but I'm having trouble processing your request.",
                is_final=True,
                session_id=session_id,
                chunk_index=0
            )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        return {
            "vector_db_type": self.vector_db_service.vector_db_type,
            "max_context_docs": self.max_context_docs,
            "gemini_available": self.gemini_service.is_available(),
            "vector_db_available": self.vector_db_service.is_available(),
        }
    
    async def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get chat history for a session from database
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
        
        Returns:
            List of chat messages
        """
        return await db_get_chat_history(session_id, limit)
    
    async def clear_chat_history(self, session_id: str) -> None:
        """
        Clear chat history for a session from database
        
        Args:
            session_id: Session identifier
        """
        await db_clear_chat_history(session_id)
    
    # Private helper methods
    
    async def _save_to_history(
        self, 
        session_id: str, 
        question: str, 
        answer: str,
        confidence_score: float = None,
        from_cache: bool = False,
        processing_time_ms: int = None,
        user_id: int = None
    ) -> None:
        """Save question and answer to database with additional metadata"""
        try:
            await save_chat_message(
                session_id=session_id, 
                question=question, 
                answer=answer,
                confidence_score=confidence_score,
                from_cache=from_cache,
                processing_time_ms=processing_time_ms,
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            score = doc.get("score", 0.0)
            context_parts.append(f"[{i}] (Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _build_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build metadata filters for vector search"""
        if not filters:
            return None
        
        db_filters = {}
        
        if filters.get("appointment_date"):
            db_filters["date"] = filters["appointment_date"]
        
        if filters.get("service_type"):
            db_filters["service_type"] = filters["service_type"]
        
        return db_filters if db_filters else None
    
    def _format_history(self, conversation_history: List[ChatMessage]) -> List[Dict[str, str]]:
        """Format conversation history for Gemini"""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in conversation_history
        ]
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on document relevance"""
        if not documents:
            return 0.0
        
        # Average of top document scores
        scores = [doc.get("score", 0.0) for doc in documents[:3]]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_appointment_id(self, question: str) -> Optional[int]:
        """Extract appointment ID from user query"""
        import re
        
        # Look for various patterns that might indicate appointment ID
        patterns = [
            r'appointment\s+(?:id\s+)?#?(\d+)',  # "appointment #123" or "appointment id 123"
            r'#(\d+)',  # "#123"
            r'id\s+(\d+)',  # "id 123"
            r'appointment\s+(\d+)',  # "appointment 123"
            r'number\s+(\d+)',  # "number 123"
            r'(?:details|detail|info|information).*?(?:appointment|for)\s+(?:#|id\s+)?(\d+)',  # "details for appointment 123"
            r'(\d+)(?:\s+appointment)',  # "123 appointment"
        ]
        
        question_lower = question.lower()
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                try:
                    appointment_id = int(match.group(1))
                    logger.debug(f"Extracted appointment ID: {appointment_id} from query: {question}")
                    return appointment_id
                except (ValueError, IndexError):
                    continue
        
        logger.debug(f"No appointment ID found in query: {question}")
        return None
