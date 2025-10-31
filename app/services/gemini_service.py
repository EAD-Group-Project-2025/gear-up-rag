"""Google Gemini LLM Service"""

import google.generativeai as genai
import os
from typing import List, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini LLM"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        
        genai.configure(api_key=self.api_key)
        
        # Configure model
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        logger.info(f"Initialized Gemini service with model: {self.model_name}")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.api_key is not None
    
    async def generate_response(
        self,
        prompt: str,
        context: str = "",
        conversation_history: List[dict] = None
    ) -> str:
        """
        Generate response from Gemini
        
        Args:
            prompt: User question
            context: Retrieved context from vector DB
            conversation_history: Previous conversation messages
        
        Returns:
            Generated answer
        """
        try:
            # Build the full prompt with context
            full_prompt = self._build_prompt(prompt, context, conversation_history)
            
            # Generate response
            response = await self.model.generate_content_async(full_prompt)
            
            return response.text
        
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}", exc_info=True)
            raise
    
    async def generate_response_stream(
        self,
        prompt: str,
        context: str = "",
        conversation_history: List[dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from Gemini
        
        Yields:
            Text chunks
        """
        try:
            full_prompt = self._build_prompt(prompt, context, conversation_history)
            
            response = await self.model.generate_content_async(
                full_prompt,
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        except Exception as e:
            logger.error(f"Error streaming Gemini response: {e}", exc_info=True)
            raise
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        conversation_history: List[dict] = None
    ) -> str:
        """Build the complete prompt with system instructions and context"""
        
        system_instruction = """You are an AI assistant for GearUp, a vehicle service and maintenance platform.
Your role is to help customers with:
1. Checking available appointment slots
2. Providing information about services
3. Answering questions about vehicle maintenance
4. Helping schedule appointments

Use the provided context to give accurate, helpful responses.
If you don't have enough information, politely ask for clarification.
Keep responses concise and friendly.
"""
        
        prompt_parts = [system_instruction]
        
        # Add conversation history
        if conversation_history:
            prompt_parts.append("\nConversation History:")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        # Add context from RAG
        if context:
            prompt_parts.append(f"\nRelevant Information:\n{context}")
        
        # Add current question
        prompt_parts.append(f"\nUser Question: {question}")
        prompt_parts.append("\nAssistant Response:")
        
        return "\n".join(prompt_parts)
