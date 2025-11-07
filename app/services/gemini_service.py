"""Google Gemini LLM Service"""

import google.generativeai as genai
import os
from typing import List, AsyncGenerator, Dict, Any, Optional
import logging
import json

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

        # Define function/tool declarations for Gemini
        self.tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_user_appointments",
                        "description": "Get all appointments for the authenticated user. Use this when user asks about their appointments, bookings, or scheduled services.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "appointment_type": {
                                    "type": "string",
                                    "enum": ["all", "upcoming", "available"],
                                    "description": "Filter by appointment type: 'all' for all appointments, 'upcoming' for future appointments, 'available' for pending appointments"
                                }
                            },
                            "required": []
                        }
                    },
                    {
                        "name": "get_user_vehicles",
                        "description": "Get all vehicles owned by the authenticated user. Use this when user wants to book an appointment or asks about their vehicles.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "book_appointment",
                        "description": "Book a new service appointment for the user's vehicle. ONLY call this function when user confirms they want to book (e.g., says 'yes', 'confirm', 'book it'). Do NOT call this for initial requests - ask for confirmation first.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "vehicle_id": {
                                    "type": "integer",
                                    "description": "ID of the vehicle for the appointment"
                                },
                                "appointment_date": {
                                    "type": "string",
                                    "description": "Date for the appointment in YYYY-MM-DD format"
                                },
                                "start_time": {
                                    "type": "string",
                                    "description": "Start time in HH:MM:SS format (24-hour)"
                                },
                                "consultation_type": {
                                    "type": "string",
                                    "enum": ["GENERAL_CHECKUP", "SPECIFIC_ISSUE", "MAINTENANCE_ADVICE", "PERFORMANCE_ISSUE", "SAFETY_CONCERN", "OTHER"],
                                    "description": "Type of consultation needed. Valid values: GENERAL_CHECKUP (routine checkup), SPECIFIC_ISSUE (specific problem like oil change, brake service), MAINTENANCE_ADVICE (maintenance guidance), PERFORMANCE_ISSUE (performance problems), SAFETY_CONCERN (safety issues), OTHER (other services)"
                                },
                                "customer_issue": {
                                    "type": "string",
                                    "description": "Description of the issue or service request"
                                }
                            },
                            "required": ["vehicle_id", "appointment_date", "start_time", "consultation_type", "customer_issue"]
                        }
                    }
                ]
            }
        ]

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            tools=self.tools
        )

        logger.info(f"Initialized Gemini service with model: {self.model_name} and function calling tools")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.api_key is not None

    async def generate_response_with_functions(
        self,
        prompt: str,
        conversation_history: List[dict] = None
    ) -> Dict[str, Any]:
        """
        Generate response with function calling support

        Args:
            prompt: User question
            conversation_history: Previous conversation messages

        Returns:
            Dict with either text response or function call
        """
        try:
            # Build conversation history for chat
            history = []
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history.append({
                        "role": role,
                        "parts": [content]
                    })

            # Start chat session
            chat = self.model.start_chat(history=history)

            # Send message
            response = await chat.send_message_async(prompt)

            logger.info(f"Gemini response candidates: {len(response.candidates)}")

            # Check if response contains function calls
            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]

                # Check for function call
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    logger.info(f"Function call detected: {function_call.name}")

                    return {
                        "type": "function_call",
                        "function_name": function_call.name,
                        "function_args": dict(function_call.args)
                    }

                # Regular text response
                if hasattr(part, 'text') and part.text:
                    return {
                        "type": "text",
                        "text": part.text
                    }

            # Fallback
            return {
                "type": "text",
                "text": "I'm sorry, I couldn't process that request. Could you please rephrase?"
            }

        except Exception as e:
            logger.error(f"Error in function calling: {e}", exc_info=True)
            raise

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
1. Viewing their appointments
2. Booking service appointments
3. Answering questions about vehicle maintenance
4. Providing information about services

CRITICAL BOOKING RULES:
1. BEFORE booking an appointment, you MUST:
   - Call get_user_vehicles() first to see their vehicles
   - Show them the vehicle list and ask which one
   - Get the vehicle_id from the list (NEVER make up IDs)

2. Booking workflow:
   Step 1: User requests appointment → Call get_user_vehicles()
   Step 2: Show vehicles with IDs → Ask which vehicle
   Step 3: User chooses → Extract details (date, time, service)
   Step 4: Ask for confirmation with all details
   Step 5: User confirms "yes" → Call book_appointment() with the REAL vehicle_id from step 1

3. Consultation Type Mapping (IMPORTANT - use correct enum values):
   - Oil change, tire rotation, brake service, specific repairs → SPECIFIC_ISSUE
   - Routine checkup, inspection → GENERAL_CHECKUP
   - Performance problems, sluggish engine → PERFORMANCE_ISSUE
   - Brake issues, steering problems → SAFETY_CONCERN
   - Maintenance questions, advice → MAINTENANCE_ADVICE
   - Everything else → OTHER

4. NEVER call book_appointment() without first getting real vehicle IDs via get_user_vehicles()

5. When showing appointments, use ACTUAL data from get_user_appointments()

EXAMPLE CORRECT FLOW:
User: "book oil change for tomorrow"
You: Call get_user_vehicles() → See [{"id": 5, "year": 2018, "make": "Toyota", "model": "Camry"}]
You: "I see you have: 1. 2018 Toyota Camry (ID: 5). Which vehicle?"
User: "the Toyota"
You: "Confirm oil change for 2018 Toyota Camry tomorrow at [time]?"
User: "yes"
You: Call book_appointment(
    vehicle_id=5,  ← Real ID from "id" field
    consultation_type="SPECIFIC_ISSUE",  ← Correct enum value
    customer_issue="Oil change",  ← Description
    ...
)

Keep responses concise, accurate, and friendly.
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
