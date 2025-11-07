"""Database operations for chat history"""

import logging
from typing import List, Dict, Any
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.models.chat_history import ChatHistory, Base
from app.database.db import get_engine, get_async_session

logger = logging.getLogger(__name__)


async def init_chat_history_table():
    """Create chat_history table if it doesn't exist"""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Chat history table initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing chat history table: {e}", exc_info=True)
        raise


async def save_chat_message(
    session_id: str,
    question: str,
    answer: str,
    confidence_score: float = None,
    from_cache: bool = False,
    processing_time_ms: int = None,
    user_id: int = None,
    session: AsyncSession = None
) -> None:
    """
    Save a chat message to the database
    
    Args:
        session_id: Session identifier
        question: User question
        answer: Bot answer
        confidence_score: Optional confidence score
        from_cache: Whether response was from cache
        processing_time_ms: Processing time in milliseconds
        user_id: Optional user ID
        session: Optional database session
    """
    try:
        if session is None:
            session_maker = get_async_session()
            async with session_maker() as session:
                return await save_chat_message(
                    session_id, question, answer, confidence_score, 
                    from_cache, processing_time_ms, user_id, session
                )
        
        # Set current timestamp for both required timestamp fields
        current_time = datetime.utcnow()
        
        chat_message = ChatHistory(
            session_id=session_id,
            question=question,
            answer=answer,
            timestamp=current_time,
            created_at=current_time,
            confidence_score=confidence_score,
            from_cache=from_cache,
            processing_time_ms=processing_time_ms,
            user_id=user_id
        )
        
        session.add(chat_message)
        await session.commit()
        logger.debug(f"Saved chat message for session: {session_id}")
    
    except Exception as e:
        logger.error(f"Error saving chat message: {e}", exc_info=True)
        if session:
            await session.rollback()
        raise


async def get_chat_history(
    session_id: str,
    limit: int = 10,
    session: AsyncSession = None
) -> List[Dict[str, Any]]:
    """
    Get chat history for a session
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return
        session: Optional database session
    
    Returns:
        List of chat messages
    """
    try:
        if session is None:
            session_maker = get_async_session()
            async with session_maker() as session:
                return await get_chat_history(session_id, limit, session)
        
        # Query chat history ordered by created_at (oldest first for chronological order)
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.asc())
            .limit(limit)
        )
        
        result = await session.execute(stmt)
        messages = result.scalars().all()
        
        # Convert to dict format efficiently
        return [
            {
                "id": msg.id,
                "session_id": msg.session_id,
                "question": msg.question,
                "answer": msg.answer,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "confidence_score": msg.confidence_score,
                "from_cache": msg.from_cache,
                "processing_time_ms": msg.processing_time_ms,
                "user_id": msg.user_id
            }
            for msg in messages
        ]
    
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}", exc_info=True)
        return []


async def clear_chat_history(
    session_id: str,
    session: AsyncSession = None
) -> int:
    """
    Clear chat history for a session
    
    Args:
        session_id: Session identifier
        session: Optional database session
    
    Returns:
        Number of messages deleted
    """
    try:
        if session is None:
            session_maker = get_async_session()
            async with session_maker() as session:
                return await clear_chat_history(session_id, session)
        
        stmt = delete(ChatHistory).where(ChatHistory.session_id == session_id)
        result = await session.execute(stmt)
        await session.commit()
        
        deleted_count = result.rowcount
        logger.info(f"Cleared {deleted_count} messages for session: {session_id}")
        return deleted_count
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}", exc_info=True)
        if session:
            await session.rollback()
        raise


async def get_recent_sessions(limit: int = 10, user_id: Optional[int] = None) -> List[str]:
    """
    Get list of recent session IDs

    Args:
        limit: Maximum number of sessions to return
        user_id: Optional user ID to filter sessions

    Returns:
        List of session IDs
    """
    try:
        session_maker = get_async_session()
        async with session_maker() as session:
            # Get distinct session IDs with the most recent created_at for each session
            from sqlalchemy import func

            # Base query for distinct sessions
            base_query = select(ChatHistory.session_id).distinct()

            # Filter by user_id if provided
            if user_id is not None:
                base_query = base_query.where(ChatHistory.user_id == user_id)

            base_query = base_query.order_by(ChatHistory.session_id)
            stmt = base_query.subquery()

            # Then get the most recent session for each
            final_stmt = (
                select(stmt.c.session_id)
                .select_from(
                    stmt.join(
                        ChatHistory,
                        stmt.c.session_id == ChatHistory.session_id
                    )
                )
                .group_by(stmt.c.session_id)
                .order_by(func.max(ChatHistory.created_at).desc())
                .limit(limit)
            )

            result = await session.execute(final_stmt)
            session_ids = result.scalars().all()
            return list(session_ids)
    
    except Exception as e:
        logger.error(f"Error fetching recent sessions: {e}", exc_info=True)
        return []
