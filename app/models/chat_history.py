"""SQLAlchemy models for chat history"""

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class ChatHistory(Base):
    """Chat history table for storing conversation messages"""
    __tablename__ = "chat_history"
    
    # Match the actual database schema exactly
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    from_cache = Column(Boolean, nullable=True)
    processing_time_ms = Column(BigInteger, nullable=True)
    user_id = Column(BigInteger, nullable=True)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "from_cache": self.from_cache,
            "processing_time_ms": self.processing_time_ms,
            "user_id": self.user_id
        }
