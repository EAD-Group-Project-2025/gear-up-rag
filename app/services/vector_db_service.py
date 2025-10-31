"""Vector Database Service for embeddings storage and retrieval"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing vector embeddings with Pinecone or FAISS"""
    
    def __init__(self):
        self.use_pinecone = os.getenv("USE_PINECONE", "false").lower() == "true"
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize vector database
        if self.use_pinecone:
            self._init_pinecone()
        else:
            self._init_faiss()
        
        logger.info(f"Initialized vector DB (Pinecone: {self.use_pinecone})")
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            index_name = os.getenv("PINECONE_INDEX_NAME", "gearup-chatbot")
            
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=api_key)
            
            # Create index if doesn't exist
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=environment)
                )
            
            self.index = pc.Index(index_name)
            self.vector_db_type = "pinecone"
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def _init_faiss(self):
        """Initialize FAISS vector database (local)"""
        try:
            import faiss
            
            dimension = 384  # all-MiniLM-L6-v2 dimension
            self.index = faiss.IndexFlatL2(dimension)
            self.vector_db_type = "faiss"
            
            # Storage for metadata (FAISS only stores vectors)
            self.metadata_store = []
            
            # Load existing index if available
            index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
            if os.path.exists(f"{index_path}/index.faiss"):
                self.index = faiss.read_index(f"{index_path}/index.faiss")
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if vector DB is available"""
        return self.index is not None
    
    async def initialize(self):
        """Load initial data from database"""
        try:
            from app.database.db import fetch_appointment_data
            
            logger.info("Loading appointment data for embeddings...")
            appointments = await fetch_appointment_data()
            
            if appointments:
                await self.add_documents(appointments)
                logger.info(f"Loaded {len(appointments)} appointments into vector DB")
        
        except Exception as e:
            logger.error(f"Error initializing vector DB data: {e}")
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to vector database
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        try:
            texts = [doc["text"] for doc in documents]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            if self.vector_db_type == "pinecone":
                # Prepare vectors for Pinecone
                vectors = [
                    {
                        "id": doc.get("id", f"doc_{i}"),
                        "values": embeddings[i].tolist(),
                        "metadata": doc.get("metadata", {})
                    }
                    for i, doc in enumerate(documents)
                ]
                self.index.upsert(vectors=vectors)
            
            else:  # FAISS
                self.index.add(embeddings)
                self.metadata_store.extend([doc.get("metadata", {}) for doc in documents])
                
                # Save index
                index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
                os.makedirs(index_path, exist_ok=True)
                
                import faiss
                faiss.write_index(self.index, f"{index_path}/index.faiss")
            
            logger.info(f"Added {len(documents)} documents to vector DB")
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of matching documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            if self.vector_db_type == "pinecone":
                # Search Pinecone
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    filter=filters,
                    include_metadata=True
                )
                
                return [
                    {
                        "text": match["metadata"].get("text", ""),
                        "score": match["score"],
                        "metadata": match["metadata"]
                    }
                    for match in results["matches"]
                ]
            
            else:  # FAISS
                # Search FAISS
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1),
                    top_k
                )
                
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.metadata_store):
                        metadata = self.metadata_store[idx]
                        results.append({
                            "text": metadata.get("text", ""),
                            "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                            "metadata": metadata
                        })
                
                return results
        
        except Exception as e:
            logger.error(f"Error searching vector DB: {e}")
            return []
    
    async def update_from_database(self):
        """Update embeddings from database (for real-time updates)"""
        try:
            from app.database.db import fetch_appointment_data
            
            logger.info("Updating embeddings from database...")
            
            # Clear existing data (or implement incremental update)
            if self.vector_db_type == "faiss":
                import faiss
                dimension = 384
                self.index = faiss.IndexFlatL2(dimension)
                self.metadata_store = []
            else:
                # For Pinecone, upsert will update existing vectors
                pass
            
            # Reload data
            appointments = await fetch_appointment_data()
            if appointments:
                await self.add_documents(appointments)
                logger.info(f"Updated {len(appointments)} appointments in vector DB")
        
        except Exception as e:
            logger.error(f"Error updating from database: {e}")
            raise
    
    async def close(self):
        """Cleanup resources"""
        if self.vector_db_type == "faiss":
            # Save final state
            index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
            os.makedirs(index_path, exist_ok=True)
            
            import faiss
            faiss.write_index(self.index, f"{index_path}/index.faiss")
            logger.info("Saved FAISS index")
