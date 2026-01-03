"""
Module to handle Retrieval Augmented Generation (RAG) functionality.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from utils import logger
from config import settings

# Global variables for RAG
vector_db = None
embedding_model = None
collection = None

def initialize_rag(db_path="./data/chroma_db"):
    """
    Initialize the RAG system with ChromaDB and embeddings model.
    
    Args:
        db_path: Path to store the ChromaDB database
    """
    global vector_db, embedding_model, collection
    
    try:
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {db_path}")
        vector_db = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model
        logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Get or create collection
        collection = vector_db.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False

def add_documents(documents: list, metadatas: list = None):
    """
    Add documents to the vector database.
    
    Args:
        documents: List of document texts
        metadatas: List of metadata dicts (optional)
    
    Returns:
        True if successful, False otherwise
    """
    if collection is None:
        logger.warning("RAG system not initialized. Call initialize_rag() first.")
        return False
    
    try:
        # Generate unique IDs for documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # If no metadata provided, create empty metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Add documents to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to vector database")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add documents to RAG: {e}")
        return False

def retrieve_context(query: str, num_results: int = 3):
    """
    Retrieve relevant documents from the vector database based on a query.
    
    Args:
        query: The query string
        num_results: Number of results to retrieve
    
    Returns:
        List of relevant documents or empty list if not found
    """
    if collection is None:
        logger.warning("RAG system not initialized.")
        return []
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )
        
        # Extract documents from results
        if results and results['documents'] and len(results['documents']) > 0:
            return results['documents'][0]
        
        return []
        
    except Exception as e:
        logger.error(f"Failed to retrieve context from RAG: {e}")
        return []

def augment_prompt(user_input: str, num_context_docs: int = 3) -> tuple:
    """
    Augment user input with relevant context from the vector database.
    
    Args:
        user_input: The original user query
        num_context_docs: Number of context documents to retrieve
    
    Returns:
        Tuple of (augmented_prompt, retrieved_context)
    """
    if collection is None:
        return user_input, []
    
    try:
        # Retrieve relevant context
        context_docs = retrieve_context(user_input, num_context_docs)
        
        if context_docs:
            # Build augmented prompt with context
            context_str = "\n".join([f"- {doc}" for doc in context_docs])
            augmented_prompt = f"""Based on the following context, answer the user's question:

Context:
{context_str}

User Question: {user_input}

Answer:"""
            
            logger.info(f"Augmented prompt with {len(context_docs)} context documents")
            return augmented_prompt, context_docs
        else:
            logger.info("No relevant context found, using original prompt")
            return user_input, []
        
    except Exception as e:
        logger.error(f"Failed to augment prompt: {e}")
        return user_input, []

def get_collection_stats():
    """
    Get statistics about the current collection.
    
    Returns:
        Dict with collection statistics
    """
    if collection is None:
        return {"status": "RAG system not initialized"}
    
    try:
        count = collection.count()
        return {
            "status": "initialized",
            "document_count": count,
            "collection_name": collection.name
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {"status": "error", "message": str(e)}
