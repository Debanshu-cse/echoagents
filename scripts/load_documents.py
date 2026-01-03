"""
Script to load documents into the RAG system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import rag
from utils import logger

def load_from_file(filepath):
    """
    Load documents from a text file (one document per line).
    
    Args:
        filepath: Path to the file
    """
    logger.info(f"Loading documents from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            documents = [line.strip() for line in f if line.strip()]
        
        rag.initialize_rag()
        rag.add_documents(documents)
        logger.info(f"Successfully loaded {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")

def load_knowledge_base():
    """
    Load a sample knowledge base for testing.
    """
    sample_documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "RAG (Retrieval Augmented Generation) combines retrieval and generation for better AI responses.",
        "Natural Language Processing (NLP) is the field of AI that deals with text and language.",
        "ChromaDB is a vector database designed for storing embeddings and semantic search.",
        "Sentence transformers convert text into dense vector representations.",
        "LLMs (Large Language Models) are trained on vast amounts of text data.",
        "Vector embeddings capture semantic meaning of text in a mathematical form.",
    ]
    
    logger.info("Initializing RAG system with sample documents")
    rag.initialize_rag()
    rag.add_documents(sample_documents)
    
    logger.info("Sample knowledge base loaded successfully")
    stats = rag.get_collection_stats()
    logger.info(f"Collection stats: {stats}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        load_from_file(sys.argv[1])
    else:
        load_knowledge_base()
