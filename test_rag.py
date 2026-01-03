"""
Test script for RAG functionality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import rag
from core import responder
from utils import logger

def test_rag():
    """Test RAG system."""
    
    # Initialize RAG
    logger.info("=" * 50)
    logger.info("Testing RAG System")
    logger.info("=" * 50)
    
    rag.initialize_rag()
    
    # Add sample documents
    sample_docs = [
        "Python is a programming language created by Guido van Rossum.",
        "Machine learning algorithms learn patterns from data.",
        "Natural Language Processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Transformers revolutionized NLP with attention mechanisms.",
    ]
    
    logger.info("\nAdding sample documents...")
    rag.add_documents(sample_docs)
    
    # Test retrieval
    test_queries = [
        "What is Python?",
        "Tell me about machine learning",
        "How do transformers work?",
    ]
    
    logger.info("\n" + "=" * 50)
    logger.info("Testing Document Retrieval")
    logger.info("=" * 50)
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        context = rag.retrieve_context(query, num_results=2)
        logger.info(f"Retrieved Context:")
        for i, doc in enumerate(context, 1):
            logger.info(f"  {i}. {doc}")
    
    # Test prompt augmentation
    logger.info("\n" + "=" * 50)
    logger.info("Testing Prompt Augmentation")
    logger.info("=" * 50)
    
    query = "What is Python used for?"
    augmented, context = rag.augment_prompt(query, num_context_docs=2)
    logger.info(f"\nOriginal Query: {query}")
    logger.info(f"\nAugmented Prompt:\n{augmented}")
    
    # Get collection stats
    logger.info("\n" + "=" * 50)
    logger.info("Collection Statistics")
    logger.info("=" * 50)
    stats = rag.get_collection_stats()
    logger.info(f"Stats: {stats}")
    
    logger.info("\nRAG tests completed successfully!")

if __name__ == "__main__":
    test_rag()
