"""
Configuration settings for the ReACT AI Agents Research Pipeline
"""
import os
from typing import Dict, Any

class Config:
    """Central configuration class for all agents and tools"""
    
    # LLM Configuration
    LLM_MODEL = "llama3.1"
    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 4000
    
    # Research Pipeline Configuration
    MAX_CHUNK_WORDS = 1500
    OVERLAP_WORDS = 300
    MAX_PDF_CHARS = 1000000  # Increased to ~1M chars to process full papers
    MAX_GAP_CHARS = 10000
    MAX_PER_QUERY = 2
    NUM_THREADS = 4
    
    # Directory Configuration
    PDF_DIR = "pdf_cache"
    CACHE_DIR = "cache"
    OUTPUT_DIR = "output"
    
    # ElasticSearch Configuration
    ELASTICSEARCH_HOST = "localhost:9200"
    ELASTICSEARCH_INDEX = "research_papers"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RAG_RETRIEVAL_TOP_K = 10
    KEYWORD_EXTRACTION_TOP_N = 5
    
    # File paths
    LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.pkl")
    
    # Agent Configuration
    AGENT_VERBOSE = True
    AGENT_MAX_ITERATIONS = 10
    AGENT_EARLY_STOPPING_METHOD = "generate"
    
    # Tool Configuration
    ARXIV_MAX_RESULTS = 50
    ARXIV_TIMEOUT = 30
    PDF_DOWNLOAD_TIMEOUT = 60
    PDF_DOWNLOAD_RETRY_ATTEMPTS = 3
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.PDF_DIR, cls.CACHE_DIR, cls.OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

# Create directories on import
Config.create_directories()
