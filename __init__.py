"""
ReACT AI Research Pipeline

A comprehensive research paper analysis pipeline using ReACT agents and LangChain.
"""

__version__ = "1.0.0"
__author__ = "AI Research Pipeline Team"

from .config import Config
from .main import run_research_pipeline, initialize_all_agents

__all__ = [
    "Config",
    "run_research_pipeline", 
    "initialize_all_agents"
]
