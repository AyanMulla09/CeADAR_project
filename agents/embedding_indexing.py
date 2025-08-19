"""
Embedding/Indexing Agent - Indexes papers to ElasticSearch with embeddings
"""
import json
import logging
import os
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseReACTAgent
from tools.research_tools import elasticsearch_tool
from config import Config

logger = logging.getLogger(__name__)


class EmbeddingIndexingAgent(BaseReACTAgent):
    """Agent responsible for indexing papers to ElasticSearch with embeddings"""
    
    def __init__(self):
        # Use the elasticsearch tool if available
        tools: List[BaseTool] = []
        if elasticsearch_tool is not None:
            tools.append(elasticsearch_tool)
        
        # If no elasticsearch tool available, create a dummy tool
        if not tools:
            from langchain.tools import Tool
            dummy_tool = Tool(
                name="elasticsearch_placeholder",
                description="ElasticSearch is not configured. This is a placeholder tool.",
                func=lambda x: "ElasticSearch not available. Please configure ElasticSearch to use indexing features."
            )
            tools.append(dummy_tool)
        
        super().__init__(
            name="embedding_indexing_agent",
            description="Indexes research papers to ElasticSearch with semantic embeddings for RAG retrieval",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for embedding/indexing agent"""
        template = """
You are a Embedding and Indexing Agent responsible for indexing papers to ElasticSearch with semantic embeddings.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You should:
1. Generate embeddings for paper content
2. Index papers to ElasticSearch
3. Manage search indexes
4. Provide indexing statistics

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _extract_paper_chunks(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract chunks from a paper for indexing"""
        try:
            extracted_text = paper.get("extracted_text", "")
            if not extracted_text:
                return []
            
            # Simple chunking strategy - could be enhanced
            max_chunk_length = 1000  # Characters per chunk for embedding
            chunks = []
            
            # Split by paragraphs first
            paragraphs = extracted_text.split('\n\n')
            current_chunk = ""
            chunk_num = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "chunk_id": f"{paper.get('url', '').rsplit('/', 1)[-1].split('v')[0]}_chunk_{chunk_num}",
                            "paper_id": paper.get('url', '').rsplit('/', 1)[-1].split('v')[0],
                            "title": paper.get("title", ""),
                            "url": paper.get("url", ""),
                            "content": current_chunk.strip(),
                            "chunk_number": chunk_num,
                            "word_count": len(current_chunk.strip().split())
                        })
                        chunk_num += 1
                    current_chunk = para + "\n\n"
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    "chunk_id": f"{paper.get('url', '').rsplit('/', 1)[-1].split('v')[0]}_chunk_{chunk_num}",
                    "paper_id": paper.get('url', '').rsplit('/', 1)[-1].split('v')[0],
                    "title": paper.get("title", ""),
                    "url": paper.get("url", ""),
                    "content": current_chunk.strip(),
                    "chunk_number": chunk_num,
                    "word_count": len(current_chunk.strip().split())
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting chunks from paper: {e}")
            return []
    
    def _prepare_documents_for_indexing(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare all documents for ElasticSearch indexing"""
        all_documents = []
        
        for paper in papers:
            try:
                # Extract chunks from the paper
                chunks = self._extract_paper_chunks(paper)
                
                # Add metadata to each chunk
                for chunk in chunks:
                    document = {
                        "paper_id": chunk["paper_id"],
                        "title": chunk["title"],
                        "url": chunk["url"],
                        "content": chunk["content"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_number": chunk["chunk_number"],
                        "word_count": chunk["word_count"],
                        "research_gap": paper.get("research_gap", ""),
                        "gap_direction": paper.get("gap_direction", ""),
                        "gap_solution": paper.get("gap_solution", ""),
                        "gap_keywords": paper.get("gap_keywords", [])
                    }
                    all_documents.append(document)
                    
            except Exception as e:
                logger.error(f"Error preparing document for paper {paper.get('title', 'Unknown')}: {e}")
                continue
        
        return all_documents
    
    def _perform_sample_search(self, query: str) -> Dict[str, Any]:
        """Perform a sample search to verify indexing works"""
        try:
            # Use the search_documents method directly
            search_result = elasticsearch_tool._search_documents(query, top_k=5)
            
            # Handle dict response format
            if isinstance(search_result, dict):
                if "error" in search_result:
                    return {"error": search_result["error"]}
                
                results = search_result.get("results", [])
                
                return {
                    "search_successful": True,
                    "query": query,
                    "results_found": len(results),
                    "sample_results": [
                        {
                            "title": r["document"].get("title", ""),
                            "score": r.get("score", 0),
                            "content_preview": r["document"].get("content", "")[:200] + "..."
                        }
                        for r in results[:3]
                    ]
                }
            
            # Handle string response format
            if isinstance(search_result, str):
                if "error" in search_result.lower():
                    return {"error": search_result}
                else:
                    return {
                        "search_successful": True,
                        "query": query,
                        "results_found": 1,  # Assume some results if no error
                        "sample_results": []
                    }
            
            return {"error": f"Unexpected search result type: {type(search_result)}"}
            
        except Exception as e:
            return {"error": f"Sample search failed: {str(e)}"}
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute embedding and indexing to ElasticSearch"""
        papers = input_data.get("papers_with_gap_analysis", [])
        topic = input_data.get("original_topic", "")
        
        if not papers:
            return {"error": "No papers with gap analysis provided for indexing"}
        
        logger.info(f"Starting indexing of {len(papers)} papers to ElasticSearch")
        
        try:
            # Check if ElasticSearch is available
            if not elasticsearch_tool.es_client or not elasticsearch_tool.embedding_model:
                logger.error("ElasticSearch or embedding model not available")
                return {
                    "error": "ElasticSearch or embedding model not available",
                    "suggestion": "Please ensure ElasticSearch is running on localhost:9200 and sentence-transformers is installed"
                }
            
            logger.info("ElasticSearch and embedding model are available")
            
            # Prepare documents for indexing
            documents = self._prepare_documents_for_indexing(papers)
            
            if not documents:
                logger.error("No documents could be prepared for indexing")
                return {"error": "No documents could be prepared for indexing"}
            
            logger.info(f"Prepared {len(documents)} document chunks for indexing")
            
            # Use the ElasticSearch tool directly for indexing
            try:
                logger.info("Starting document formatting for ElasticSearch")
                
                # Prepare documents in the format expected by elasticsearch tool
                formatted_docs = []
                for doc in documents:
                    formatted_doc = {
                        "paper_id": doc["paper_id"],
                        "title": doc["title"], 
                        "url": doc["url"],
                        "content": doc["content"],
                        "chunk_id": doc["chunk_id"],
                        "chunk_number": doc["chunk_number"],
                        "word_count": doc["word_count"]
                    }
                    formatted_docs.append(formatted_doc)
                
                logger.info(f"Formatted {len(formatted_docs)} documents for indexing")
                
                # Use the elasticsearch tool's index method
                logger.info("Calling elasticsearch tool _index_documents method")
                result = elasticsearch_tool._index_documents(formatted_docs)
                
                logger.info(f"Indexing result: {result}")
                
                if "error" in result:
                    logger.error(f"Indexing failed with error: {result['error']}")
                    return {"error": f"Indexing failed: {result['error']}"}
                
                indexed_count = result.get("indexed_count", 0)
                logger.info(f"Successfully indexed {indexed_count} document chunks")
                
            except Exception as e:
                logger.error(f"Error during indexing: {e}")
                return {"error": f"Indexing failed: {str(e)}"}
            
            # Perform sample searches to verify functionality
            sample_searches = []
            search_queries = [topic, "methodology", "evaluation"]
            
            for query in search_queries:
                search_result = self._perform_sample_search(query)
                sample_searches.append(search_result)
            
            # Calculate statistics
            unique_papers = len(set(doc["paper_id"] for doc in documents))
            avg_chunks_per_paper = len(documents) / unique_papers if unique_papers > 0 else 0
            total_words = sum(doc["word_count"] for doc in documents)
            
            result = {
                "indexing_successful": True,
                "documents_indexed": indexed_count,
                "unique_papers_indexed": unique_papers,
                "total_chunks": len(documents),
                "average_chunks_per_paper": avg_chunks_per_paper,
                "total_words_indexed": total_words,
                "elasticsearch_index": Config.ELASTICSEARCH_INDEX,
                "embedding_model": Config.EMBEDDING_MODEL,
                "sample_searches": sample_searches,
                "rag_system_ready": all(not s.get("error") for s in sample_searches),
                "indexing_statistics": {
                    "papers_processed": len(papers),
                    "chunks_created": len(documents),
                    "indexing_success_rate": indexed_count / len(documents) if documents else 0
                }
            }
            
            logger.info(f"Indexing completed: {indexed_count} chunks indexed from {unique_papers} papers")
            
            return result
            
        except Exception as e:
            error_msg = f"Embedding/indexing process failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_embedding_indexing_agent() -> EmbeddingIndexingAgent:
    """Factory function to create embedding/indexing agent"""
    from agents.base_agent import agent_registry
    agent = EmbeddingIndexingAgent()
    agent_registry.register_agent(agent)
    return agent
