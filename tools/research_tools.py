"""
Custom tools for the ReACT AI Research Pipeline
"""
import os
import time
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Type

import fitz
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

# Optional imports with fallbacks
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("Warning: ElasticSearch not available. Install with: pip install elasticsearch")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: Sentence Transformers not available. Install with: pip install sentence-transformers")


# Input schemas for tools
class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query for ArXiv papers")
    max_results: int = Field(default=10, description="Maximum number of results to return")


class PDFDownloadInput(BaseModel):
    url: str = Field(description="ArXiv URL to download PDF from")


class PDFExtractInput(BaseModel):
    pdf_path: str = Field(description="Path to PDF file to extract text from")
    max_chars: Optional[int] = Field(default=None, description="Maximum characters to extract")


class ElasticsearchIndexInput(BaseModel):
    documents: List[Dict] = Field(description="List of documents to index")


class ElasticsearchSearchInput(BaseModel):
    query: str = Field(description="Search query")
    top_k: int = Field(default=10, description="Number of top results to return")


# Tool implementations
class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = "Search for research papers on ArXiv"
    args_schema: Type[BaseModel] = ArxivSearchInput
    
    def _run(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search ArXiv for papers"""
        url = (f"https://export.arxiv.org/api/query?search_query=all:"
               f"{query.replace(' ', '+')}&start=0&max_results={max_results}")
        
        try:
            response = requests.get(url, timeout=Config.ARXIV_TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception as e:
            return [{"error": f"ArXiv search failed: {str(e)}"}]
        
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
            link_elem = entry.find("{http://www.w3.org/2005/Atom}id")
            
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
            link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
            
            papers.append({
                "title": title,
                "summary": summary,
                "url": link
            })
        
        return papers


class PDFDownloadTool(BaseTool):
    name: str = "pdf_download"
    description: str = "Download PDF from ArXiv URL"
    args_schema: Type[BaseModel] = PDFDownloadInput
    
    def _run(self, url: str) -> str:
        """Download PDF and return local path"""
        try:
            # Convert ArXiv abs URL to PDF URL
            arxiv_id = url.rsplit("/", 1)[-1]
            arxiv_id_no_version = arxiv_id.split("v")[0]
            pdf_path = os.path.join(Config.PDF_DIR, f"{arxiv_id_no_version}.pdf")
            
            # Check if already downloaded
            if os.path.exists(pdf_path):
                return pdf_path
            
            # Download PDF
            pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
            response = requests.get(pdf_url, timeout=Config.PDF_DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            
            time.sleep(0.5)  # Be gentle on ArXiv
            return pdf_path
            
        except Exception as e:
            return f"Error downloading PDF: {str(e)}"


class PDFTextExtractTool(BaseTool):
    name: str = "pdf_extract_text"
    description: str = "Extract text from PDF file"
    args_schema: Type[BaseModel] = PDFExtractInput
    
    def _run(self, pdf_path: str, max_chars: Optional[int] = None) -> str:
        """Extract text from PDF"""
        try:
            txt = []
            char_count = 0
            
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    page_text = page.get_text("text")  # Correct PyMuPDF method
                    txt.append(page_text)
                    char_count += len(page_text)
                    
                    if max_chars and char_count >= max_chars:
                        break
            
            full_text = "\n".join(txt)
            
            if max_chars:
                return full_text[:max_chars]
            return full_text
            
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"


class ElasticsearchTool(BaseTool):
    name: str = "elasticsearch_operation"
    description: str = "Perform ElasticSearch operations (index, search)"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._es_client = None
        self._embedding_model = None
        if ELASTICSEARCH_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_clients()
    
    @property
    def es_client(self):
        return self._es_client
    
    @property
    def embedding_model(self):
        return self._embedding_model
    
    def _initialize_clients(self):
        """Initialize ElasticSearch and embedding model"""
        if not ELASTICSEARCH_AVAILABLE:
            print("Warning: ElasticSearch not available")
            return
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: Sentence Transformers not available")
            return
            
        try:
            self._es_client = Elasticsearch([f"http://{Config.ELASTICSEARCH_HOST}"])
            # Test connection
            self._es_client.cluster.health()
            print(f"ElasticSearch connected successfully at {Config.ELASTICSEARCH_HOST}")
        except Exception as e:
            print(f"Warning: ElasticSearch connection failed: {e}")
            self._es_client = None
        
        try:
            self._embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print(f"Embedding model loaded: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"Warning: Embedding model not available: {e}")
            self._embedding_model = None
    
    def _run(self, operation: str, **kwargs) -> Dict:
        """Perform ElasticSearch operation"""
        if not self.es_client:
            return {"error": "ElasticSearch not available"}
        
        try:
            if operation == "index":
                return self._index_documents(kwargs.get("documents", []))
            elif operation == "search":
                return self._search_documents(
                    kwargs.get("query", ""),
                    kwargs.get("top_k", 10)
                )
            else:
                return {"error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"error": f"ElasticSearch operation failed: {str(e)}"}
    
    def _index_documents(self, documents: List[Dict]) -> Dict:
        """Index documents to ElasticSearch"""
        if not self.embedding_model:
            return {"error": "Embedding model not available"}
        
        try:
            # Create index if it doesn't exist
            index_name = Config.ELASTICSEARCH_INDEX
            if self.es_client and not self.es_client.indices.exists(index=index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "paper_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "url": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384
                            }
                        }
                    }
                }
                if self.es_client:
                    self.es_client.indices.create(index=index_name, body=mapping)
            
            # Index documents
            indexed_count = 0
            for doc in documents:
                # Generate embedding
                content = doc.get("content", "")
                embedding = self.embedding_model.encode(content)
                
                doc_with_embedding = {
                    **doc,
                    "embedding": embedding.tolist()
                }
                
                if self.es_client:
                    self.es_client.index(
                        index=index_name,
                        body=doc_with_embedding
                    )
                indexed_count += 1
            
            return {"indexed_count": indexed_count}
            
        except Exception as e:
            return {"error": f"Indexing failed: {str(e)}"}
    
    def _search_documents(self, query: str, top_k: int) -> Dict:
        """Search documents in ElasticSearch using text search (vector search disabled for compatibility)"""
        try:
            # Use reliable text search instead of vector similarity
            search_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"]
                    }
                }
            }
            
            if self.es_client:
                response = self.es_client.search(
                    index=Config.ELASTICSEARCH_INDEX,
                    **search_body
                )
            
            results = []
            if response and "hits" in response:
                for hit in response["hits"]["hits"]:
                    results.append({
                        "score": hit["_score"],
                        "document": hit["_source"]
                    })
            
            return {"results": results}
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}


# Tool instances - create them carefully to avoid errors
try:
    arxiv_search_tool = ArxivSearchTool()
except Exception as e:
    print(f"Warning: Could not create ArxivSearchTool: {e}")
    arxiv_search_tool = None

try:
    pdf_download_tool = PDFDownloadTool()
except Exception as e:
    print(f"Warning: Could not create PDFDownloadTool: {e}")
    pdf_download_tool = None

try:
    pdf_extract_tool = PDFTextExtractTool()
except Exception as e:
    print(f"Warning: Could not create PDFTextExtractTool: {e}")
    pdf_extract_tool = None

try:
    elasticsearch_tool = ElasticsearchTool()
except Exception as e:
    print(f"Warning: Could not create ElasticsearchTool: {e}")
    elasticsearch_tool = None

# List of all available tools
ALL_TOOLS = [tool for tool in [arxiv_search_tool, pdf_download_tool, pdf_extract_tool, elasticsearch_tool] if tool is not None]
