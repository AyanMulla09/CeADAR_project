"""
Download/Extract Agent - Downloads PDFs and extracts text content
"""
import json
import logging
import os
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseReACTAgent
from tools.research_tools import pdf_download_tool, pdf_extract_tool
from config import Config

logger = logging.getLogger(__name__)


class DownloadExtractAgent(BaseReACTAgent):
    """Agent responsible for downloading PDFs and extracting text content"""
    
    def __init__(self):
        # Use the PDF tools
        tools: List[BaseTool] = [pdf_download_tool, pdf_extract_tool]
        
        super().__init__(
            name="download_extract_agent",
            description="Downloads PDF files and extracts text content for analysis",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for download/extract agent"""
        template = """
You are a Download and Extraction Agent responsible for downloading and extracting content from research papers.

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
1. Download papers from provided URLs
2. Extract text content from PDFs
3. Handle various file formats
4. Provide extraction statistics

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _download_single_pdf(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Download a single PDF file"""
        try:
            url = paper.get("url", "")
            if not url:
                return {"error": "No URL provided", "paper": paper}
            
            # Download PDF
            pdf_path = pdf_download_tool._run(url)
            
            if pdf_path.startswith("Error"):
                return {"error": pdf_path, "paper": paper}
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "url": url,
                "title": paper.get("title", ""),
                "paper": paper
            }
            
        except Exception as e:
            return {"error": f"Download failed: {str(e)}", "paper": paper}
    
    def _extract_single_pdf(self, pdf_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from a single PDF"""
        try:
            pdf_path = pdf_info.get("pdf_path", "")
            if not pdf_path or not os.path.exists(pdf_path):
                return {"error": "PDF file not found", "pdf_info": pdf_info}
            
            # Extract text (limited for memory efficiency)
            extracted_text = pdf_extract_tool._run(pdf_path, max_chars=Config.MAX_PDF_CHARS)
            
            if extracted_text.startswith("Error"):
                return {"error": extracted_text, "pdf_info": pdf_info}
            
            # Calculate basic statistics
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "extracted_text": extracted_text,
                "word_count": word_count,
                "char_count": char_count,
                "title": pdf_info.get("title", ""),
                "url": pdf_info.get("url", ""),
                "paper": pdf_info.get("paper", {})
            }
            
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}", "pdf_info": pdf_info}
    
    def _batch_download_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Download multiple papers in parallel"""
        logger.info(f"Starting batch download of {len(papers)} papers")
        
        successful_downloads = []
        failed_downloads = []
        
        with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
            # Submit download tasks
            future_to_paper = {
                executor.submit(self._download_single_pdf, paper): paper 
                for paper in papers
            }
            
            # Process completed downloads
            for future in as_completed(future_to_paper):
                result = future.result()
                
                if result.get("success"):
                    successful_downloads.append(result)
                    logger.info(f"Downloaded: {result.get('title', 'Unknown')[:50]}...")
                else:
                    failed_downloads.append(result)
                    logger.warning(f"Download failed: {result.get('error', 'Unknown error')}")
        
        return {
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "download_stats": {
                "total_papers": len(papers),
                "successful": len(successful_downloads),
                "failed": len(failed_downloads),
                "success_rate": len(successful_downloads) / len(papers) if papers else 0
            }
        }
    
    def _batch_extract_papers(self, download_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract text from multiple PDFs in parallel"""
        logger.info(f"Starting batch extraction of {len(download_results)} PDFs")
        
        successful_extractions = []
        failed_extractions = []
        
        with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
            # Submit extraction tasks
            future_to_download = {
                executor.submit(self._extract_single_pdf, download_result): download_result 
                for download_result in download_results
            }
            
            # Process completed extractions
            for future in as_completed(future_to_download):
                result = future.result()
                
                if result.get("success"):
                    successful_extractions.append(result)
                    logger.info(f"Extracted: {result.get('title', 'Unknown')[:50]}... ({result.get('word_count', 0)} words)")
                else:
                    failed_extractions.append(result)
                    logger.warning(f"Extraction failed: {result.get('error', 'Unknown error')}")
        
        return {
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
            "extraction_stats": {
                "total_pdfs": len(download_results),
                "successful": len(successful_extractions),
                "failed": len(failed_extractions),
                "success_rate": len(successful_extractions) / len(download_results) if download_results else 0,
                "total_words_extracted": sum(e.get("word_count", 0) for e in successful_extractions),
                "average_words_per_paper": (
                    sum(e.get("word_count", 0) for e in successful_extractions) / len(successful_extractions)
                    if successful_extractions else 0
                )
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download and extraction tasks"""
        papers = input_data.get("filtered_papers", input_data.get("papers", []))
        
        if not papers:
            return {"error": "No papers provided for download/extraction"}
        
        logger.info(f"Starting download and extraction for {len(papers)} papers")
        
        try:
            # Step 1: Download PDFs
            download_results = self._batch_download_papers(papers)
            
            if not download_results["successful_downloads"]:
                return {
                    "error": "No PDFs were successfully downloaded",
                    "download_results": download_results
                }
            
            # Step 2: Extract text from successful downloads
            extraction_results = self._batch_extract_papers(download_results["successful_downloads"])
            
            # Combine results
            result = {
                "download_results": download_results,
                "extraction_results": extraction_results,
                "papers_with_content": extraction_results["successful_extractions"],
                "final_stats": {
                    "input_papers": len(papers),
                    "downloaded_pdfs": len(download_results["successful_downloads"]),
                    "extracted_papers": len(extraction_results["successful_extractions"]),
                    "overall_success_rate": (
                        len(extraction_results["successful_extractions"]) / len(papers)
                        if papers else 0
                    ),
                    "total_content_words": extraction_results["extraction_stats"]["total_words_extracted"]
                }
            }
            
            logger.info(
                f"Download/extraction completed: {len(extraction_results['successful_extractions'])}/{len(papers)} papers processed"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Download/extraction process failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_download_extract_agent() -> DownloadExtractAgent:
    """Factory function to create download/extract agent"""
    from agents.base_agent import agent_registry
    agent = DownloadExtractAgent()
    agent_registry.register_agent(agent)
    return agent
