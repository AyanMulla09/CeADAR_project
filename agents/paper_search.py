"""
Paper Search Agent - Searches for research papers using various sources
"""
import logging
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseReACTAgent
from tools.research_tools import arxiv_search_tool
from config import Config

logger = logging.getLogger(__name__)


class PaperSearchAgent(BaseReACTAgent):
    """Agent responsible for searching and collecting research papers"""
    
    def __init__(self):
        # Use the arxiv search tool
        tools: List[BaseTool] = [arxiv_search_tool]
        
        super().__init__(
            name="paper_search_agent",
            description="Searches for research papers from multiple academic sources",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for paper search agent"""
        template = """
You are a Paper Search Agent specialized in finding relevant research papers from academic databases.

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

Search for papers using the provided search queries. For each query:
1. Use the arxiv_search tool to find papers
2. Collect and organize the results
3. Remove duplicates
4. Provide summary statistics

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute paper search using the provided queries"""
        search_queries = input_data.get("search_queries", [])
        max_per_query = input_data.get("max_per_query", Config.MAX_PER_QUERY)
        
        if not search_queries:
            return {"error": "No search queries provided"}
        
        logger.info(f"Searching for papers using {len(search_queries)} queries")
        
        try:
            all_papers = []
            seen_titles = set()
            seen_urls = set()
            query_results = {}
            
            for idx, query in enumerate(search_queries):
                logger.info(f"Searching query {idx+1}/{len(search_queries)}: {query}")
                
                try:
                    # Search using arxiv tool
                    papers = arxiv_search_tool._run(query, max_per_query)
                    
                    # Filter duplicates
                    unique_papers = []
                    for paper in papers:
                        if isinstance(paper, dict) and "error" not in paper:
                            # Extract arxiv ID for deduplication
                            arxiv_id = paper.get("url", "").rsplit("/", 1)[-1].split("v")[0] if paper.get("url") else ""
                            title = paper.get("title", "")
                            
                            if title not in seen_titles and arxiv_id not in seen_urls:
                                seen_titles.add(title)
                                seen_urls.add(arxiv_id)
                                unique_papers.append(paper)
                                all_papers.append(paper)
                    
                    query_results[query] = {
                        "papers_found": len(papers),
                        "unique_papers": len(unique_papers),
                        "papers": unique_papers
                    }
                    
                    logger.info(f"Query '{query}': {len(papers)} found, {len(unique_papers)} unique")
                    
                except Exception as e:
                    error_msg = f"Error searching query '{query}': {str(e)}"
                    logger.error(error_msg)
                    query_results[query] = {"error": error_msg}
            
            # Compile final results
            result = {
                "total_papers_collected": len(all_papers),
                "total_queries_executed": len(search_queries),
                "successful_queries": len([q for q in query_results.values() if "error" not in q]),
                "papers": all_papers,
                "query_results": query_results,
                "search_statistics": {
                    "unique_titles": len(seen_titles),
                    "unique_urls": len(seen_urls),
                    "average_papers_per_query": len(all_papers) / len(search_queries) if search_queries else 0
                }
            }
            
            logger.info(f"Paper search completed: {len(all_papers)} unique papers collected")
            return result
            
        except Exception as e:
            error_msg = f"Paper search failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_paper_search_agent() -> PaperSearchAgent:
    """Factory function to create paper search agent"""
    from agents.base_agent import agent_registry
    agent = PaperSearchAgent()
    agent_registry.register_agent(agent)
    return agent
