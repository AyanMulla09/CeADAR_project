"""
Gap Generation Agent - Analyzes research gaps that papers address
"""
import json
import logging
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.prompts import PromptTemplate
from langchain.tools import Tool, BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseReACTAgent
from config import Config

logger = logging.getLogger(__name__)


class GapGenerationAgent(BaseReACTAgent):
    """Agent responsible for analyzing research gaps that papers address"""
    
    def __init__(self):
        # Create tools for this agent
        tools: List[BaseTool] = [
            Tool(
                name="analyze_single_paper_gaps",
                description="Analyze research gaps addressed by a single paper",
                func=self._analyze_single_paper_gaps
            ),
            Tool(
                name="extract_research_keywords",
                description="Extract key research keywords from gap analysis",
                func=self._extract_research_keywords
            ),
            Tool(
                name="categorize_research_approaches",
                description="Categorize research approaches and methodologies",
                func=self._categorize_research_approaches
            ),
            Tool(
                name="batch_gap_analysis",
                description="Perform gap analysis on multiple papers",
                func=self._batch_gap_analysis
            )
        ]
        
        super().__init__(
            name="gap_generation_agent",
            description="Analyzes research gaps and contributions that papers address",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for gap generation agent"""
        template = """
You are a Gap Generation Agent that analyzes research gaps that papers address.

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
1. Identify research gaps addressed by papers
2. Analyze methodological contributions
3. Assess theoretical contributions
4. Provide gap analysis summaries

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _analyze_single_paper_gaps(self, paper_data: str) -> str:
        """Analyze research gaps addressed by a single paper"""
        try:
            data = json.loads(paper_data) if isinstance(paper_data, str) else paper_data
            paper = data.get("paper", {})
            topic = data.get("topic", "")
            extracted_text = paper.get("extracted_text", "")
            
            if not extracted_text:
                return json.dumps({"error": "No extracted text available"})
            
            # Limit text for processing
            text_chunk = extracted_text[:Config.MAX_GAP_CHARS]
            
            prompt = f"""
You are an expert academic research assistant writing a survey.

Given the detailed topic:

{topic}

And the following paper's content:

{text_chunk}

Provide a comprehensive analysis in THREE distinct sections:

1. **RESEARCH GAPS ADDRESSED**: What specific research gaps or problems does this paper explicitly address or discuss? Focus on the problems/gaps the paper claims to solve rather than what it doesn't cover.

2. **RESEARCH DIRECTION**: What is the overall research direction/approach this paper takes? (e.g., "Machine learning-based optimization", "Empirical study of user behavior", "Novel algorithmic approach", etc.)

3. **SOLUTION APPROACH**: How does this paper solve or attempt to solve the identified gaps? What methods, techniques, or frameworks do they propose or use?

Format your response exactly as:

RESEARCH_GAPS: [120-180 words describing the specific gaps this paper addresses]

RESEARCH_DIRECTION: [20-40 words describing the overall research direction/approach]

SOLUTION_APPROACH: [60-100 words describing how they solved the gaps]
"""
            
            response = self.llm.invoke(prompt)
            
            # Parse the structured response
            gap_text, direction, solution = self._parse_gap_analysis_response(response.strip())
            
            return json.dumps({
                "title": paper.get("title", ""),
                "url": paper.get("url", ""),
                "research_gap": gap_text,
                "gap_direction": direction,
                "gap_solution": solution,
                "analysis_complete": True
            })
            
        except Exception as e:
            logger.error(f"Error analyzing paper gaps: {e}")
            return json.dumps({"error": str(e)})
    
    def _parse_gap_analysis_response(self, response_text: str) -> tuple[str, str, str]:
        """Parse the structured gap analysis response into three components"""
        # Initialize defaults
        gap_text = "Gap analysis not available"
        direction = "Research direction not specified"
        solution = "Solution approach not specified"
        
        # Clean up the response text by removing markdown formatting
        cleaned_text = response_text.replace("**", "").replace("*", "")
        
        # Use regex to extract each section more precisely
        import re
        
        # Extract RESEARCH_GAPS section
        gaps_match = re.search(r'RESEARCH_GAPS\s*:?\s*\n?(.*?)(?=\n\s*RESEARCH_DIRECTION|\n\s*SOLUTION_APPROACH|$)', cleaned_text, re.DOTALL | re.IGNORECASE)
        if gaps_match:
            gap_text = gaps_match.group(1).strip()
        
        # Extract RESEARCH_DIRECTION section
        direction_match = re.search(r'RESEARCH_DIRECTION\s*:?\s*\n?(.*?)(?=\n\s*SOLUTION_APPROACH|\n\s*RESEARCH_GAPS|$)', cleaned_text, re.DOTALL | re.IGNORECASE)
        if direction_match:
            direction = direction_match.group(1).strip()
        
        # Extract SOLUTION_APPROACH section
        solution_match = re.search(r'SOLUTION_APPROACH\s*:?\s*\n?(.*?)(?=\n\s*RESEARCH_DIRECTION|\n\s*RESEARCH_GAPS|$)', cleaned_text, re.DOTALL | re.IGNORECASE)
        if solution_match:
            solution = solution_match.group(1).strip()
        
        return gap_text, direction, solution
    
    def _extract_research_keywords(self, gap_text: str) -> str:
        """Extract key research keywords from gap analysis text"""
        try:
            prompt = f"""
Extract the top {Config.KEYWORD_EXTRACTION_TOP_N} most important research keywords or phrases from this research gap analysis.
Output ONLY the keywords/phrases, one per line, no numbers, no explanations, no extra text.

Text: {gap_text}
"""
            
            response = self.llm.invoke(prompt)
            keywords_text = response.strip()
            
            # Clean up the response - remove numbering, extra text, etc.
            lines = keywords_text.split('\n')
            keywords = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove common prefixes like "1.", "2.", "-", "*", "Here are", etc.
                import re
                line = re.sub(r'^\d+\.?\s*', '', line)  # Remove "1. " "2. " etc
                line = re.sub(r'^[-*•]\s*', '', line)   # Remove "- " "* " "• " etc
                line = line.replace('Here are the top', '').replace('research keywords', '').replace(':', '')
                line = line.strip()
                
                # Skip lines that are just instructions or headers
                if line and not any(skip in line.lower() for skip in ['here are', 'keywords', 'phrases', 'most important']):
                    keywords.append(line)
            
            # Take only the requested number
            keywords = keywords[:Config.KEYWORD_EXTRACTION_TOP_N]
            
            return json.dumps({
                "keywords": keywords,
                "num_keywords": len(keywords)
            })
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return json.dumps({"error": str(e)})
    
    def _categorize_research_approaches(self, papers_data: str) -> str:
        """Categorize research approaches and methodologies"""
        try:
            data = json.loads(papers_data) if isinstance(papers_data, str) else papers_data
            papers = data.get("papers", [])
            
            categories = {
                "theoretical": [],
                "empirical": [],
                "experimental": [],
                "applied": [],
                "survey_review": [],
                "algorithmic": [],
                "system_design": [],
                "other": []
            }
            
            for paper in papers:
                direction = paper.get("gap_direction", "").lower()
                solution = paper.get("gap_solution", "").lower()
                title = paper.get("title", "").lower()
                
                combined_text = f"{direction} {solution} {title}"
                
                # Categorize based on content analysis
                if any(word in combined_text for word in ["theoretical", "theory", "framework", "model", "conceptual"]):
                    categories["theoretical"].append(paper)
                elif any(word in combined_text for word in ["empirical", "study", "analysis", "investigation", "case study"]):
                    categories["empirical"].append(paper)
                elif any(word in combined_text for word in ["experiment", "evaluation", "performance", "benchmark", "testing"]):
                    categories["experimental"].append(paper)
                elif any(word in combined_text for word in ["algorithm", "optimization", "computational", "mathematical"]):
                    categories["algorithmic"].append(paper)
                elif any(word in combined_text for word in ["system", "implementation", "architecture", "design", "tool"]):
                    categories["system_design"].append(paper)
                elif any(word in combined_text for word in ["application", "applied", "practical", "real-world"]):
                    categories["applied"].append(paper)
                elif any(word in combined_text for word in ["survey", "review", "overview", "comprehensive"]):
                    categories["survey_review"].append(paper)
                else:
                    categories["other"].append(paper)
            
            summary = {
                "category_counts": {k: len(v) for k, v in categories.items()},
                "total_papers": len(papers),
                "categories": categories
            }
            
            return json.dumps(summary)
            
        except Exception as e:
            logger.error(f"Error categorizing approaches: {e}")
            return json.dumps({"error": str(e)})
    
    def _batch_gap_analysis(self, papers_data: str) -> str:
        """Perform gap analysis on multiple papers"""
        try:
            data = json.loads(papers_data) if isinstance(papers_data, str) else papers_data
            papers = data.get("papers", [])
            topic = data.get("topic", "")
            
            if not papers:
                return json.dumps({"error": "No papers provided"})
            
            def process_paper(paper):
                try:
                    analysis_data = {
                        "paper": paper,
                        "topic": topic
                    }
                    result = self._analyze_single_paper_gaps(json.dumps(analysis_data))
                    analysis = json.loads(result)
                    
                    if "error" not in analysis:
                        # Extract keywords for this paper
                        keywords_result = self._extract_research_keywords(analysis.get("research_gap", ""))
                        keywords_data = json.loads(keywords_result)
                        
                        if "error" not in keywords_data:
                            analysis["gap_keywords"] = keywords_data.get("keywords", [])
                        else:
                            analysis["gap_keywords"] = []
                    
                    return analysis
                    
                except Exception as e:
                    return {
                        "title": paper.get("title", "Unknown"),
                        "error": f"Analysis failed: {str(e)}"
                    }
            
            # Process papers in parallel
            results = []
            with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
                futures = [executor.submit(process_paper, paper) for paper in papers]
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # Separate successful and failed analyses
            successful_analyses = [r for r in results if "error" not in r]
            failed_analyses = [r for r in results if "error" in r]
            
            return json.dumps({
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "analysis_stats": {
                    "total_papers": len(papers),
                    "successful": len(successful_analyses),
                    "failed": len(failed_analyses),
                    "success_rate": len(successful_analyses) / len(papers) if papers else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error in batch gap analysis: {e}")
            return json.dumps({"error": str(e)})
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gap generation analysis"""
        papers = input_data.get("fulltext_filtered_papers", input_data.get("papers_with_content", []))
        topic = input_data.get("topic_elaboration", input_data.get("original_topic", ""))
        
        if not papers:
            return {"error": "No papers provided for gap analysis"}
        
        if not topic:
            return {"error": "No research topic provided for gap analysis"}
        
        logger.info(f"Starting gap analysis for {len(papers)} papers")
        
        try:
            # Perform batch gap analysis
            analysis_data = {
                "papers": papers,
                "topic": topic
            }
            
            batch_result = self._batch_gap_analysis(json.dumps(analysis_data))
            batch_info = json.loads(batch_result)
            
            if "error" in batch_info:
                return {"error": batch_info["error"]}
            
            successful_analyses = batch_info["successful_analyses"]
            
            # Categorize research approaches
            if successful_analyses:
                categorization_result = self._categorize_research_approaches(
                    json.dumps({"papers": successful_analyses})
                )
                categorization_info = json.loads(categorization_result)
            else:
                categorization_info = {"error": "No successful analyses to categorize"}
            
            # Prepare final papers with gap analysis
            papers_with_gaps = []
            for analysis in successful_analyses:
                if "error" not in analysis:
                    # Find the original paper to preserve extracted_text
                    original_paper = None
                    for paper in papers:
                        if paper.get("title") == analysis.get("title") or paper.get("url") == analysis.get("url"):
                            original_paper = paper
                            break
                    
                    paper_with_gap = {
                        "title": analysis.get("title", ""),
                        "url": analysis.get("url", ""),
                        "extracted_text": original_paper.get("extracted_text", "") if original_paper else "",
                        "research_gap": analysis.get("research_gap", ""),
                        "gap_direction": analysis.get("gap_direction", ""),
                        "gap_solution": analysis.get("gap_solution", ""),
                        "analysis_complete": analysis.get("analysis_complete", False),
                        "gap_keywords": analysis.get("gap_keywords", [])
                    }
                    papers_with_gaps.append(paper_with_gap)
            
            result = {
                "papers_with_gap_analysis": papers_with_gaps,
                "gap_analysis_stats": batch_info["analysis_stats"],
                "research_approach_categorization": categorization_info,
                "failed_analyses_count": len(batch_info["failed_analyses"]),
                "analysis_method": "LLM-based structured gap identification"
            }
            
            logger.info(f"Gap analysis completed: {len(papers_with_gaps)} papers analyzed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Gap generation analysis failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_gap_generation_agent() -> GapGenerationAgent:
    """Factory function to create gap generation agent"""
    from agents.base_agent import agent_registry
    agent = GapGenerationAgent()
    agent_registry.register_agent(agent)
    return agent
