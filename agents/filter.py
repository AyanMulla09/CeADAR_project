"""
Filter Agent - Filters papers based on relevance using LLM-based analysis
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


class FilterAgent(BaseReACTAgent):
    """Agent responsible for filtering papers based on relevance to the research topic"""
    
    def __init__(self):
        # Create tools for this agent
        tools: List[BaseTool] = [
            Tool(
                name="evaluate_paper_relevance",
                description="Evaluate if a paper is relevant to the research topic",
                func=self._evaluate_paper_relevance
            ),
            Tool(
                name="batch_filter_papers",
                description="Filter multiple papers for relevance in parallel",
                func=self._batch_filter_papers
            ),
            Tool(
                name="analyze_filtering_results",
                description="Analyze the results of paper filtering",
                func=self._analyze_filtering_results
            )
        ]
        
        super().__init__(
            name="filter_agent",
            description="Filters research papers based on relevance using LLM analysis",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for filter agent"""
        template = """
You are a Paper Filtering Agent that evaluates research papers for relevance to specific topics.

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
1. Analyze each paper's title and abstract
2. Determine relevance to the research topic
3. Use lenient filtering (when in doubt, include the paper)
4. Provide clear reasoning for decisions

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _evaluate_paper_relevance(self, paper_data: str) -> str:
        """Evaluate a single paper's relevance"""
        try:
            # Parse paper data (expecting JSON string or formatted text)
            import json
            if isinstance(paper_data, str):
                try:
                    paper_info = json.loads(paper_data)
                except:
                    # If not JSON, assume it's formatted text
                    paper_info = {"title": paper_data, "summary": ""}
            else:
                paper_info = paper_data
            
            title = paper_info.get("title", "")
            summary = paper_info.get("summary", "")
            topic = paper_info.get("topic", "")
            
            prompt = f"""
You are doing initial screening for papers for a survey on "{topic}".

Be LENIENT in this screening - this is just the first filtering step. Only exclude papers that are clearly unrelated.

Title: {title}
Abstract: {summary}

Is this paper potentially relevant to the topic "{topic}"? Consider:
- Does it relate to the topic directly or indirectly?
- Could it provide useful context, methods, or insights?
- Is it in a related field that might be valuable?

Be generous in your assessment. When in doubt, choose YES.

Respond with only "YES" or "NO".
"""
            
            response = self.llm.invoke(prompt)
            decision = self._parse_yes_no_decision(response.strip())
            
            return json.dumps({
                "decision": decision,
                "title": title,
                "reasoning": f"LLM decision: {decision}"
            })
            
        except Exception as e:
            logger.error(f"Error evaluating paper relevance: {e}")
            return json.dumps({"decision": "YES", "error": str(e)})  # Default to YES for lenient filtering
    
    def _batch_filter_papers(self, batch_data: str) -> str:
        """Filter multiple papers in parallel"""
        try:
            import json
            data = json.loads(batch_data)
            papers = data.get("papers", [])
            topic = data.get("topic", "")
            
            if not papers:
                return json.dumps({"error": "No papers provided"})
            
            def process_paper(paper):
                try:
                    prompt = f"""
You are doing initial screening for papers for a survey on "{topic}".

Be LENIENT in this screening - this is just the first filtering step. Only exclude papers that are clearly unrelated.

Title: {paper.get('title', '')}
Abstract: {paper.get('summary', '')}

Is this paper potentially relevant to the topic "{topic}"? Consider:
- Does it relate to the topic directly or indirectly?
- Could it provide useful context, methods, or insights?
- Is it in a related field that might be valuable?

Be generous in your assessment. When in doubt, choose YES.

Respond with only "YES" or "NO".
"""
                    
                    response = self.llm.invoke(prompt)
                    decision = self._parse_yes_no_decision(response.strip())
                    paper["abstract_decision"] = decision
                    return paper, decision
                    
                except Exception as e:
                    logger.error(f"Error processing paper '{paper.get('title', 'Unknown')}': {e}")
                    paper["abstract_decision"] = "YES"  # Default to YES for lenient filtering
                    return paper, "YES"
            
            relevant_papers = []
            rejected_count = 0
            
            # Process papers in parallel
            with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
                futures = [executor.submit(process_paper, paper) for paper in papers]
                
                for future in as_completed(futures):
                    paper, decision = future.result()
                    if decision == "YES":
                        relevant_papers.append(paper)
                    else:
                        rejected_count += 1
            
            results = {
                "relevant_papers": relevant_papers,
                "total_papers": len(papers),
                "relevant_count": len(relevant_papers),
                "rejected_count": rejected_count,
                "relevance_rate": len(relevant_papers) / len(papers) if papers else 0
            }
            
            return json.dumps(results)
            
        except Exception as e:
            logger.error(f"Error in batch filtering: {e}")
            return json.dumps({"error": str(e)})
    
    def _analyze_filtering_results(self, results_data: str) -> str:
        """Analyze the results of filtering"""
        try:
            import json
            results = json.loads(results_data)
            
            analysis = f"""
Filtering Analysis:
- Total papers processed: {results.get('total_papers', 0)}
- Papers marked as relevant: {results.get('relevant_count', 0)}
- Papers rejected: {results.get('rejected_count', 0)}
- Relevance rate: {results.get('relevance_rate', 0):.2%}

The filtering used a lenient approach, including papers when in doubt.
This ensures comprehensive coverage for the literature survey.
"""
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing results: {str(e)}"
    
    def _parse_yes_no_decision(self, response_text: str) -> str:
        """Parse Yes/No decision from LLM response"""
        response_upper = response_text.strip().upper()
        
        # Look for explicit YES/NO
        if "YES" in response_upper and "NO" not in response_upper:
            return "YES"
        elif "NO" in response_upper and "YES" not in response_upper:
            return "NO"
        elif "YES" in response_upper and "NO" in response_upper:
            # If both are present, look for the first one
            yes_pos = response_upper.find("YES")
            no_pos = response_upper.find("NO")
            return "YES" if yes_pos < no_pos else "NO"
        else:
            # Default to YES for lenient filtering when unclear
            return "YES"
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute paper filtering"""
        papers = input_data.get("papers", [])
        topic = input_data.get("original_topic", input_data.get("topic", ""))
        
        if not papers:
            return {"error": "No papers provided for filtering"}
        
        if not topic:
            return {"error": "No research topic provided for filtering"}
        
        logger.info(f"Filtering {len(papers)} papers for relevance to: {topic}")
        
        try:
            # Prepare batch data
            batch_data = {
                "papers": papers,
                "topic": topic
            }
            
            # Use batch filtering
            results_str = self._batch_filter_papers(json.dumps(batch_data))
            results = json.loads(results_str)
            
            if "error" in results:
                return {"error": results["error"]}
            
            # Get analysis
            analysis = self._analyze_filtering_results(results_str)
            
            final_result = {
                "filtered_papers": results["relevant_papers"],
                "filtering_statistics": {
                    "total_papers": results["total_papers"],
                    "relevant_count": results["relevant_count"],
                    "rejected_count": results["rejected_count"],
                    "relevance_rate": results["relevance_rate"]
                },
                "filtering_analysis": analysis,
                "filter_method": "LLM-based lenient abstract screening"
            }
            
            logger.info(f"Filtering completed: {results['relevant_count']}/{results['total_papers']} papers retained")
            return final_result
            
        except Exception as e:
            error_msg = f"Paper filtering failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_filter_agent() -> FilterAgent:
    """Factory function to create filter agent"""
    from agents.base_agent import agent_registry
    agent = FilterAgent()
    agent_registry.register_agent(agent)
    return agent
