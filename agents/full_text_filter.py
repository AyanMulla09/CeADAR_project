"""
Full Text Filter Agent - Performs detailed relevance filtering on full paper content
"""
import json
import logging
import re
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


class FullTextFilterAgent(BaseReACTAgent):
    """Agent responsible for detailed relevance filtering using full paper content"""
    
    def __init__(self):
        # Create tools for this agent
        tools: List[BaseTool] = [
            Tool(
                name="chunk_paper_content",
                description="Split paper content into semantic chunks for analysis",
                func=self._chunk_paper_content
            ),
            Tool(
                name="evaluate_chunk_relevance",
                description="Evaluate relevance of a paper chunk",
                func=self._evaluate_chunk_relevance
            ),
            Tool(
                name="aggregate_paper_score",
                description="Aggregate chunk scores to determine overall paper relevance",
                func=self._aggregate_paper_score
            ),
            Tool(
                name="batch_fulltext_filter",
                description="Filter multiple papers using full-text analysis",
                func=self._batch_fulltext_filter
            )
        ]
        
        super().__init__(
            name="fulltext_filter_agent",
            description="Performs detailed relevance filtering using full paper content analysis",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for full-text filter agent"""
        template = """
You are a Full Text Filter Agent that performs detailed relevance filtering on complete paper content.

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
1. Analyze full paper content for relevance
2. Use semantic analysis for filtering
3. Provide detailed relevance scores
4. Apply sophisticated filtering criteria

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _chunk_paper_content(self, content_data: str) -> str:
        """Split paper content into semantic chunks"""
        try:
            data = json.loads(content_data) if isinstance(content_data, str) else content_data
            text = data.get("text", "")
            max_chunk_words = data.get("max_chunk_words", Config.MAX_CHUNK_WORDS)
            overlap_words = data.get("overlap_words", Config.OVERLAP_WORDS)
            
            if not text:
                return json.dumps({"error": "No text provided"})
            
            # Simple paragraph-based chunking
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            current_chunk = []
            current_word_count = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_words = para.split()
                para_word_count = len(para_words)
                
                # If adding this paragraph would exceed the limit
                if current_word_count + para_word_count > max_chunk_words and current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if overlap_words > 0 and current_chunk:
                        last_para_words = current_chunk[-1].split()
                        if len(last_para_words) > overlap_words:
                            overlap_text = " ".join(last_para_words[-overlap_words:])
                            current_chunk = [overlap_text, para]
                            current_word_count = overlap_words + para_word_count
                        else:
                            current_chunk = [para]
                            current_word_count = para_word_count
                    else:
                        current_chunk = [para]
                        current_word_count = para_word_count
                else:
                    # Add paragraph to current chunk
                    current_chunk.append(para)
                    current_word_count += para_word_count
            
            # Add final chunk if it has content
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)
            
            return json.dumps({
                "chunks": chunks,
                "num_chunks": len(chunks),
                "total_words": len(text.split()),
                "average_chunk_size": sum(len(chunk.split()) for chunk in chunks) / len(chunks) if chunks else 0
            })
            
        except Exception as e:
            logger.error(f"Error chunking content: {e}")
            return json.dumps({"error": str(e)})
    
    def _evaluate_chunk_relevance(self, chunk_data: str) -> str:
        """Evaluate relevance of a paper chunk"""
        try:
            data = json.loads(chunk_data) if isinstance(chunk_data, str) else chunk_data
            chunk_text = data.get("chunk", "")
            topic = data.get("topic", "")
            chunk_index = data.get("chunk_index", 0)
            
            if not chunk_text or not topic:
                return json.dumps({"error": "Missing chunk text or topic"})
            
            prompt = f"""
You are writing a survey on the following detailed research topic:

{topic}

Below is a chunk from a research paper. Rate its relevance on a scale of 1-5:
5 = Essential/Highly relevant - directly addresses the topic
4 = Very relevant - strongly related with useful insights
3 = Moderately relevant - some relevant content
2 = Slightly relevant - tangentially related
1 = Not relevant - unrelated to the topic

Chunk {chunk_index + 1}:
\"\"\"{chunk_text[:1000]}...\"\"\"

Respond with:
Relevance: [1-5]
Reasoning: [Brief explanation]
"""
            
            response = self.llm.invoke(prompt)
            
            # Parse the response
            score_match = re.search(r"Relevance:\s*([1-5])", response)
            score = int(score_match.group(1)) if score_match else 3
            
            reasoning_match = re.search(r"Reasoning:\s*(.+)", response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return json.dumps({
                "chunk_index": chunk_index,
                "relevance_score": score,
                "reasoning": reasoning,
                "chunk_length": len(chunk_text.split())
            })
            
        except Exception as e:
            logger.error(f"Error evaluating chunk relevance: {e}")
            return json.dumps({"error": str(e)})
    
    def _aggregate_paper_score(self, scores_data: str) -> str:
        """Aggregate chunk scores to determine overall paper relevance"""
        try:
            data = json.loads(scores_data) if isinstance(scores_data, str) else scores_data
            chunk_scores = data.get("chunk_scores", [])
            
            if not chunk_scores:
                return json.dumps({"error": "No chunk scores provided"})
            
            # Calculate various aggregation metrics
            scores = [score.get("relevance_score", 0) for score in chunk_scores]
            max_score = max(scores) if scores else 0
            avg_score = sum(scores) / len(scores) if scores else 0
            high_relevance_chunks = len([s for s in scores if s >= 4])
            total_chunks = len(scores)
            
            # Decision logic: paper is relevant if max score >= 4 or average >= 3.5
            is_relevant = max_score >= 4 or avg_score >= 3.5
            
            result = {
                "is_relevant": is_relevant,
                "max_score": max_score,
                "average_score": avg_score,
                "high_relevance_chunks": high_relevance_chunks,
                "total_chunks": total_chunks,
                "relevance_ratio": high_relevance_chunks / total_chunks if total_chunks > 0 else 0,
                "decision_reasoning": (
                    f"Max score: {max_score}, Avg score: {avg_score:.2f}, "
                    f"High relevance chunks: {high_relevance_chunks}/{total_chunks}"
                )
            }
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error aggregating scores: {e}")
            return json.dumps({"error": str(e)})
    
    def _batch_fulltext_filter(self, papers_data: str) -> str:
        """Filter multiple papers using full-text analysis"""
        try:
            data = json.loads(papers_data) if isinstance(papers_data, str) else papers_data
            papers_with_content = data.get("papers", [])
            
            # Use topic_description first, fallback to topic
            topic_description = data.get("topic_description", data.get("topic", ""))
            
            if not papers_with_content:
                return json.dumps({"error": "No papers with content provided"})
            
            if not topic_description:
                return json.dumps({"error": "No topic description provided"})
            
            def process_paper(paper):
                try:
                    extracted_text = paper.get("extracted_text", "")
                    if not extracted_text:
                        return {
                            "paper": paper,
                            "is_relevant": False,
                            "error": "No extracted text found"
                        }
                    
                    # Chunk the paper
                    chunk_data = {"text": extracted_text}
                    chunks_result = self._chunk_paper_content(json.dumps(chunk_data))
                    chunks_info = json.loads(chunks_result)
                    
                    if "error" in chunks_info:
                        return {
                            "paper": paper,
                            "is_relevant": False,
                            "error": chunks_info["error"]
                        }
                    
                    chunks = chunks_info.get("chunks", [])
                    
                    # Evaluate each chunk
                    chunk_scores = []
                    for i, chunk in enumerate(chunks):
                        eval_data = {
                            "chunk": chunk,
                            "topic": topic_description,  # Use topic_description instead of topic
                            "chunk_index": i
                        }
                        eval_result = self._evaluate_chunk_relevance(json.dumps(eval_data))
                        eval_info = json.loads(eval_result)
                        
                        if "error" not in eval_info:
                            chunk_scores.append(eval_info)
                    
                    # Aggregate scores
                    agg_data = {"chunk_scores": chunk_scores}
                    agg_result = self._aggregate_paper_score(json.dumps(agg_data))
                    agg_info = json.loads(agg_result)
                    
                    # Combine paper info with relevance assessment
                    result = {
                        "paper": paper,
                        "is_relevant": agg_info.get("is_relevant", False),
                        "relevance_analysis": agg_info,
                        "chunk_analysis": {
                            "total_chunks": len(chunks),
                            "chunk_scores": chunk_scores
                        }
                    }
                    
                    return result
                    
                except Exception as e:
                    return {
                        "paper": paper,
                        "is_relevant": False,
                        "error": f"Processing error: {str(e)}"
                    }
            
            # Process papers in parallel
            relevant_papers = []
            irrelevant_papers = []
            error_papers = []
            
            with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
                futures = [executor.submit(process_paper, paper) for paper in papers_with_content]
                
                for future in as_completed(futures):
                    result = future.result()
                    
                    if "error" in result:
                        error_papers.append(result)
                    elif result.get("is_relevant"):
                        relevant_papers.append(result)
                    else:
                        irrelevant_papers.append(result)
            
            return json.dumps({
                "relevant_papers": relevant_papers,
                "irrelevant_papers": irrelevant_papers,
                "error_papers": error_papers,
                "filtering_stats": {
                    "total_papers": len(papers_with_content),
                    "relevant_count": len(relevant_papers),
                    "irrelevant_count": len(irrelevant_papers),
                    "error_count": len(error_papers),
                    "relevance_rate": len(relevant_papers) / len(papers_with_content) if papers_with_content else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error in batch filtering: {e}")
            return json.dumps({"error": str(e)})
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full-text filtering"""
        papers_with_content = input_data.get("papers_with_content", [])
        
        # Try to get the detailed topic description from description_gen_agent
        topic_description = input_data.get("topic_description", "")
        
        # Fallback to topic elaboration or original topic if no description available
        if not topic_description:
            topic_description = input_data.get("topic_elaboration", input_data.get("original_topic", ""))
        
        if not papers_with_content:
            return {"error": "No papers with extracted content provided for full-text filtering"}
        
        if not topic_description:
            return {"error": "No research topic or description provided for full-text filtering"}
        
        logger.info(f"Starting full-text filtering for {len(papers_with_content)} papers")
        logger.info(f"Using topic description: {topic_description[:100]}...")
        
        try:
            # Prepare data for batch filtering
            filter_data = {
                "papers": papers_with_content,
                "topic_description": topic_description
            }
            
            # Perform batch filtering
            filter_result = self._batch_fulltext_filter(json.dumps(filter_data))
            filter_info = json.loads(filter_result)
            
            if "error" in filter_info:
                return {"error": filter_info["error"]}
            
            # Extract relevant papers for next stage
            relevant_papers = []
            for result in filter_info["relevant_papers"]:
                paper = result["paper"]
                paper["fulltext_analysis"] = result["relevance_analysis"]
                relevant_papers.append(paper)
            
            final_result = {
                "fulltext_filtered_papers": relevant_papers,
                "fulltext_filtering_stats": filter_info["filtering_stats"],
                "irrelevant_papers_count": len(filter_info["irrelevant_papers"]),
                "error_papers_count": len(filter_info["error_papers"]),
                "filter_method": "LLM-based chunk-level relevance scoring"
            }
            
            logger.info(
                f"Full-text filtering completed: {len(relevant_papers)}/{len(papers_with_content)} papers retained"
            )
            
            return final_result
            
        except Exception as e:
            error_msg = f"Full-text filtering failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_fulltext_filter_agent() -> FullTextFilterAgent:
    """Factory function to create full-text filter agent"""
    from agents.base_agent import agent_registry
    agent = FullTextFilterAgent()
    agent_registry.register_agent(agent)
    return agent
