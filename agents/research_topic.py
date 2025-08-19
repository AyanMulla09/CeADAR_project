"""
Research Topic LLM Agent - Elaborates and expands research topics
"""
import logging
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain.tools import Tool, BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseReACTAgent
from config import Config

logger = logging.getLogger(__name__)


class ResearchTopicAgent(BaseReACTAgent):
    """Agent responsible for elaborating research topics and generating search queries"""
    
    def __init__(self):
        # Create tools for this agent
        tools: List[BaseTool] = [
            Tool(
                name="elaborate_topic",
                description="Elaborate on a research topic to provide detailed context",
                func=self._elaborate_topic
            ),
            Tool(
                name="generate_search_queries",
                description="Generate multiple search queries for comprehensive paper collection",
                func=self._generate_search_queries
            ),
            Tool(
                name="analyze_topic_scope",
                description="Analyze the scope and boundaries of a research topic",
                func=self._analyze_topic_scope
            )
        ]
        
        super().__init__(
            name="research_topic_agent",
            description="Elaborates research topics and generates comprehensive search strategies",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for research topic agent"""
        template = """
You are a Research Topic Analysis Agent. Your role is to:

1. Elaborate on research topics to provide comprehensive context
2. Generate diverse search queries to ensure comprehensive paper collection
3. Analyze topic scope and boundaries
4. Identify related sub-topics and keywords

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

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _elaborate_topic(self, topic: str) -> str:
        """Elaborate on a research topic"""
        prompt = f"""
You are an academic research assistant with deep expertise across multiple fields.

Please provide a comprehensive elaboration of the following research topic:

Topic: {topic}

Your elaboration should include:
1. Clear definition and scope of the topic
2. Key concepts and terminology
3. Current state of research in this area
4. Major challenges and open problems
5. Interdisciplinary connections
6. Recent developments and trends

Provide a detailed paragraph (150-200 words) that captures the essence and complexity of this research area.
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error elaborating topic: {e}")
            return f"Error elaborating topic: {str(e)}"
    
    def _generate_search_queries(self, topic: str) -> str:
        """Generate multiple search queries for comprehensive paper collection"""
        prompt = f"""
You are an expert academic researcher skilled in literature search strategies.

Generate a comprehensive list of search queries for the research topic: "{topic}"

Your queries should:
1. Cover different aspects and sub-topics
2. Include various terminology and synonyms
3. Target different research approaches (theoretical, empirical, applied)
4. Consider interdisciplinary perspectives
5. Include both broad and specific queries

Generate 8-12 diverse search queries that will ensure no relevant papers are missed.
Format as a numbered list, one query per line.

Example format:
1. [query 1]
2. [query 2]
...
"""
        
        try:
            response = self.llm.invoke(prompt)
            # Parse the response to extract queries
            lines = response.strip().split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and clean up
                    cleaned = line.split('.', 1)[-1].strip() if '.' in line else line
                    cleaned = cleaned.lstrip('-•').strip()
                    if cleaned:
                        queries.append(cleaned)
            
            return '\n'.join(queries)
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            return f"Error generating search queries: {str(e)}"
    
    def _analyze_topic_scope(self, topic: str) -> str:
        """Analyze the scope and boundaries of a research topic"""
        prompt = f"""
Analyze the scope and boundaries of this research topic: "{topic}"

Provide analysis on:
1. Primary research domain(s)
2. Related/adjacent fields
3. Temporal scope (recent vs. historical research)
4. Methodological approaches typically used
5. Key research questions in this area
6. Potential limitations or exclusions

Format your response as a structured analysis with clear sections.
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error analyzing topic scope: {e}")
            return f"Error analyzing topic scope: {str(e)}"
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research topic analysis"""
        topic = input_data.get("topic", "")
        if not topic:
            return {"error": "No research topic provided"}
        
        logger.info(f"Analyzing research topic: {topic}")
        
        try:
            # Step 1: Elaborate the topic
            elaboration = self._elaborate_topic(topic)
            
            # Step 2: Generate search queries
            search_queries_text = self._generate_search_queries(topic)
            search_queries = [q.strip() for q in search_queries_text.split('\n') if q.strip()]
            
            # Step 3: Analyze topic scope
            scope_analysis = self._analyze_topic_scope(topic)
            
            result = {
                "original_topic": topic,
                "topic_elaboration": elaboration,
                "search_queries": search_queries,
                "scope_analysis": scope_analysis,
                "num_queries_generated": len(search_queries)
            }
            
            logger.info(f"Generated {len(search_queries)} search queries for topic analysis")
            return result
            
        except Exception as e:
            error_msg = f"Research topic analysis failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_research_topic_agent() -> ResearchTopicAgent:
    """Factory function to create research topic agent"""
    from agents.base_agent import agent_registry
    agent = ResearchTopicAgent()
    agent_registry.register_agent(agent)
    return agent
