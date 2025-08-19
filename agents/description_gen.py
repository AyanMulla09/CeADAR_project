"""
Description Generation Agent - Generates detailed descriptions and summaries
"""
import json
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


class DescriptionGenAgent(BaseReACTAgent):
    """Agent responsible for generating detailed research topic descriptions for filtering"""
    
    def __init__(self):
        # Create tools for this agent
        tools: List[BaseTool] = [
            Tool(
                name="generate_topic_description",
                description="Generate a detailed description of the research topic for filtering",
                func=self._generate_topic_description
            ),
            Tool(
                name="elaborate_research_scope",
                description="Elaborate on the scope and boundaries of the research topic",
                func=self._elaborate_research_scope
            ),
            Tool(
                name="identify_key_concepts",
                description="Identify key concepts and terminology for the research topic",
                func=self._identify_key_concepts
            ),
            Tool(
                name="define_inclusion_criteria",
                description="Define what papers should be included for this research topic",
                func=self._define_inclusion_criteria
            )
        ]
        
        super().__init__(
            name="description_gen_agent",
            description="Generates detailed research topic descriptions and filtering criteria",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for description generation agent"""
        template = """
You are a Research Topic Description Agent that creates detailed, comprehensive descriptions of research topics for paper filtering.

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
1. Generate detailed topic descriptions for filtering
2. Define research scope and boundaries
3. Identify key concepts and terminology
4. Create inclusion/exclusion criteria

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
    
    def _generate_topic_description(self, topic_data: str) -> str:
        """Generate a comprehensive description of the research topic"""
        try:
            data = json.loads(topic_data) if isinstance(topic_data, str) else topic_data
            
            original_topic = data.get("original_topic", "")
            topic_elaboration = data.get("topic_elaboration", "")
            
            # Use elaboration if available, otherwise use original topic
            topic_text = topic_elaboration if topic_elaboration else original_topic
            
            prompt = f"""
Create a comprehensive research topic description for: {original_topic}

{f"Based on this elaboration: {topic_elaboration}" if topic_elaboration else ""}

Generate a detailed description (300-400 words) that includes:

1. **Core Research Focus**: What the main research area encompasses
2. **Key Domains**: Specific subfields and application areas
3. **Technical Aspects**: Important technical concepts, methods, and approaches
4. **Application Context**: Real-world applications and use cases
5. **Research Boundaries**: What should be included vs excluded

The description should help identify relevant research papers by clearly defining:
- What types of research papers are highly relevant
- What methodologies and approaches are important
- What applications and domains are in scope
- What keywords and concepts to look for

Write this as a comprehensive research scope definition that will be used to filter academic papers.
"""
            
            response = self.llm.invoke(prompt)
            
            return json.dumps({
                "original_topic": original_topic,
                "detailed_description": response.strip(),
                "description_length": len(response.strip().split()),
                "generation_method": "LLM-based topic elaboration"
            })
            
        except Exception as e:
            logger.error(f"Error generating topic description: {e}")
            return json.dumps({"error": str(e)})
    
    def _elaborate_research_scope(self, scope_data: str) -> str:
        """Elaborate on the research scope and boundaries"""
        try:
            data = json.loads(scope_data) if isinstance(scope_data, str) else scope_data
            topic = data.get("topic", "")
            
            prompt = f"""
Define the research scope and boundaries for: {topic}

Provide a structured analysis:

**INCLUDE papers that focus on:**
- [List 5-7 specific areas that should be included]

**PRIORITIZE papers that involve:**
- [List 3-5 high-priority research aspects]

**EXCLUDE papers that primarily focus on:**
- [List 3-5 areas that should be excluded]

**KEY METHODOLOGIES of interest:**
- [List relevant research methodologies]

**IMPORTANT KEYWORDS to look for:**
- [List 10-15 important keywords and phrases]

Make this specific and actionable for paper filtering.
"""
            
            response = self.llm.invoke(prompt)
            
            return json.dumps({
                "topic": topic,
                "scope_definition": response.strip(),
                "scope_type": "inclusion_exclusion_criteria"
            })
            
        except Exception as e:
            logger.error(f"Error elaborating research scope: {e}")
            return json.dumps({"error": str(e)})
    
    def _identify_key_concepts(self, concept_data: str) -> str:
        """Identify key concepts and terminology for the research topic"""
        try:
            data = json.loads(concept_data) if isinstance(concept_data, str) else concept_data
            topic = data.get("topic", "")
            
            prompt = f"""
Identify key concepts and terminology for the research topic: {topic}

Extract and organize:

1. **Core Technical Terms** (8-10 terms):
   - [List fundamental technical concepts]

2. **Related Fields and Disciplines** (5-7 areas):
   - [List related research areas]

3. **Common Methodologies** (5-6 methods):
   - [List typical research methodologies]

4. **Important Metrics and Measures** (4-5 metrics):
   - [List evaluation criteria and metrics]

5. **Application Domains** (6-8 domains):
   - [List practical application areas]

6. **Alternative Terms and Synonyms**:
   - [List alternative ways this topic might be described]

Format as a structured list that can be used for semantic matching and filtering.
"""
            
            response = self.llm.invoke(prompt)
            
            return json.dumps({
                "topic": topic,
                "key_concepts": response.strip(),
                "concept_extraction_method": "structured_terminology_analysis"
            })
            
        except Exception as e:
            logger.error(f"Error identifying key concepts: {e}")
            return json.dumps({"error": str(e)})
    
    def _define_inclusion_criteria(self, criteria_data: str) -> str:
        """Define specific inclusion criteria for paper filtering"""
        try:
            data = json.loads(criteria_data) if isinstance(criteria_data, str) else criteria_data
            topic = data.get("topic", "")
            description = data.get("description", "")
            
            prompt = f"""
Define specific inclusion criteria for filtering papers on: {topic}

{f"Based on this description: {description}" if description else ""}

Create filtering criteria:

**HIGHLY RELEVANT (Score 4-5) - Papers that:**
- [3-4 specific criteria for high relevance]

**MODERATELY RELEVANT (Score 3) - Papers that:**
- [3-4 criteria for moderate relevance]

**SLIGHTLY RELEVANT (Score 2) - Papers that:**
- [2-3 criteria for slight relevance]

**NOT RELEVANT (Score 1) - Papers that:**
- [3-4 criteria for exclusion]

**SPECIFIC INDICATORS TO LOOK FOR:**
- Abstract contains: [list key phrases]
- Methods include: [list methodologies]
- Applications involve: [list domains]
- Evaluation includes: [list metrics]

**RED FLAGS (automatic exclusion):**
- [List 3-4 clear exclusion signals]

Make these criteria specific enough for consistent application.
"""
            
            response = self.llm.invoke(prompt)
            
            return json.dumps({
                "topic": topic,
                "inclusion_criteria": response.strip(),
                "criteria_type": "scoring_based_filtering"
            })
            
        except Exception as e:
            logger.error(f"Error defining inclusion criteria: {e}")
            return json.dumps({"error": str(e)})
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute description generation for research topic"""
        original_topic = input_data.get("original_topic", input_data.get("topic", ""))
        topic_elaboration = input_data.get("topic_elaboration", "")
        
        if not original_topic:
            return {"error": "No research topic provided for description generation"}
        
        logger.info(f"Generating topic description for: {original_topic}")
        
        try:
            # Prepare topic data
            topic_data = {
                "original_topic": original_topic,
                "topic_elaboration": topic_elaboration
            }
            
            # Generate comprehensive topic description
            description_result = self._generate_topic_description(json.dumps(topic_data))
            description_data = json.loads(description_result)
            
            if "error" in description_data:
                return {"error": f"Failed to generate topic description: {description_data['error']}"}
            
            # Generate research scope definition
            scope_result = self._elaborate_research_scope(json.dumps({"topic": original_topic}))
            scope_data = json.loads(scope_result)
            
            # Identify key concepts
            concepts_result = self._identify_key_concepts(json.dumps({"topic": original_topic}))
            concepts_data = json.loads(concepts_result)
            
            # Define inclusion criteria
            criteria_result = self._define_inclusion_criteria(json.dumps({
                "topic": original_topic,
                "description": description_data.get("detailed_description", "")
            }))
            criteria_data = json.loads(criteria_result)
            
            result = {
                "original_topic": original_topic,
                "topic_description": description_data.get("detailed_description", ""),
                "research_scope": scope_data.get("scope_definition", ""),
                "key_concepts": concepts_data.get("key_concepts", ""),
                "inclusion_criteria": criteria_data.get("inclusion_criteria", ""),
                "description_stats": {
                    "description_length": description_data.get("description_length", 0),
                    "generation_method": description_data.get("generation_method", ""),
                },
                "description_generation_complete": True
            }
            
            logger.info(f"Topic description generation completed for: {original_topic}")
            return result
            
        except Exception as e:
            error_msg = f"Description generation failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


def create_description_gen_agent() -> DescriptionGenAgent:
    """Factory function to create description generation agent"""
    from agents.base_agent import agent_registry
    agent = DescriptionGenAgent()
    agent_registry.register_agent(agent)
    return agent
