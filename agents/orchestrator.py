"""
ReACT Orchestrator Agent - Coordinates the entire research pipeline
"""
import json
import logging
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseReACTAgent, agent_registry
from config import Config

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseReACTAgent):
    """Main orchestrator that coordinates all other agents in the research pipeline"""
    
    def __init__(self):
        super().__init__(
            name="orchestrator",
            description="Coordinates the entire research paper analysis pipeline",
            tools=[]  # Orchestrator doesn't use tools directly
        )
        self.pipeline_steps = [
            "research_topic_agent",
            "paper_search_agent", 
            "filter_agent",
            "description_gen_agent",
            "download_extract_agent",
            "fulltext_filter_agent",
            "gap_generation_agent",
            "embedding_indexing_agent"
        ]
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for orchestrator"""
        template = """
You are the Orchestrator Agent for a research paper analysis pipeline. Your role is to:

1. Coordinate the execution of multiple specialized agents
2. Manage data flow between agents
3. Handle errors and retry logic
4. Provide status updates and final results

Current pipeline steps: {pipeline_steps}

Available agents: {available_agents}

Your task: {input}

Current step: {current_step}
Previous results: {previous_results}

Think step by step about what needs to be done next and coordinate with the appropriate agent.

Thought: I need to analyze the current situation and determine the next action.
Action: Determine what to do next based on the pipeline state.
Observation: What I observe from the current state.
Thought: Based on the observation, I should...
Action: Take the appropriate next action.
Final Answer: Provide the result or next instruction.
"""
        
        return PromptTemplate(
            input_variables=[
                "input", "pipeline_steps", "available_agents", 
                "current_step", "previous_results"
            ],
            template=template
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire research pipeline"""
        logger.info("Starting research pipeline orchestration")
        
        results = {
            "status": "started",
            "input_data": input_data,
            "pipeline_results": {},
            "errors": [],
            "current_step": 0,
            "total_steps": len(self.pipeline_steps)
        }
        
        try:
            current_data = input_data.copy()
            
            for step_idx, agent_name in enumerate(self.pipeline_steps):
                logger.info(f"Executing step {step_idx + 1}/{len(self.pipeline_steps)}: {agent_name}")
                results["current_step"] = step_idx + 1
                
                agent = agent_registry.get_agent(agent_name)
                if not agent:
                    error_msg = f"Agent {agent_name} not found in registry"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue
                
                try:
                    # Execute the agent
                    step_result = agent.execute(current_data)
                    
                    results["pipeline_results"][agent_name] = step_result
                    
                    # Update current data with results for next agent
                    if isinstance(step_result, dict):
                        current_data.update(step_result)
                    
                    logger.info(f"Completed step {step_idx + 1}: {agent_name}")
                    
                except Exception as e:
                    error_msg = f"Error in {agent_name}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["pipeline_results"][agent_name] = {"error": str(e)}
            
            results["status"] = "completed"
            results["final_data"] = current_data
            
        except Exception as e:
            error_msg = f"Pipeline orchestration failed: {str(e)}"
            logger.error(error_msg)
            results["status"] = "failed"
            results["errors"].append(error_msg)
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline"""
        available_agents = agent_registry.list_agents()
        
        return {
            "orchestrator": self.name,
            "pipeline_steps": self.pipeline_steps,
            "available_agents": available_agents,
            "missing_agents": [
                step for step in self.pipeline_steps 
                if step not in available_agents
            ]
        }
    
    def run_single_step(self, step_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single step of the pipeline"""
        agent = agent_registry.get_agent(step_name)
        if not agent:
            return {"error": f"Agent {step_name} not found"}
        
        try:
            return agent.execute(input_data)
        except Exception as e:
            return {"error": f"Step {step_name} failed: {str(e)}"}
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """Validate that all required agents are available"""
        available_agents = agent_registry.list_agents()
        missing_agents = [
            step for step in self.pipeline_steps 
            if step not in available_agents
        ]
        
        validation_result = {
            "valid": len(missing_agents) == 0,
            "missing_agents": missing_agents,
            "available_agents": available_agents,
            "total_steps": len(self.pipeline_steps)
        }
        
        if validation_result["valid"]:
            logger.info("Pipeline validation successful - all agents available")
        else:
            logger.warning(f"Pipeline validation failed - missing agents: {missing_agents}")
        
        return validation_result
    
    def save_pipeline_results(self, results: Dict[str, Any], filepath: str):
        """Save pipeline results to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Pipeline results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")


def create_orchestrator() -> OrchestratorAgent:
    """Factory function to create and register orchestrator"""
    from agents.base_agent import agent_registry
    orchestrator = OrchestratorAgent()
    agent_registry.register_agent(orchestrator)
    return orchestrator
