"""
Base Agent class for the ReACT AI Research Pipeline
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseReACTAgent(ABC):
    """Base class for all ReACT agents in the research pipeline"""
    
    def __init__(self, name: str, description: str, tools: List[BaseTool] | None = None):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent_executor = None
        self._initialize_agent()
    
    def _initialize_llm(self) -> Ollama:
        """Initialize the LLM for the agent"""
        return Ollama(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            num_predict=Config.MAX_TOKENS
        )
    
    def _initialize_agent(self):
        """Initialize the ReACT agent with tools and prompt"""
        if not self.tools:
            logger.warning(f"No tools provided for agent {self.name}")
            return
        
        prompt = self._create_prompt_template()
        
        # Create ReACT agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=Config.AGENT_VERBOSE,
            max_iterations=Config.AGENT_MAX_ITERATIONS,
            early_stopping_method=Config.AGENT_EARLY_STOPPING_METHOD,
            handle_parsing_errors=True
        )
    
    @abstractmethod
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for this specific agent"""
        pass
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main task"""
        pass
    
    def run(self, input_text: str, **kwargs) -> str:
        """Run the agent with input text"""
        if not self.agent_executor:
            raise ValueError(f"Agent {self.name} not properly initialized")
        
        try:
            result = self.agent_executor.invoke({
                "input": input_text,
                **kwargs
            })
            return result.get("output", "")
        except Exception as e:
            logger.error(f"Error running agent {self.name}: {e}")
            return f"Error: {str(e)}"
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        self._initialize_agent()  # Reinitialize with new tool
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear the agent's memory"""
        self.memory.clear()
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            "name": self.name,
            "description": self.description,
            "memory": self.memory.buffer,
            "conversation_history": [msg.dict() for msg in self.get_conversation_history()]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Clear and reload memory - using proper memory methods
            self.memory.clear()
            logger.info(f"Loaded state for agent {self.name}")
        except Exception as e:
            logger.error(f"Error loading state for agent {self.name}: {e}")


class AgentRegistry:
    """Registry to manage all agents in the pipeline"""
    
    def __init__(self):
        self.agents: Dict[str, BaseReACTAgent] = {}
    
    def register_agent(self, agent: BaseReACTAgent):
        """Register an agent"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseReACTAgent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self.agents.keys())
    
    def execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire pipeline with all agents"""
        results = {}
        current_data = input_data.copy()
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Executing agent: {agent_name}")
            try:
                result = agent.execute(current_data)
                results[agent_name] = result
                current_data.update(result)
            except Exception as e:
                logger.error(f"Error in agent {agent_name}: {e}")
                results[agent_name] = {"error": str(e)}
        
        return results


# Global agent registry
agent_registry = AgentRegistry()
