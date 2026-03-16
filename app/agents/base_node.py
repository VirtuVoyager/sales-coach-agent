from abc import ABC, abstractmethod
from typing import Any, Union
from langgraph.types import Command
from langchain_core.runnables.config import RunnableConfig

class BaseNode(ABC):
    """
    Abstract base class for all LangGraph nodes in the system.
    """
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def ainvoke(self, state: dict[str, Any], config: RunnableConfig | None = None) -> Union[dict[str, Any], Command]:
        """
        Asynchronously executes the node's logic.
        
        Args:
            state: The current state of the LangGraph execution.
            config: Optional runtime configuration.
            
        Returns:
            A dictionary containing state updates OR a LangGraph Command object for routing.
        """
        pass
        
    def __call__(self, state: dict[str, Any], config: RunnableConfig | None = None) -> Union[dict[str, Any], Command]:
        """
        Allows the node instance to be called directly by LangGraph, wrapping the async execution.
        Note: LangGraph natively supports async nodes, so this simply maps to ainvoke.
        """
        # LangGraph handles async resolution automatically when passing coroutines.
        return self.ainvoke(state, config)