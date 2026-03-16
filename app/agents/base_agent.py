from abc import ABC, abstractmethod
from typing import Any, Dict
from langgraph.graph.state import CompiledStateGraph

class BaseAgent(ABC):
    """
    Abstract base class for defining and compiling sub-agents or orchestrators.
    """
    
    def __init__(self):
        self.graph: CompiledStateGraph = self._build_graph()

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """
        Defines the nodes and edges of the agent's state graph.
        
        Returns:
            A compiled LangGraph StateGraph.
        """
        pass

    async def ainvoke(self, state: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Executes the compiled agent graph asynchronously.
        
        Args:
            state: The input state to the agent.
            config: Optional runtime configuration (e.g., thread_id for checkpointer).
            
        Returns:
            The final output state of the agent's graph.
        """
        return await self.graph.ainvoke(state, config)