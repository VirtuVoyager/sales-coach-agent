from langgraph.graph import StateGraph, START
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.base_agent import BaseAgent
from app.agents.state import SimulationState
from app.agents.orchestrator.router import OrchestratorNode
from app.agents.sales_exec.node import SalesExecNode
from app.agents.customer.node import CustomerNode
from app.core.config import settings
from app.core.llm import get_llm

class SimulationOrchestrator(BaseAgent):
    """
    The master graph that binds the multi-agent system together.
    Handles the initialization of the LLMs, nodes, and state management.
    """
    def __init__(self, checkpointer: BaseCheckpointSaver):
        self.checkpointer = checkpointer
        
        # Initialize the shared Azure OpenAI LLM using secure settings
        self.llm = get_llm()
        
        # Calling super() automatically triggers self._build_graph()
        super().__init__()

    def _build_graph(self):
        """
        Constructs the state graph, adds nodes, and attaches the checkpointer.
        """
        builder = StateGraph(SimulationState)
        
        # 1. Instantiate Nodes
        orchestrator_node = OrchestratorNode("orchestrator", self.llm)
        sales_exec_node = SalesExecNode("sales_exec", self.llm)
        customer_node = CustomerNode("customer", self.llm)
        
        # 2. Add Nodes to the Graph
        # We pass the async ainvoke method of each node to LangGraph
        builder.add_node("orchestrator", orchestrator_node.ainvoke)
        builder.add_node("sales_exec", sales_exec_node.ainvoke)
        builder.add_node("customer", customer_node.ainvoke)
        
        # 3. Define the Entry Point
        # The simulation always starts with the Orchestrator evaluating the state
        builder.add_edge(START, "orchestrator")
        
        # NOTE ON EDGES:
        # We do NOT need to define `add_conditional_edges` or outbound edges here.
        # Because our nodes return `Command(goto="...")`, LangGraph dynamically 
        # handles the routing at runtime. This keeps our graph definition highly modular.
        
        # 4. Compile the Graph with Short-Term Memory
        return builder.compile(checkpointer=self.checkpointer)