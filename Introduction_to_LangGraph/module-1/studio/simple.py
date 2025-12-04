import random
from typing import Literal

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# 1. State Schema (separate class)
class SimpleState(TypedDict):
    graph_state: str

# 2. Node Functions (free functions, easier to test/reuse)
def node_1(state: SimpleState) -> dict:
    """Node 1: Add 'I am'"""
    print("---Node 1---")
    return {"graph_state": state['graph_state'] + " I am"}

def node_2(state: SimpleState) -> dict:
    """Node 2: Add 'happy!'"""
    print("---Node 2---")
    return {"graph_state": state['graph_state'] + " happy!"}

def node_3(state: SimpleState) -> dict:
    """Node 3: Add 'sad!'"""
    print("---Node 3---")
    return {"graph_state": state['graph_state'] + " sad!"}

def decide_mood(state: SimpleState) -> Literal["node_2", "node_3"]:
    """50/50 mood decision"""
    return "node_2" if random.random() < 0.5 else "node_3"

class SimpleMoodGraph:
    def __init__(self):
        self.builder = StateGraph(SimpleState)
        self._setup_nodes()
        self._setup_edges()
        self.graph = self.builder.compile()

    def _setup_nodes(self):
        """Register all nodes"""
        self.builder.add_node("node_1", node_1)
        self.builder.add_node("node_2", node_2)
        self.builder.add_node("node_3", node_3)

    def _setup_edges(self):
        """Register all edges"""
        self.builder.add_edge(START, "node_1")
        self.builder.add_conditional_edges("node_1", decide_mood,{
            "node_2": "node_2",
            "node_3": "node_3"
        })
        self.builder.add_edge("node_2", END)
        self.builder.add_edge("node_3", END)

    def invoke(self, input_text: str) -> SimpleState:
        return self.graph.invoke({"graph_state": input_text})

    def stream(self, input_text: str):
        return self.graph.stream({"graph_state": input_text})

    def show(self):
        """Display Mermaid diagram"""
        display(Image(self.graph.get_graph().draw_mermaid_png()))
graph = SimpleMoodGraph().graph
# Usage
if __name__ == "__main__":
    graph = SimpleMoodGraph()
    graph.show()
    result = graph.invoke("Hi, this is Lance.")
    print(result)
