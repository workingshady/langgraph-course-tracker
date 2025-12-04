import math
import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set")

from typing import Callable

from IPython.display import Image, display
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from typing_extensions import TypedDict


def multiply(a: float, b: float) -> float:
    """Multiply two floats.
    Args:
        a: first float
        b: second float
    """
    return a * b

def add(a: float, b: float) -> float:
    """Add two floats.
    Args:
        a: first float
        b: second float
    """
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract two floats.
    Args:
        a: first float
        b: second float
    """
    return a - b

def divide(a: float, b: float) -> float:
    """Divide two floats.
    Args:
        a: first float
        b: second float
    """
    return a / b

def power(a: float, b: float) -> float:
    """Power of two floats.
    Args:
        a: first float
        b: second float
    """
    return a ** b

def sqrt(a: float) -> float:
    """Square root of a float.
    Args:
        a: first float
    """
    return a ** 0.5

def log(a: float) -> float:
    """Logarithm of a float.
    Args:
        a: first float
    """
    return math.log(a)

MATH_TOOLS = [multiply, add, subtract, divide, power, sqrt, log]

class ReActAgent:

    SYSTEM_MESSAGE  = """
You are a helpful math assistant. Use the provided math tools to solve calculation problems.

When the user asks a math question:
1. Use appropriate tools to compute the exact answer
2. Show your step-by-step reasoning
3. Provide the final numerical answer clearly
4. Explain the result in simple terms
"""

    def __init__(self,
                 Model_name: str = "gemini-2.5-flash",
                 tools: list[Callable] = MATH_TOOLS,
                 temperature: float = 0.0):
        """Initialize ReAct agent with LLM and tools."""
        self.builder = StateGraph(MessagesState)
        self.tools = tools
        self.checkpointer = MemorySaver()
        self._setup_model(Model_name, temperature)
        self._setup_nodes()
        self._setup_edges()
        self.graph = self.builder.compile(checkpointer=self.checkpointer)

    def _setup_model(self, model_name: str, temperature: float):
        """Setup model WITH SYSTEM MESSAGE."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model=model_name,
            temperature=temperature
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _agent_node(self, state: MessagesState):
        """LLM decides: respond directly or call tool."""
        messages = [SystemMessage(content=self.SYSTEM_MESSAGE)] + state["messages"]
        return {"messages": [self.llm_with_tools.invoke(messages)]}

    def _setup_nodes(self):
        """Register all nodes"""
        self.builder.add_node("agent", self._agent_node)
        self.builder.add_node("tools", ToolNode(self.tools))

    def _setup_edges(self):
        """Register all edges"""
        self.builder.add_edge(START, "agent")
        self.builder.add_conditional_edges("agent", tools_condition)
        self.builder.add_edge("tools", "agent")
        self.builder.add_edge("agent", END)

    def invoke(self, messages: list[AnyMessage], config: dict = None):
        return self.graph.invoke({"messages": messages}, config=config)

    def stream(self, messages: list[AnyMessage], config: dict = None):
        return self.graph.stream({"messages": messages}, config=config)

    def show(self, filename: str = "graph_diagram.png"):
        """Display Mermaid diagram and save it as a PNG file."""
        png_data = self.graph.get_graph().draw_mermaid_png()
        # Save PNG to disk
        with open(filename, "wb") as f:
            f.write(png_data)
        display(Image(png_data))
        print(f"Diagram saved as {filename}")

graph = ReActAgent().graph

@traceable(name="ReActAgent")
def main():
    agent = ReActAgent()
    agent.show("outputs/react_agent_graph.png")

    thread_id = "math_session_1"  # Same thread_id = conversation memory
    config = {"configurable": {"thread_id": thread_id}}

    # Test 1: Simple math
    result1 = agent.invoke(
        [HumanMessage(content="calculate 3*3*3")],
        config
    )
    print("Result 1:", [msg.content for msg in result1["messages"] if isinstance(msg, AIMessage)][-1])

    # Test 2: Follow-up using previous result (memory works!)
    result2 = agent.invoke(
        [HumanMessage(content="double that result")],
        config  # Same thread_id loads previous state
    )
    print("Result 2:", [msg.content for msg in result2["messages"] if isinstance(msg, AIMessage)][-1])

if __name__ == "__main__":
    main()
