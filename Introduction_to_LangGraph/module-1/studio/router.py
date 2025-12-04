import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set")

from typing import Callable

from IPython.display import Image, display
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    pass

class RouterAgent:
    def __init__(self,
                 Model_name: str="gemini-2.5-flash",
                 tools: list[Callable]=[multiply],
                 temperature: float=0.0):
        """Initialize router agent with LLM and tools."""
        self.builder = StateGraph(MessagesState)
        self.tools = tools
        self._setup_model(Model_name, temperature)
        self._setup_nodes()
        self._setup_edges()
        self.graph = self.builder.compile()


    def _setup_model(self, model_name: str, temperature: float):
        """Setup model"""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model=model_name,
            temperature=temperature
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)


    def _agent_node(self, state: MessagesState):
        """LLM decides: respond directly or call tool."""
        return {"messages": [self.llm_with_tools.invoke(state["messages"])]}

    def _setup_nodes(self):
        """Register all nodes"""
        self.builder.add_node("agent", self._agent_node)
        self.builder.add_node("tools", ToolNode(self.tools))

    def _setup_edges(self):
        """Register all edges"""
        self.builder.add_edge(START, "agent")
        self.builder.add_conditional_edges("agent", tools_condition)
        self.builder.add_edge("tools", END)


    def invoke(self, messages: list[AnyMessage]):
        return self.graph.invoke({"messages": messages},
                                 )

    def stream(self, messages: list[AnyMessage]):
        return self.graph.stream({"messages": messages},
                                 )

    def show(self, filename: str = "graph_diagram.png"):
        """Display Mermaid diagram and save it as a PNG file."""
        png_data = self.graph.get_graph().draw_mermaid_png()
        # Save PNG to disk
        with open(filename, "wb") as f:
            f.write(png_data)
        display(Image(png_data))
        print(f"Diagram saved as {filename}")

graph = RouterAgent().graph


@traceable(name="RouterAgent")
def main():
    # Create agent
    agent = RouterAgent()

    # Show graph and save png
    agent.show("outputs/router_agent_graph.png")

    # Test cases
    test_cases = [
        "What is 3.5 * 2.1?",
        "Hello, how are you?",
        "3*3*3"
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        messages = [HumanMessage(content=query)]
        result = agent.invoke(messages)

        for msg in result["messages"]:
            msg.pretty_print()


if __name__ == "__main__":
    main()
