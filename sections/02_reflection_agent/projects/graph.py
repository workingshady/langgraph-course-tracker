import json
import logging
from typing import Literal, Optional

from langchain_core.messages import BaseMessage  # For type checking
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from nodes import critique_post, generate_post
from states import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkedInPostAgent:
    def __init__(
        self,
        max_attempts: int = 3,
    ):
        self.max_attempts = max_attempts

        logger.info(f"Initializing LinkedInPostAgent with max_attempts={self.max_attempts}")
        self._graph = self._build_graph()
        self._runner = self._graph.compile()

    def _route_post(self, state: AgentState) -> Literal["critique_post", "generate_post", "__end__"]:
        """Route based on number of attempts."""
        logger.debug(f"Routing state with num_attempts={state.get('num_attempts', None)}")
        if state["num_attempts"] < self.max_attempts:
            logger.info("Routing to 'critique_post'.")
            return "critique_post"
        else:
            logger.info("Routing to '__end__'. Maximum attempts reached.")
            return END

    def _build_graph(self) -> StateGraph:
        logger.info("Building workflow graph...")
        workflow = StateGraph(state_schema=AgentState)

        # Add all nodes
        workflow.add_node("generate_post", generate_post)
        workflow.add_node("critique_post", critique_post)
        logger.debug("Added nodes: generate_post, critique_post.")

        # Define edges
        workflow.add_edge(START, "generate_post")
        logger.debug("Added edge: START -> generate_post.")
        workflow.add_conditional_edges(
            "generate_post",
            self._route_post,
            {
                "critique_post": "critique_post",
                "__end__": END
            }
        )
        logger.debug("Added conditional edges for generate_post.")
        workflow.add_edge("critique_post", "generate_post")
        logger.debug("Added edge: critique_post -> generate_post.")

        logger.info("Workflow graph built.")
        return workflow

    def save_workflow_png(self, filename: str = "linkedin_workflow.png") -> str:
        """Save workflow visualization as PNG."""
        logger.info(f"Saving workflow visualization to {filename}...")
        graph_png = self._runner.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(graph_png)
        logger.info(f"Workflow saved: {filename}")
        return filename

    def run(self, topic: str) -> AgentState:
        """Run the agent with a topic."""
        logger.info(f"Running agent on topic: '{topic}'")
        initial_state = {
            "messages": [],
            "topic": topic,
            "generated_post": "",
            "critique": "",
            "num_attempts": 0,
        }
        result = self._runner.invoke(initial_state)
        logger.info("Agent run complete.")
        return result

if __name__ == "__main__":
    logger.info("Starting LinkedInPostAgent main run...")
    agent = LinkedInPostAgent()
    agent.save_workflow_png("linkedin_workflow.png")
    result = agent.run("How to become a software engineer")

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(message.content)
        elif isinstance(message, HumanMessage):
            print(message.content)
        elif isinstance(message, SystemMessage):
            print(message.content)
        elif isinstance(message, ToolMessage):
            print(message.content)
        else:
            print(message)
        print("--------------------------------")
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(result["generated_post"])
    logger.info("Agent run complete.")
