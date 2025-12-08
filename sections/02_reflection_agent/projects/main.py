import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from nodes import critique_post, generate_post, print_divider
from states import AgentState, get_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


def main():
    """Main entry point for the LinkedIn post generation agent."""
    print_divider()
    print("LinkedIn Post Generation Agent")
    print_divider()

    # Get initial state
    state = get_initial_state()

    # Get topic from user
    topic = input("Enter a topic for your LinkedIn post: ").strip()
    if not topic:
        logger.error("Topic cannot be empty")
        return

    # Update state with topic
    state["topic"] = topic
    state["messages"] = [HumanMessage(content=f"Write a LinkedIn post about: {topic}")]

    # Generate initial post
    logger.info("Generating post...")
    state = generate_post(state)
    print_divider()
    print("Generated Post:")
    print(state["generated_post"])
    print_divider()

    # Critique the post
    logger.info("Critiquing post...")
    state = critique_post(state)
    print_divider()
    print("Critique:")
    print(state["critique"])
    print_divider()


if __name__ == "__main__":
    main()

