# nodes.py
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from states import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def get_llm() -> ChatOllama:
    """Factory function to create and return Ollama LLM instance."""
    try:
        llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0.1,
        )
        logger.info(f"Successfully initialized {llm.model}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

LLM_MODEL = get_llm()

def generate_post(state: AgentState) -> dict[str, Any]:
    """Generate LinkedIn post based on topic and previous critique."""
    logger.info(f"Starting post generation for topic: {state['topic']} | Attempt #{state['num_attempts']+1}")

    # Include topic and previous critique in context
    system_prompt = """You are a LinkedIn influencer writing viral AI engineer posts.
    Write engaging posts (200-300 words) with hooks, insights, emojis, and calls-to-action.
    Make them professional yet conversational."""

    if state["critique"]:
        logger.debug(f"Including previous critique in system prompt")
        system_prompt += f"\n\nPREVIOUS CRITIQUE (address these issues):\n{state['critique']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", f"Topic: {state['topic']}"),
    ])

    chain = prompt | LLM_MODEL

    try:
        logger.info("Invoking LLM chain to generate post.")
        response = chain.invoke({"messages": state["messages"]})
        post = response.content
        logger.info("Post generation complete.")
    except Exception as e:
        logger.error(f"Error during post generation: {e}")
        raise

    logger.debug(f"Generated post preview: {post[:100]}...")

    return {
        "messages": [HumanMessage(content=f"Generated post (attempt {state['num_attempts']+1}):\n\n{post}")],
        "topic": state["topic"],
        "generated_post": post,
        "critique": state["critique"],
        "num_attempts": state["num_attempts"] + 1,
    }

def critique_post(state: AgentState) -> dict[str, Any]:
    """Critique the generated post and provide score + improvements."""
    logger.info("Starting critique for generated post.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a viral LinkedIn strategist grading posts (1-10 scale).
        Critique structure, hook, value, engagement, length, hashtags, CTA - EVERYTHING.
        Be specific: "Hook too weak", "Add code example", "Too long", etc.
        End with exactly "IMPROVEMENTS: 1. ... 2. ... 3. ... 4. ..."."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", f"Post to critique:\n\n{state['generated_post']}\n\nTopic: {state['topic']}"),
    ])

    chain = prompt | LLM_MODEL

    try:
        logger.info("Invoking LLM chain to critique post.")
        response = chain.invoke({"messages": state["messages"]})
        critique = response.content
        logger.info("Critique complete.")
    except Exception as e:
        logger.error(f"Error during critique: {e}")
        raise

    logger.debug(f"Critique preview: {critique[:100]}...")

    return {
        "messages": [HumanMessage(content=f"Critique:\n\n{critique}")],
        "topic": state["topic"],
        "generated_post": state["generated_post"],
        "critique": critique,
        "num_attempts": state["num_attempts"],
    }
