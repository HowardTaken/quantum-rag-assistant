"""LangGraph agent: tool-using ReAct loop + a self-critique/re-retrieval step.

Graph shape:

    START -> agent -> [has tool calls?] -> tools -> agent (loop)
                    -> [no tool calls]  -> grade -> [grounded? / out of retries?] -> END
                                                  -> [ungrounded, retries left]    -> agent (loop)

`agent` is a normal ReAct node: the LLM decides whether to call `search_papers` /
`list_available_papers` or answer directly. Once it answers without a tool call, `grade`
asks a second LLM call to judge whether the answer is actually grounded in what came back
from the tools; if not, it injects a corrective instruction and loops the agent back to
search again, up to `max_agent_iterations` times.
"""
from __future__ import annotations

import logging
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_settings
from tools import load_vector_store, make_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an elite ECE / quantum-physics research assistant helping a student "
    "navigate a small library of semiconductor and photonics papers.\n\n"
    "You have two tools: `search_papers` to search the paper contents, and "
    "`list_available_papers` to see what documents exist.\n\n"
    "Rules:\n"
    "- Always use `search_papers` before answering any question that requires specific "
    "facts, numbers, or claims from the papers. Never answer such questions from general "
    "knowledge alone.\n"
    "- Always cite the source document name (and page, if given) for any claim drawn "
    "from search results.\n"
    "- If the search results don't contain the answer, say you do not know rather than "
    "guessing."
)

GRADE_PROMPT = (
    "You are grading whether an assistant's answer is properly grounded in the tool "
    "results retrieved earlier in this conversation. Respond with exactly one word: "
    "'grounded' if the answer's factual claims are supported by the retrieved passages "
    "and cited appropriately, or 'ungrounded' if it makes claims unsupported by the "
    "retrieved passages, ignores relevant retrieved information, or omits citations for "
    "a factual claim. If no factual claim was made (e.g. the assistant just listed "
    "available papers or declined to answer), respond 'grounded'."
)

_SOURCE_RE = re.compile(r"\[Source: (?P<source>[^,]+), page (?P<page>[^\]]+)\]")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    attempts: int


def route_after_agent(state: AgentState) -> Literal["tools", "grade"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "grade"


def route_after_grade(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "agent"
    return END


def build_agent(
    *,
    db=None,
    llm: BaseChatModel | None = None,
    grader_llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Assemble the compiled graph. Dependencies are injectable for testing."""
    settings = get_settings()
    db = db if db is not None else load_vector_store()
    tools_list = make_tools(db)

    llm = llm or ChatGoogleGenerativeAI(
        model=settings.llm_model, temperature=0, google_api_key=settings.google_api_key
    )
    llm_with_tools = llm.bind_tools(tools_list)

    grader_llm = grader_llm or ChatGoogleGenerativeAI(
        model=settings.llm_model, temperature=0, google_api_key=settings.google_api_key
    )

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def grade_node(state: AgentState) -> dict:
        attempts = state.get("attempts", 0)
        if attempts >= settings.max_agent_iterations:
            logger.info("grade: retry budget exhausted (attempts=%d)", attempts)
            return {}

        transcript = "\n\n".join(
            f"{m.type}: {m.content}" for m in state["messages"] if getattr(m, "content", None)
        )
        verdict = grader_llm.invoke(
            [SystemMessage(content=GRADE_PROMPT), HumanMessage(content=transcript)]
        )
        grounded = "ungrounded" not in str(verdict.content).lower()
        logger.info("grade attempt=%d verdict=%r grounded=%s", attempts, verdict.content, grounded)

        if grounded:
            return {}
        return {
            "attempts": attempts + 1,
            "messages": [
                HumanMessage(
                    content=(
                        "Your previous answer wasn't well grounded in the search results. "
                        "Search again with a more specific or differently worded query "
                        "before answering, or say you don't know if nothing relevant exists."
                    )
                )
            ],
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools_list))
    graph.add_node("grade", grade_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "grade": "grade"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("grade", route_after_grade, {"agent": "agent", END: END})

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def extract_sources(messages: list[BaseMessage]) -> list[dict[str, str]]:
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "search_papers":
            for match in _SOURCE_RE.finditer(str(m.content)):
                key = (match.group("source"), match.group("page"))
                seen[key] = {"source": key[0], "page": key[1]}
    return list(seen.values())


def ask(question: str, agent, thread_id: str = "default") -> str:
    return ask_with_sources(question, agent, thread_id=thread_id)["answer"]


def ask_with_sources(question: str, agent, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)], "attempts": 0}, config=config
    )
    messages = result["messages"]
    final = messages[-1]
    return {
        "answer": str(final.content),
        "sources": extract_sources(messages),
        "attempts": result.get("attempts", 0),
    }
