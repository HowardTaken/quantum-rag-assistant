from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent import extract_sources, route_after_agent, route_after_grade


def test_route_after_agent_goes_to_tools_when_tool_calls_present():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_papers", "args": {"query": "x"}, "id": "1", "type": "tool_call"}],
            )
        ]
    }
    assert route_after_agent(state) == "tools"


def test_route_after_agent_goes_to_grade_when_no_tool_calls():
    state = {"messages": [AIMessage(content="final answer")]}
    assert route_after_agent(state) == "grade"


def test_route_after_grade_loops_back_to_agent_on_corrective_human_message():
    state = {"messages": [HumanMessage(content="try again")]}
    assert route_after_grade(state) == "agent"


def test_route_after_grade_ends_when_last_message_is_the_answer():
    state = {"messages": [AIMessage(content="final")]}
    assert route_after_grade(state) == END


def test_extract_sources_parses_tool_message_content():
    msg = ToolMessage(
        content="[Source: a.pdf, page 3]\nsome text\n\n---\n\n[Source: b.pdf, page 7]\nmore text",
        name="search_papers",
        tool_call_id="1",
    )
    sources = extract_sources([msg])
    assert {"source": "a.pdf", "page": "3"} in sources
    assert {"source": "b.pdf", "page": "7"} in sources


def test_extract_sources_deduplicates():
    msg = ToolMessage(
        content="[Source: a.pdf, page 3]\ntext\n\n---\n\n[Source: a.pdf, page 3]\ntext again",
        name="search_papers",
        tool_call_id="1",
    )
    assert extract_sources([msg]) == [{"source": "a.pdf", "page": "3"}]


def test_extract_sources_ignores_non_search_tool_messages():
    msg = ToolMessage(content="- a.pdf (2 chunks)", name="list_available_papers", tool_call_id="1")
    assert extract_sources([msg]) == []
