"""End-to-end tests of the compiled LangGraph agent, with fake LLMs/vector store
standing in for the network. Exercises the ReAct tool loop and the grade/retry loop."""
from langchain_core.documents import Document

from agent import ask_with_sources, build_agent
from tests.fakes import FakeChroma, FakeLLM, final_message, tool_call_message


def _build(agent_responses, grader_responses, docs=None, metadatas=None):
    db = FakeChroma(docs=docs or [], metadatas=metadatas)
    llm = FakeLLM(agent_responses)
    grader = FakeLLM(grader_responses)
    return build_agent(db=db, llm=llm, grader_llm=grader)


def test_agent_answers_via_search_tool_when_grounded():
    docs = [
        Document(
            page_content="the QCL tunes from 4.4 to 5.74 THz",
            metadata={"source": "qcl.pdf", "page": 2},
        )
    ]
    agent = _build(
        agent_responses=[
            tool_call_message("search_papers", {"query": "tuning range"}),
            final_message("It tunes from 4.4 to 5.74 THz [Source: qcl.pdf, page 2]."),
        ],
        grader_responses=[final_message("grounded")],
        docs=docs,
    )
    result = ask_with_sources("What is the tuning range?", agent, thread_id="t1")
    assert "4.4 to 5.74 THz" in result["answer"]
    assert result["sources"] == [{"source": "qcl.pdf", "page": "2"}]
    assert result["attempts"] == 0


def test_agent_uses_list_papers_tool_without_a_search():
    agent = _build(
        agent_responses=[
            tool_call_message("list_available_papers", {}),
            final_message("You have a.pdf and b.pdf."),
        ],
        grader_responses=[final_message("grounded")],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
    )
    result = ask_with_sources("What papers do you have?", agent, thread_id="t-list")
    assert result["answer"] == "You have a.pdf and b.pdf."
    assert result["sources"] == []  # list_available_papers hits don't count as cited sources


def test_agent_retries_once_when_grader_flags_ungrounded_then_succeeds():
    docs = [
        Document(
            page_content="the QCL tunes from 4.4 to 5.74 THz",
            metadata={"source": "qcl.pdf", "page": 2},
        )
    ]
    agent = _build(
        agent_responses=[
            tool_call_message("search_papers", {"query": "tuning"}),
            final_message("I think it's about 5 THz."),  # ungrounded: no citation
            tool_call_message("search_papers", {"query": "tuning range specifics"}),
            final_message("It tunes from 4.4 to 5.74 THz [Source: qcl.pdf, page 2]."),
        ],
        grader_responses=[
            final_message("ungrounded"),
            final_message("grounded"),
        ],
        docs=docs,
    )
    result = ask_with_sources("What is the tuning range?", agent, thread_id="t2")
    assert result["attempts"] == 1
    assert "4.4 to 5.74 THz" in result["answer"]


def test_agent_gives_up_after_max_retries_instead_of_looping_forever():
    agent = _build(
        agent_responses=[final_message("I don't know.") for _ in range(5)],
        grader_responses=[final_message("ungrounded") for _ in range(5)],
    )
    result = ask_with_sources("Unanswerable question", agent, thread_id="t3")
    assert result["attempts"] == 2  # default max_agent_iterations
    assert result["answer"] == "I don't know."


def test_conversation_memory_persists_across_turns_on_same_thread():
    agent = _build(
        agent_responses=[
            final_message("Hi, ask me about the papers."),
            final_message("You just said hi to me."),
        ],
        grader_responses=[final_message("grounded"), final_message("grounded")],
    )
    ask_with_sources("Hello", agent, thread_id="shared")
    result = ask_with_sources("What did I just say?", agent, thread_id="shared")
    assert result["answer"] == "You just said hi to me."


def test_separate_threads_do_not_share_memory():
    agent = _build(
        agent_responses=[
            final_message("Hi from thread A."),
            final_message("Hi from thread B."),
        ],
        grader_responses=[final_message("grounded"), final_message("grounded")],
    )
    a = ask_with_sources("Hello", agent, thread_id="a")
    b = ask_with_sources("Hello", agent, thread_id="b")
    assert a["answer"] == "Hi from thread A."
    assert b["answer"] == "Hi from thread B."
