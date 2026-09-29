from langchain_core.messages import HumanMessage

from agent import ask_with_sources, build_agent
from checkpointer import build_sqlite_checkpointer
from tests.fakes import FakeChroma, FakeLLM, final_message


def _ask(checkpointer, question: str, thread_id: str) -> None:
    agent = build_agent(
        db=FakeChroma(),
        llm=FakeLLM([final_message(f"Reply to: {question}")]),
        grader_llm=FakeLLM([final_message("grounded")]),
        checkpointer=checkpointer,
    )
    ask_with_sources(question, agent, thread_id=thread_id)


def test_conversation_survives_a_fresh_agent_sharing_the_same_checkpointer():
    # A new agent object (e.g. after a process restart) can still see prior turns as long
    # as it's handed the same checkpointer -- exactly what a shared/on-disk SQLite
    # connection gives you that a fresh in-process MemorySaver wouldn't.
    checkpointer = build_sqlite_checkpointer(":memory:")
    _ask(checkpointer, "Hello", thread_id="persistent-thread")

    second_agent = build_agent(
        db=FakeChroma(),
        llm=FakeLLM([final_message("unused")]),
        grader_llm=FakeLLM([final_message("grounded")]),
        checkpointer=checkpointer,
    )
    state = second_agent.get_state({"configurable": {"thread_id": "persistent-thread"}})
    human_messages = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert any(m.content == "Hello" for m in human_messages)


def test_conversation_survives_closing_and_reopening_the_sqlite_file(tmp_path):
    # This is the actual scenario the feature exists for: the process dies and a new one
    # starts, reopening the same file from disk -- as opposed to the in-memory case above,
    # which only proves a shared connection object works.
    db_path = str(tmp_path / "checkpoints.sqlite")

    checkpointer = build_sqlite_checkpointer(db_path)
    _ask(checkpointer, "What is a QCL?", thread_id="thread-1")
    checkpointer.conn.close()

    reopened = build_sqlite_checkpointer(db_path)
    agent = build_agent(
        db=FakeChroma(),
        llm=FakeLLM([final_message("unused")]),
        grader_llm=FakeLLM([final_message("grounded")]),
        checkpointer=reopened,
    )
    state = agent.get_state({"configurable": {"thread_id": "thread-1"}})
    human_messages = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert any(m.content == "What is a QCL?" for m in human_messages)


def test_different_threads_stay_isolated_across_a_reopen(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")

    checkpointer = build_sqlite_checkpointer(db_path)
    _ask(checkpointer, "thread A message", thread_id="a")
    checkpointer.conn.close()

    reopened = build_sqlite_checkpointer(db_path)
    agent = build_agent(
        db=FakeChroma(),
        llm=FakeLLM([final_message("unused")]),
        grader_llm=FakeLLM([final_message("grounded")]),
        checkpointer=reopened,
    )
    state = agent.get_state({"configurable": {"thread_id": "b"}})
    assert not state.values
