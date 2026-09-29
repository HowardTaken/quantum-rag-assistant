"""Test doubles shared across the test suite. No network calls anywhere."""
from __future__ import annotations

from langchain_core.messages import AIMessage


class _FakeCollection:
    def __init__(self, n: int):
        self._n = n

    def count(self) -> int:
        return self._n


class FakeChroma:
    """Minimal stand-in for langchain_chroma.Chroma."""

    def __init__(self, docs=None, metadatas=None):
        self._docs = docs or []
        self._metadatas = metadatas if metadatas is not None else [d.metadata for d in self._docs]
        self._collection = _FakeCollection(len(self._metadatas))

    def similarity_search(self, query: str, k: int = 5):
        return self._docs[:k]

    def get(self, include=None):
        return {"metadatas": self._metadatas}


class FakeLLM:
    """Chat-model stand-in that replays a scripted sequence of responses."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.invocations: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)


def tool_call_message(tool_name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final_message(content: str) -> AIMessage:
    return AIMessage(content=content)
