"""Ensure tool use and exact retrieved evidence are required, not merely prompted."""

import json

import pytest

from src.agent import ask, local_chat
from src.index import Index


def scripted(citations: list[dict]):
    replies = iter([
        {"role": "assistant", "tool_calls": [{"function": {
            "name": "search_films", "arguments": {"query": "orbit"},
        }}]},
        {"role": "assistant", "content": json.dumps({"citations": citations})},
    ])
    return lambda messages, tools: next(replies)


def test_exact_citation_and_explicit_abstention(small_index: Index) -> None:
    citation = {"show_id": "s1", "field": "description", "quote": "A stranded space crew."}
    answer = ask(small_index, "Find space films", scripted([citation]))
    assert answer == {"citations": [citation | {"title": "Orbit"}], "abstained": False, "tool_queries": ["orbit"]}
    assert ask(small_index, "Find space films", scripted([]))["abstained"] is True


@pytest.mark.parametrize("citation", [
    {"show_id": "s1", "field": "description", "quote": "A cure for disease."},
    {"show_id": "unknown", "field": "description", "quote": "A stranded space crew."},
    {"show_id": "s1", "field": "description", "quote": "stranded space"},
])
def test_hallucinated_id_or_text_is_rejected(small_index: Index, citation: dict) -> None:
    with pytest.raises(ValueError, match="Grounding failed"):
        ask(small_index, "Find space films", scripted([citation]))


def test_no_tool_call_and_turn_limit_are_errors(small_index: Index) -> None:
    with pytest.raises(ValueError, match="did not call"):
        ask(small_index, "Find films", lambda messages, tools: {"role": "assistant", "content": '{"citations":[]}'})
    repeated = {"role": "assistant", "tool_calls": [{"function": {
        "name": "search_films", "arguments": {"query": "orbit"},
    }}]}
    with pytest.raises(ValueError, match="four-turn"):
        ask(small_index, "Find films", lambda messages, tools: repeated)


def test_disabled_local_model_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NETFLIX_AGENT_MODEL", raising=False)
    with pytest.raises(ValueError, match="Agent disabled"):
        local_chat([], [])


def test_existing_but_unretrieved_record_is_not_evidence(small_index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent

    monkeypatch.setattr(src.agent, "search", lambda *args: [{"show_id": "s1", "score": 1.0}])
    citation = {"show_id": "s2", "field": "description", "quote": "A cooking competition."}
    with pytest.raises(ValueError, match="Grounding failed"):
        ask(small_index, "Find space films", scripted([citation]))
