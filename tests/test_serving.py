"""Check bounded caching, endpoint errors, and visible comparison without real-model downloads."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app import api
from src.index import Index


def test_api_cache_and_three_endpoints(small_index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "build_index", lambda: small_index)
    api.cached.cache_clear()
    with TestClient(api.app) as client:
        payload = {"query": "orbit", "k": 2, "depth": 3}
        first = client.post("/search", json=payload)
        assert first.status_code == 200 and first.json()[0]["show_id"] == "s1"
        assert client.post("/search", json=payload).json() == first.json()
        assert api.cached.cache_info().hits == 1
        assert api.cached.cache_info().maxsize == 256
        assert client.post("/rerank", json=payload).json()[0]["show_id"] == "s2"
        monkeypatch.delenv("NETFLIX_AGENT_MODEL", raising=False)
        response = client.post("/ask", json=payload)
        assert response.status_code == 503 and "Agent disabled" in response.json()["detail"]
        for changes in ({"query": "   "}, {"query": "!!!"}, {"k": 0}, {"mode": "invalid"},
                        {"k": 4}, {"depth": 101}):
            assert client.post("/search", json=payload | changes).status_code == 422
    api.cached.cache_clear()


def test_single_search_page_and_comparison(small_index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.index

    monkeypatch.delenv("NETFLIX_AGENT_MODEL", raising=False)
    monkeypatch.setattr(src.index, "build_index", lambda: small_index)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=120).run()
    assert not app.exception and not app.dataframe
    assert app.title[0].value == "Netflix hybrid search"
    assert [section.label for section in app.expander] == ["Local grounded agent"]
    assert not app.code
    assert app.button[1].disabled
    app.text_input(key="hybrid_query").set_value("orbit")
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.dataframe) == 4
    assert app.dataframe[0].value.iloc[0]["show_id"] == "s1"
    assert app.dataframe[3].value.iloc[0]["show_id"] == "s2"
    app.run()
    assert len(app.dataframe) == 4
    app.text_input(key="hybrid_query").set_value("!!!")
    app.button[0].click().run()
    assert not app.exception and app.error and not app.dataframe


def test_agent_form_submits_question_and_rejects_empty_input_before_loading(
    small_index: Index, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent
    import src.index

    monkeypatch.setenv("NETFLIX_AGENT_MODEL", "test-model")
    questions: list[str] = []
    loads: list[bool] = []

    def load() -> Index:
        loads.append(True)
        return small_index

    def answer(index: Index, question: str) -> dict:
        questions.append(question)
        return {"citations": [], "abstained": True, "tool_queries": [question]}

    monkeypatch.setattr(src.index, "build_index", load)
    monkeypatch.setattr(src.agent, "ask", answer)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=120).run()
    assert not loads
    app.text_input(key="hybrid_question").set_value("Find orbit")
    app.button[1].click().run()
    assert not app.exception and not app.error and app.get("json")
    assert questions == ["Find orbit"] and loads == [True]
    app.text_input(key="hybrid_question").set_value("")
    app.button[1].click().run()
    assert not app.exception and app.error and not app.get("json")
    assert questions == ["Find orbit"] and loads == [True]
