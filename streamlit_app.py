"""Show one inspectable search comparison and an optional locally grounded agent."""

import os
import httpx
import streamlit as st

st.set_page_config(page_title="Netflix hybrid search", page_icon=":material/search:", layout="wide")
st.title("Netflix hybrid search", anchor=False)
st.caption("One catalogue, four retrieval methods. Demo queries are not research judgments; scores use different scales.")
with st.form("hybrid_search"):
    query = st.text_input("Query", placeholder="Try: movies about a crew stranded in space", key="hybrid_query")
    k = st.slider("Results per method", 1, 20, 5)
    submitted = st.form_submit_button("Compare four methods")
if submitted:
    st.session_state.pop("hybrid_results", None)
    try:
        from src.index import FIELDS, build_index, tokens
        from src.search import MODES, search, validate_query

        validate_query(query)
        with st.spinner("Loading models and indexing the catalogue; the first run can take several minutes..."):
            index = build_index()
            by_id = {row["show_id"]: row for row in index.records}
            tables = {}
            for mode in MODES:
                results = search(index, query, mode, k)
                tables[mode] = [{"rank": rank, **by_id[hit["show_id"]], "score": hit["score"],
                                "literal_query_words": ", ".join(sorted(set(tokens(query)) &
                                    set(tokens(" ".join(by_id[hit["show_id"]][name] for name in FIELDS)))))}
                               for rank, hit in enumerate(results, 1)]
            st.session_state["hybrid_results"] = {"query": query, "tables": tables}
    except ImportError as error:
        st.error(f"Search dependencies are missing: {error}. Install requirements.txt.")
    except (OSError, ValueError, httpx.HTTPError) as error:
        st.error(f"Search failed: {error}")
output = st.session_state.get("hybrid_results")
if output:
    st.caption(f"Results for: {output['query']}")
for mode, rows in (output["tables"] if output else {}).items():
    st.subheader(mode)
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.info("No positive lexical matches. Other stages can still retrieve candidates.")
st.info("BM25 matches words; vectors retrieve nearby meanings; hybrid fuses ranks; rerank reorders its shortlist. "
        "Literal word overlap explains matching words, not relevance. Only your labels can tell us which is better.")
with st.expander("Local grounded agent"):
    st.caption("Requires Ollama on this server and NETFLIX_AGENT_MODEL. Disabled on Community Cloud by default. "
               "Answers are exact catalogue quotations, not free-form recommendations or clinical advice.")
    with st.form("local_agent"):
        question = st.text_input("Question for the agent", key="hybrid_question")
        asked = st.form_submit_button("Ask local agent", disabled=not os.environ.get("NETFLIX_AGENT_MODEL"))
    if asked:
        try:
            from src.agent import ask
            from src.index import build_index
            from src.search import validate_query

            validate_query(question)
            with st.spinner("Calling the local model and validating its citations..."):
                answer = ask(build_index(), question)
            st.json(answer)
        except (OSError, ValueError, ImportError, httpx.HTTPError) as error:
            st.error(f"Agent failed: {error}")
