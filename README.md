# Netflix hybrid search

Search **8,807 Netflix movie and TV records** with BM25, vector retrieval,
reciprocal-rank fusion, and transformer reranking. Inspect four ranked lists,
call cached FastAPI endpoints, and use a local tool-calling agent whose outputs
must be exact quotations from retrieved records.

**Status:** working local prototype, not a completed relevance study. There is
no checked-in human judgment set or established winning method. Human evaluation
is omitted from the Streamlit interface; its local CLI and planned protocol are
preserved below in a collapsed section. No relevance improvement or
hallucination-free answering is claimed.

The former genre classifier and separate failure-cause/BM25-ablation track are
not part of this minimal prototype. The original upstream-failure thesis remains untested.

**[Open the hosted app](https://cheruto-netflix-relevance.streamlit.app/)**
(the deployment may still need updating).

<details>
<summary>Previous interface, before the hybrid rewrite</summary>

This screenshot is retained as project history, not as a preview of the current interface.

<img width="1582" height="771" alt="Earlier Netflix project interface" src="https://github.com/user-attachments/assets/de70980c-6728-430b-b2ab-18aa4874d79d" />

</details>

## Files you need to understand

| File | Its one job |
|---|---|
| `src\data.py` | Read, clean, and audit the CSV without overwriting it. |
| `src\index.py` | Build BM25, MiniLM embeddings, and a FAISS HNSW index; cache embeddings. |
| `src\search.py` | Implement BM25, vector, hybrid/RRF, and hybrid-plus-rerank search. |
| `src\agent.py` | Let a local model call search, then validate its quoted evidence. |
| `src\study.py` | Freeze queries, store human grades, blind repeat labels, calculate agreement. |
| `src\metrics.py` | Calculate precision, recall, MRR, and nDCG from their definitions. |
| `src\evaluate.py` | One research CLI: audit, split, pool, label, sample, consistency, evaluate, freeze. |
| `app\api.py` | Serve `/search`, `/rerank`, and `/ask` with a bounded search cache. |
| `streamlit_app.py` | The only search interface: four tables and an optional agent. |
| `bench\latency.py` | Measure warm-cache HTTP p50/p95 latency and concurrent throughput. |

`tests\` protects these behaviours; `Dockerfile` serves the API; the GitHub
Actions workflow runs the tests. `src\__init__.py` and `app\__init__.py` mark
packages, not extra features. README is the practical guide; [DATA.md](DATA.md)
is the cleaning audit. The raw `netflix_titles.csv` stays at the root so there
is only one source copy.

This cleanup reduces the project, but **does not meet the original 500-line
total target**. Input checks, label provenance, split safeguards, and grounding
checks are retained rather than hidden behind wrappers or removed to meet a count.

## Run the one local page

Use Python 3.12 or newer, from this checkout:

```powershell
python -m pip install -r requirements.txt
python -m src.evaluate audit
python -m streamlit run streamlit_app.py
```

Enter a query, choose the number of results, and click **Compare four methods**.
Models load on submission, not when opening the page. The first run downloads
models and computes embeddings; later builds reuse the gitignored `.hybrid_cache`.
The in-memory index is shared within a process; results stay in each user's session.

For practical comparisons, try `Stranger Things`, `movies about a crew stranded
in space`, or `documentaries about chefs`. These are **demo examples, not a
research query inventory**. Exact title words can occur in multiple titles;
semantic matches can lack the query's literal wording. Different rankings do
not, by themselves, establish which result deserves a higher relevance grade.

BM25 uses casefolded Unicode word tokens across title, description, genres,
cast, director, and country. Vectors use normalized MiniLM embeddings and
approximate HNSW neighbours. Hybrid adds reciprocal ranks, not raw scores.
Reranking scores the fused shortlist with a cross-encoder; it cannot recover a
record outside that shortlist. The word-overlap column is diagnostic, not a label.
Settings live in named constants in `src\index.py` and `src\search.py`; the
research protocol records them. See DATA.md for the representation decisions.

### Local agent

Install/start Ollama separately and use a tool-capable model:

```powershell
ollama pull qwen3:0.6b
$env:NETFLIX_AGENT_MODEL = "qwen3:0.6b"
python -m streamlit run streamlit_app.py
```

In **Local grounded agent**, ask a catalogue question. The pretrained model
first calls `search_films`; Python performs hybrid retrieval and supplies the
retrieved metadata. A second, structured response contains citations. Python
checks that each ID was retrieved and each quotation equals the complete cited
field, then adds the real title. Empty citations mean explicit abstention;
unapproved tools, invented citations, invalid outputs, and limits raise errors.

This is extractive evidence, not a newly trained Netflix model or an unrestricted
chatbot. Exact quotation proves provenance, **not relevance, source truth, or
clinical safety**. Without `NETFLIX_AGENT_MODEL`, the agent is disabled.

<details>
<summary>Planned human relevance evaluation (not completed)</summary>

## One human research workflow

Write your **own queries and a 0/1/2 rubric before labelling**. The planned
40-query study is not supplied by the code. Create `judgment_data\queries.csv`
with exactly `query_id,query`; each row is one query you authored.

```powershell
New-Item -ItemType Directory -Path judgment_data -Force
# Write queries.csv yourself before running these commands.
python -m src.evaluate split --queries judgment_data\queries.csv --output judgment_data\split.json
python -m src.evaluate pool --split judgment_data\split.json --output judgment_data\dev_pool.json -k 10 --depth 50
python -m src.evaluate label --split judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv
```

The default split is 80/20 with seed 42, frozen **before tuning**. Development
pooling saves each method's top-k ranking and shuffles their deduplicated union.
You grade each query-record pair once, even when several methods retrieve it.
The labeller shows the query and full cleaned record but hides method, rank,
score, and previous grades. Only explicit `0`, `1`, or `2` is accepted; blank
input has no default. Each submitted grade receives a UTC timestamp and is saved
immediately. Quit and rerun to resume without replacing earlier judgments.

Frozen splits, pools, and repeat queues cannot be overwritten. SHA-256 bindings
reject reuse with changed source data, splits, or pools. To include known records
outside the retrieved lists, provide an **ungraded** `query_id,show_id` CSV with
`pool --additional-pairs`; candidates must belong to the chosen query partition.
Never use the demo to tune or preview held-out queries before freezing settings.

### Blind repeat and self-consistency

```powershell
python -m src.evaluate sample --split judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv --query-count 10 --output judgment_data\repeat.json
# Return a day later; the program does not enforce the delay.
python -m src.evaluate label --split judgment_data\split.json --pool judgment_data\repeat.json --labels judgment_data\judgments.csv
python -m src.evaluate consistency --split judgment_data\split.json --pool judgment_data\repeat.json --labels judgment_data\judgments.csv
```

This repeats all pooled pairs for the sampled queries; alternatively `--size`
samples individual pairs. The saved queue hides earlier grades and records a
separate pass. Exact agreement and quadratic-weighted Cohen's kappa compare
paired grades, weighting each pair equally. Undefined kappa is reported as such.
The tool cannot prevent remembering answers or reading your original CSV.

### Evaluate development, then freeze one method

Choose the binary threshold explicitly: either grades 1-2 count as relevant,
or only grade 2 does. This affects precision, recall, and MRR; nDCG always uses
the original linear graded gain. Do not choose it using held-out results.

```powershell
$grade = Read-Host "Binary relevance threshold (1 or 2)"
python -m src.evaluate evaluate --split judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv --relevant-grade $grade --output judgment_data\dev_results.csv
$winner = Read-Host "Method selected using development results (bm25, vector, hybrid, or rerank)"
python -m src.evaluate freeze --split judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv --relevant-grade $grade --mode $winner --output judgment_data\frozen.json
python -m src.evaluate pool --split judgment_data\split.json --partition heldout --frozen judgment_data\frozen.json --output judgment_data\heldout_pool.json
python -m src.evaluate label --split judgment_data\split.json --pool judgment_data\heldout_pool.json --labels judgment_data\judgments.csv
python -m src.evaluate evaluate --split judgment_data\split.json --pool judgment_data\heldout_pool.json --labels judgment_data\judgments.csv --frozen judgment_data\frozen.json --relevant-grade $grade --output judgment_data\heldout_results.csv
```

Evaluation refuses missing human labels, never imputes zeros, and never silently
excludes undefined-metric queries. CSVs contain per-query results; stdout contains
equal-query macro averages, including MRR@k. Precision divides by k even for short
lists. Recall is relative to all judged-relevant records, not the full catalogue.
nDCG uses linear gain and base-2 log discounts. The held-out run uses only the
frozen method, cutoff, depth, protocol, and relevance threshold.

`judgment_data\` is gitignored, **not backed up**. Preserve your actual queries,
rubric, judgments, repeated passes, frozen settings, and measured outputs yourself.

### Study results (pending)

| Result | Value |
|---|---|
| Human queries and judged pairs | [RESULT: query count and judgment-list size] |
| Repeat delay and exact agreement | [RESULT: repeat delay and exact self-agreement] |
| Quadratic-weighted Cohen's kappa | [RESULT: self-consistency kappa] |
| Development metrics by method | [RESULT: precision, recall, MRR and nDCG dev] |
| Frozen held-out metrics | [RESULT: precision, recall, MRR and nDCG heldout] |

</details>

## Limitations

Limitations include a **single annotator**, judgment-list size and incomplete
coverage, and **overfitting to development queries**. If held-out judgments cover
only the frozen method's own top-k, recall@k is trivially one when any relevant
item exists; independent human-added candidates are needed for a meaningful
coverage comparison. It is still judged-list recall, not corpus recall. Query
paraphrases can leak across partitions despite duplicate-text checks. Short,
missing, historical, multilingual metadata and model truncation limit retrieval.
Approximate neighbours and the reranking shortlist can miss records. Quotation
grounding does not guarantee good recommendations or factual correctness.

## API, benchmark, and checks

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
# In another terminal, write demo queries one per line, then:
python -m bench.latency --queries demo_queries.txt --mode hybrid --requests 100 --concurrency 4
python -m pip install -r requirements-test.txt
python -m pytest -q
```

`/search` selects a method; `/rerank` forces reranking; `/ask` uses the local
agent. Search results use an in-memory LRU cache of 256 entries. The benchmark
warms repeated queries first: it measures HTTP serving, **not cold inference or
relevance**. No authentication/rate limiting is implemented; keep this API local.
Reportable benchmark summary: **[RESULT: p50, p95 and throughput with machine/load configuration]**.
Metric tests use hand-worked synthetic examples, with independent library
cross-checks only in tests. Fixture grades are not Netflix research judgments.

Docker build instructions: `docker build -t netflix-search .`, then
`docker run --rm -p 127.0.0.1:8000:8000 netflix-search`. The recipe and CI
workflow are present. The test workflow has passed on GitHub Actions; a local
Docker build remains unverified.

## Streamlit Community Cloud

Deploy the merged version from `main`:

| Deployment field | Value |
|---|---|
| Repository | `SBusienei/Netflix-Relevance-Project` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Advanced settings: Python version | `3.12` |

Sign in at <https://share.streamlit.io/>. If your existing app already uses
the feature branch `cherutosharleen-retrieval-failure-analysis` and this entry
point, it can still run while that branch exists: open its three-dot menu and
select **Reboot**. The earlier clone failure occurred while that branch was
missing. You can also use **Manage app > three-dot menu > Reboot app** from the
deployed app.

For a new deployment, select **Create app**, choose the GitHub deployment
option, enter the fields above, and select **Deploy**. Community Cloud reads
`requirements.txt` from this repository. Do not delete an existing app just
to retry cloning the restored branch.
Changing an existing app's repository, branch, or entry point requires
redeployment; those coordinates are not editable in the normal app settings.

Publishing code to GitHub does not verify a hosted deployment. Transformer
downloads and first-run indexing can take several minutes, and memory/startup
costs may exceed free-host limits. The local Ollama agent is unavailable on
Community Cloud by default; leave it disabled there. Hosted sessions are not
durable research storage.
