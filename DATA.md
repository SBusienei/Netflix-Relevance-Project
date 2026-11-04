# Data cleaning and shared index decisions

## Source and scope

The checked-in `netflix_titles.csv` is the Netflix Movies and TV Shows dataset
linked from the existing README:
<https://www.kaggle.com/datasets/shivamb/netflix-shows>.
The retrieval corpus keeps **all 8,807 records: 6,131 movies and 2,676 TV shows**.
No genres, languages, countries, years, or record types are filtered out.

The source CSV is never overwritten. `src.data.load_records()` returns cleaned
records and an audit report. `python -m src.evaluate audit` prints that report,
the source path and SHA-256 as JSON. These counts
describe the actual checked-in data, not retrieval effectiveness results.

## Every automatic cleaning decision

1. Read with Python's CSV parser, UTF-8 with optional BOM, preserving quoted
   commas and embedded newlines. Require the existing 12-column schema and its
   order. Reject malformed records, an empty file/corpus, missing identifiers,
   or conflicting records with the same identifier rather than silently skipping.
   Missing cells are represented by empty strings, not `"nan"` or `"Unknown"`.
   Literal strings such as `"NA"` are not interpreted as missing values.
2. Normalize every cell to Unicode NFC, trim surrounding whitespace, and replace
   runs of Unicode whitespace (including embedded newlines) with one space.
   Preserve case, accents, punctuation, and non-Latin text in stored records.
   Do not transliterate, remove HTML, spell-correct, or infer missing content.
   Normalized-cell counts report changes before duplicate removal or metadata
   repair. On the source, whitespace changes affect 9 title, 7 cast, 88 date-added,
   and 16 description cells; NFC also changes one cell. The audit reports the
   combined count per field, without double-counting a cell changed by both.
3. If `duration` is empty and `rating` consists exactly of digits followed by
   `" min"` after normalization, move that value to `duration` and clear `rating`.
   This narrow repair applies to `s5542` (`74 min`), `s5795` (`84 min`), and
   `s5814` (`66 min`). Their actual maturity ratings remain unknown. No other
   metadata repairs, date parsing, numeric conversion, or duration conversion
   are performed. These fields are not searched by default.
4. Remove only identical complete records after normalization and the narrow
   duration repair; retain the first occurrence in CSV order. A conflicting
   duplicate `show_id` is an error. Never deduplicate by title or content alone:
   remakes and separately identified records must remain distinct. The checked-in
   source has no duplicate rows, duplicate IDs, or repeated raw titles; no rows
   are removed.
5. Keep records with missing searchable metadata; missing fields contribute no
   tokens. Do not invent directors, cast, countries, descriptions, or relevance
   labels. Preserve stable `show_id` values and input order for later judgments.

| Field | Missing before cleaning | Missing after cleaning |
|---|---:|---:|
| show_id | 0 | 0 |
| type | 0 | 0 |
| title | 0 | 0 |
| director | 2,634 | 2,634 |
| cast | 825 | 825 |
| country | 831 | 831 |
| date_added | 10 | 10 |
| release_year | 0 | 0 |
| rating | 4 | 7 |
| duration | 3 | 0 |
| listed_in | 0 | 0 |
| description | 0 | 0 |

## One shared representation

`src.index` uses the same cleaned records for all four retrieval methods.
The indexed fields are `title`, `description`, `listed_in`, `cast`, `director`,
and `country`. BM25 concatenates their values; embedding and reranking text
adds field-name prefixes to nonempty values. Missing fields contribute no
content. Genres and names remain text, not parsed entities.

BM25 uses `rank-bm25==0.2.2`, with `k1=1.5`, `b=0.75`, and `epsilon=0.25`.
Documents and queries use `re.findall(r"\w+", text.casefold())`: Unicode word
runs, including underscores. Punctuation separates tokens, so `sci-fi` becomes
`sci`, `fi`, and `O'Brien` becomes `o`, `brien`. Accents and non-Latin characters
remain. No stemming, stop-word removal, field weighting, or language-aware
segmentation is applied. The old whitespace/stemming/field-weight ablation
interface has been retired, not silently retained under another name.

BM25 IDF is `log(N-df+0.5) - log(df+0.5)`; negative values are replaced with
epsilon times mean raw IDF. The search interface returns **positive-score
lexical hits only**; no positive matches produce an empty BM25 list, not
zero-score records presented as matches. This is the existing hybrid behaviour,
not the retired baseline's all-record tie ordering.

Embeddings use pinned `all-MiniLM-L6-v2`, 384 dimensions, normalized float32,
with a 256-token limit. FAISS HNSW uses inner product (cosine for unit vectors),
M=32, efConstruction=80, and efSearch=64. Reciprocal-rank fusion uses constant
60. The pinned cross-encoder reranks the fused shortlist with a 512-token limit.
All methods break considered-score ties using CSV order.

Named settings and model revisions live in `src.index`/`src.search`; pools and
frozen study settings record the protocol. Changes require new development
pools and invalidate incompatible held-out settings. Embedding cache keys include
source bytes, fields, embedding revision, token limit, and document-format version.
If editing document formatting or cleaning logic, bump that format version and
the study protocol version; a source-byte fingerprint alone cannot detect code edits.
Cached vectors are checked for shape, dtype, finite values, and unit norms.

## Reproduction

From the repository root, using the same Python environment:

```powershell
python -m pip install -r requirements.txt
python -m src.evaluate audit
python -m streamlit run streamlit_app.py
python -m pip install -r requirements-test.txt
python -m pytest tests\test_data.py -q
```

## Research boundary

The thesis that failures arise primarily upstream of scoring is **untested**.
The simplified project no longer includes the separate failure-cause analysis
track. No relevance labels or effectiveness claims follow from this data audit.
Evaluation must stop if required human labels are missing. The [single research
workflow](README.md#one-human-research-workflow) freezes a development/held-out
query split before tuning.
