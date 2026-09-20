# litmine — automated teacher-ranking literature mining

Implements `docs/PIPELINE.md`: finds published **agentic SFT experiments in
which the same student is fine-tuned separately on trajectories from several
teachers**, extracts the per-teacher downstream scores with evidence, verifies
the public artifacts, classifies each experiment, and exports ground-truth
teacher rankings for proxy evaluation.

```
candidate discovery  ->  retrieval  ->  LLM prescreen (title/abstract)
  ->  LLM full-text screening  ->  LLM experiment extraction (1 record per student x experiment)
  ->  multi-model reconciliation + adjudication  ->  artifact discovery & verification
  ->  deterministic classification + ranking  ->  CSV / JSONL / review queue / summary
```

## Setup

```bash
cd litmine
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# keys are read from the environment (or ../.env, which is a symlink to the secrets file)
export OPENAI_API_KEY=... DEEPSEEK_API_KEY=...        # HF_TOKEN / GITHUB_TOKEN / S2_API_KEY optional
```

## Usage

```bash
# one paper end-to-end (seed papers skip the abstract prescreen)
LLM_BACKEND=both .venv/bin/python -m litmine paper 2606.03461

# full discovery + pipeline (arXiv/OpenAlex/HF/GitHub queries + citation expansion from seeds)
LLM_BACKEND=both .venv/bin/python -m litmine run --seeds 2606.03461 --citation-depth 1 --max-candidates 200

# discovery only / re-export without any LLM call
.venv/bin/python -m litmine discover --seeds 2606.03461
# discover + cheap prescreen, then stop and print the estimated cost of full-text screening
.venv/bin/python -m litmine run --prescreen-only
.venv/bin/python -m litmine export
```

Outputs land in `work/out/`:

| file | content |
|---|---|
| `ground_truths.json` | **the compact ground-truth list**: one entry per student x trajectory distribution with student (+HF id), task dataset HF links, per-teacher trajectory HF links (file-level inside combined datasets), benchmark, published scores, tie-preserving ranking, source table, status and caveats; usable (gold/silver) entries first |
| `external_teacher_rankings.csv` | one row per student x experiment (columns from PIPELINE.md §10; `teacher_scores_json` is canonical, `teacher_rank_k` = teachers at competition rank *k*, ties joined with ` = `) |
| `external_teacher_rankings.jsonl` | full records: experiment with evidence, both backends' extractions, disagreements, adjudication, artifact checks, provenance |
| `review_queue.csv` | records needing a human decision, with the evidence needed to resolve them |
| `external_teacher_rankings_summary.md` | counts and the list of immediately usable datasets |

Intermediate state: `work/state/` (candidates, prescreen decisions, one JSON per
paper, `records.jsonl`, `stats.json`). Cache: `work/cache/<stage>/<sha256>.json`.

## Configuration (env)

| variable | default | meaning |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` \| `deepseek` \| `both` |
| `LITMINE_OPENAI_MODEL` / `LITMINE_DEEPSEEK_MODEL` | `gpt-5.5` / `deepseek-v4-flash` | model ids (never hard-coded elsewhere) |
| `LITMINE_OPENAI_REASONING` | `medium` | reasoning effort for gpt-5.x |
| `LITMINE_DEEPSEEK_TEMPERATURE` | `0.0` | DeepSeek temperature (thinking disabled) |
| `LITMINE_ADJUDICATOR` | `openai` | backend used to adjudicate disagreements when `both` |
| `LITMINE_WORK_DIR` | `litmine/work` | cache + state + outputs |
| `LITMINE_MAX_DOC_CHARS` | `260000` | paper text passed to the LLM (truncation is recorded) |
| `LITMINE_MAX_CANDIDATES`, `LITMINE_PER_QUERY`, `LITMINE_CITATION_DEPTH` | 5000 / 100 / 2 | discovery breadth |
| `LITMINE_BROAD_RESULTS`, `LITMINE_MIN_YEAR` | 1500 / 2023 | paging depth and year floor of the broad arXiv topic queries |
| `LITMINE_PRESCREEN_BACKEND` | deepseek if `DEEPSEEK_API_KEY` is set | backend for the title/abstract prescreen (~10x cheaper than gpt-5.5) |
| `LITMINE_KEYWORD_GATE` | 1 | free regex gate (agentic AND training-data vocabulary) before any prescreen LLM call |
| `LITMINE_PRESCREEN_MODEL`, `LITMINE_SCREEN_MODEL` | – | cheaper models for the gating stages (e.g. `gpt-5.4-mini`) |
| `S2_API_KEY`, `GITHUB_TOKEN`, `HF_TOKEN`, `OPENALEX_MAILTO` | – | optional; raise rate limits / access gated repos |

API keys are read only from the environment and never written to cache, state or logs.

## Design notes

**Discovery is broad, precision comes later.** The first sweep used narrow
phrase queries ("teacher trajectories" AND "SFT" AND "agent") and found only
237 candidates: the target ranking is usually an ablation table inside a paper
whose abstract never says "teacher" (e.g. OpenThoughts-Agent). Discovery now
combines (a) six topic-level arXiv boolean queries paged newest-first
(`BROAD_ARXIV_QUERIES`), (b) ~30 known agentic-SFT-data papers listed by title
in `SEED_TITLES` (resolved on arXiv, then expanded through references and
citations to depth 2 — hop 2 only from candidates that pass the keyword gate),
(c) HF dataset search plus every arXiv-tagged dataset of known trajectory
orgs (`DEFAULT_HF_AUTHORS`), (d) the original phrase queries. The regex
`keyword_gate` then drops candidates with no agentic or no training-data
vocabulary at zero cost, and the LLM prescreen (DeepSeek by default) gates the
rest before the expensive full-text screening.

* **Grounded LLM stages** (`prompts.py`): every decision is `yes|no|unclear` with a
  quoted snippet, source location, confidence and explicit/inferred marker.
  `schemas.py` enforces the shape and downgrades any `yes`/`no` without evidence
  to `unclear`. Nothing is ever filled from model memory.
* **Multi-model verification** (`multimodel.py`): with `LLM_BACKEND=both`, OpenAI
  and DeepSeek extract independently; experiments are aligned, fields diffed
  (free-text descriptions count as a dispute only when their numbers differ),
  every disagreement is stored verbatim and sent to an adjudicator LLM together
  with the source text. Fields the adjudicator leaves `unclear` are blanked and
  force `needs_review`; resolved ones keep their status but stay in the review queue.
* **Artifact verification** (`artifacts.py`): HF/GitHub/project-page identifiers
  named in the paper are fetched via public APIs; links found on project pages
  and READMEs are followed; each dataset is judged from its own metadata/card
  for role (tasks vs trajectories), teacher identity, separability and task ids.
  A paper cannot be `gold` on a release claim alone.
* **Deterministic classification** (`classify.py`): `reject` if any core criterion
  is `no`; `needs_review` if a core criterion is `unclear`, scores are missing for
  a teacher, or claimed artifacts cannot be found; `results_only` if
  tasks/trajectories are not public; `silver` for 2 teachers / unmatched tasks /
  unverified artifacts; `gold` only with everything `yes` and verified artifacts.
  Rankings use competition ranking on the exact published numbers (ties kept).
* **Caching** (`cache.py`, `llm.py`): each LLM result is keyed by stage, prompt
  version, backend, model, temperature/reasoning and the sha256 of the input.
  Changing one prompt version reruns only that stage; exports never call an LLM;
  cached raw outputs are re-validated so schema fixes apply without new calls.
  HTTP responses, documents, search results and artifact lookups are cached too.

## Tests

```bash
.venv/bin/python -m pytest -q                     # unit tests (fake LLM + fake HTTP), ~5 s
LITMINE_INTEGRATION=1 LLM_BACKEND=both .venv/bin/python -m pytest -q tests/test_integration_terminal_lego.py
```

The integration test runs the real pipeline on arXiv:2606.03461 (Terminal-Lego)
and checks that the *structure* is recovered — students, ≥2 teachers, numeric
per-teacher scores that literally occur in the source, rankings derived from
those scores, provenance — without asserting any published value.
