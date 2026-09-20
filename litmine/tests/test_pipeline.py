import csv
import json

import pytest

from conftest import ARXIV_ID, SYNTH_HTML, FakeFetcher, atom_feed, default_routes, make_responder
from litmine.cache import Cache
from litmine.export import CSV_COLUMNS
from litmine.llm import FakeLLM
from litmine.pipeline import Pipeline


def _pipe(work, fetcher, clients, adjudicator=None):
    return Pipeline(work, clients=clients, adjudicator=adjudicator, fetcher=fetcher)


def test_single_backend_end_to_end(work, fetcher, fake_openai):
    p = _pipe(work, fetcher, {"openai": fake_openai})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    assert len(recs) == 2                                   # two students -> two records
    by_student = {r["experiment"]["student_model"]["value"]: r for r in recs}
    assert set(by_student) == {"Stu-1B", "Stu-3B"}
    # paper text reached the LLM: extraction prompt contains the table row
    ext_prompts = [u for st, s, u in fake_openai.seen if st == "extract"]
    assert ext_prompts and "| T-Beta | 41.0 | 52.0 |" in ext_prompts[0]
    assert "FOCUS HINTS" in ext_prompts[0]                  # screening hints forwarded
    r1 = by_student["Stu-1B"]
    assert r1["ranking"]["ranking_string"] == "T-Beta > T-Alpha > T-Gamma"
    r3 = by_student["Stu-3B"]
    assert r3["ranking"]["ranking_string"] == "T-Alpha = T-Beta > T-Gamma"   # tie preserved
    assert r3["ranking"]["rank_columns"]["teacher_rank_1"] == "T-Alpha = T-Beta"
    # artifact verification actually inspected HF + GitHub via the fake API
    av = r1["artifact_verification"]
    assert av["trajectories"]["verified"] is True and av["trajectories"]["teacher_separable"] == "yes"
    assert av["tasks"]["verified"] is True
    assert any(c["kind"] == "github" and c["exists"] for c in av["checked"])
    assert r1["classification"]["status"] == "gold"
    # provenance
    prov = r1["provenance"]
    assert prov["doc_hash"] == r1["paper"]["doc_hash"] and prov["llm"]["openai"]["model"] == "openai-model"
    assert prov["pipeline_version"] and prov["timestamp"]
    # export
    paths = p.export(recs)
    rows = list(csv.DictReader(open(paths["csv"])))
    assert [r["student_model"] for r in rows] == ["Stu-1B", "Stu-3B"]
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert json.loads(rows[1]["teacher_scores_json"]) == {"T-Alpha": 52.0, "T-Beta": 52.0, "T-Gamma": 48.0}
    assert rows[1]["teacher_rank_1"] == "T-Alpha = T-Beta" and rows[1]["teacher_rank_2"] == "" \
        and rows[1]["teacher_rank_3"] == "T-Gamma"
    assert "Table 1" in rows[0]["table_or_section"]
    assert "fine-tune the base Stu-1B" in rows[0]["paper_evidence"]     # evidence survives to CSV
    assert "synth-org/synth-teacher-trajectories exists=True" in rows[0]["artifact_evidence"]
    assert rows[0]["status"] == "gold" and rows[0]["trajectory_dataset_hf_id"] == "synth-org/synth-teacher-trajectories"
    jl = [json.loads(l) for l in open(paths["jsonl"])]
    assert jl[0]["experiment"]["student_model"]["evidence"].startswith("fine-tune")
    assert jl[0]["screening"]["openai"]["criteria"]["multiple_teachers"]["evidence"]
    summary = paths["summary"].read_text()
    assert "- gold: 2" in summary and "distinct students (gold/silver/results_only): 2" in summary
    assert "with >=3 teachers: 2" in summary and "immediately usable" in summary
    # review queue empty for clean records
    assert len(list(csv.DictReader(open(paths["review"])))) == 0


def test_cache_resume_and_reproducible_export(work, fetcher, fake_openai):
    p = _pipe(work, fetcher, {"openai": fake_openai})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    paths = p.export(recs)
    first = {k: v.read_bytes() for k, v in paths.items()}
    calls = fake_openai.calls
    # second run: everything cached -> zero LLM calls, identical outputs
    fresh = FakeLLM(Cache(work.cache_dir), make_responder("openai"), name="openai")
    p2 = _pipe(work, FakeFetcher(Cache(work.cache_dir), default_routes()), {"openai": fresh})
    recs2 = p2.run([ARXIV_ID], run_search=False, citation_depth=0)
    assert fresh.calls == 0 and calls > 0
    paths2 = p2.export(recs2)
    for k in ("csv", "jsonl", "review"):
        assert paths2[k].read_bytes() == first[k], k          # byte-identical (timestamps live in stats only)
    # export-only from state, no LLM object involved
    p3 = _pipe(work, FakeFetcher(Cache(work.cache_dir), default_routes()), {"openai": fresh})
    assert p3.export()["csv"].read_bytes() == first["csv"] and fresh.calls == 0


def test_changing_one_prompt_reruns_only_that_stage(work, fetcher, fake_openai, monkeypatch):
    p = _pipe(work, fetcher, {"openai": fake_openai})
    p.run([ARXIV_ID], run_search=False, citation_depth=0)
    monkeypatch.setattr("litmine.pipeline.EXTRACTION_VERSION", "extract-v-test")
    fresh = FakeLLM(Cache(work.cache_dir), make_responder("openai"), name="openai")
    p2 = _pipe(work, FakeFetcher(Cache(work.cache_dir), default_routes()), {"openai": fresh})
    p2.run([ARXIV_ID], run_search=False, citation_depth=0)
    stages = [st for st, _, _ in fresh.seen]
    assert stages.count("extract") == 1 and "screen" not in stages and "prescreen" not in stages


def test_two_backends_disagreement_adjudication_and_review_queue(work, fetcher, fake_openai, fake_deepseek):
    adj = FakeLLM(Cache(work.cache_dir), make_responder("openai"), name="adjudicator")
    p = _pipe(work, fetcher, {"openai": fake_openai, "deepseek": fake_deepseek}, adjudicator=adj)
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    assert fake_openai.calls > 0 and fake_deepseek.calls > 0 and adj.calls == 2
    by_student = {r["experiment"]["student_model"]["value"]: r for r in recs}
    r1 = by_student["Stu-1B"]
    # both outputs preserved, disagreement recorded, adjudicated from source
    assert r1["extractions"]["openai"]["downstream_scores"]["SynthBench"]["T-Beta"]["value"] == 41.0
    assert r1["extractions"]["deepseek"]["downstream_scores"]["SynthBench"]["T-Beta"]["value"] == 40.0
    assert r1["disagreements"] == {"downstream_scores.tbeta": {"a": 41.0, "b": 40.0}}
    assert r1["adjudication"]["result"]["resolutions"]["downstream_scores.tbeta"]["decision"] == "openai"
    assert r1["experiment"]["downstream_scores"]["SynthBench"]["T-Beta"]["value"] == 41.0
    assert r1["ranking"]["ranking_string"] == "T-Beta > T-Alpha > T-Gamma"
    assert r1["classification"]["status"] == "gold"             # resolved from source -> status kept
    assert r1["review_flags"]                                      # ...but still listed for review
    r3 = by_student["Stu-3B"]
    assert r3["experiment"]["unresolved_fields"] == ["criteria.sft_recipe_controlled"]
    assert r3["experiment"]["criteria"]["sft_recipe_controlled"]["decision"] == "unclear"
    assert r3["classification"]["status"] == "needs_review"
    paths = p.export(recs)
    rq = list(csv.DictReader(open(paths["review"])))
    assert len(rq) == 2
    rq3 = next(r for r in rq if r["student_model"] == "Stu-3B")
    assert "criteria.sft_recipe_controlled" in rq3["unresolved_fields"]
    assert "sft_recipe_controlled unclear" in rq3["key_evidence"]
    assert "synth-org/synth-teacher-trajectories(ok)" in rq3["artifact_ids"]
    assert "needs_review: 1" in paths["summary"].read_text()


def test_screened_out_paper_produces_no_records(work, cache):
    f = FakeFetcher(cache, default_routes())
    llm = FakeLLM(cache, make_responder("irrelevant"), name="openai")
    p = _pipe(work, f, {"openai": llm})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    assert recs == [] and [st for st, _, _ in llm.seen] == ["screen"]    # seed skips prescreen
    assert p.stats["rejected_papers"][0]["arxiv_id"] == ARXIV_ID
    assert "Papers screened out" in p.export(recs)["summary"].read_text()


def test_prescreen_gates_non_seed_candidates(work, cache):
    routes = default_routes()
    routes["https://export.arxiv.org/api/query?search_query"] = (
        200, atom_feed([{"id": "2501.99999", "title": "Unrelated", "abstract": "cats"}]), {})
    routes["https://export.arxiv.org/api/query?id_list=2501.99999"] = (
        200, atom_feed([{"id": "2501.99999", "title": "Unrelated", "abstract": "cats"}]), {})
    f = FakeFetcher(cache, routes)
    seen_prescreen = []

    def pres(user):
        seen_prescreen.append(user)
        return {"relevant": "no" if "cats" in user else "yes", "reason": "r", "confidence": 0.9}
    llm = FakeLLM(cache, make_responder("openai", {"prescreen": pres}), name="openai")
    p = _pipe(work, f, {"openai": llm})
    p.settings.per_query_results = 5
    p.settings.keyword_gate = False          # exercise the LLM prescreen itself
    from litmine import discovery
    d = discovery.Discovery(f, cache)
    cands = d.run([ARXIV_ID], arxiv_queries=["q"], free_queries=[], hf_queries=[], github_queries=[], citation_depth=0)
    assert len(cands) == 2
    # drive prescreen through the pipeline's run with a monkeypatched discover
    p.discover = lambda *a, **k: cands
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    assert len(seen_prescreen) == 1 and "cats" in seen_prescreen[0]      # seed skipped, other prescreened
    assert p.stats["papers_prescreened"] == 2 and p.stats["papers_screened"] == 1 and len(recs) == 2


def test_artifact_claim_without_artifact_goes_to_review(work, cache):
    routes = default_routes()
    routes["https://huggingface.co/api/datasets/synth-org/synth-teacher-trajectories"] = (404, b"", {})
    f = FakeFetcher(cache, routes)
    llm = FakeLLM(cache, make_responder("openai"), name="openai")
    p = _pipe(work, f, {"openai": llm})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    r = recs[0]
    assert r["artifact_verification"]["trajectories"]["verified"] is False
    assert r["classification"]["status"] == "needs_review"
    assert any("could not be found" in x for x in r["classification"]["review_flags"])
    chk = next(c for c in r["artifact_verification"]["checked"] if c.get("role") == "trajectories")
    assert chk["exists"] is False and "404" in chk["error"]


def test_llm_failure_on_one_paper_does_not_abort_run(work, cache, monkeypatch):
    f = FakeFetcher(cache, default_routes())

    def boom(user):
        raise RuntimeError("provider down")
    llm = FakeLLM(cache, make_responder("openai", {"screen": boom}), name="openai")
    p = _pipe(work, f, {"openai": llm})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    assert recs == [] and p.stats["rejected_papers"][0]["reason"].startswith("ERROR")


def test_candidate_cap_prioritises_seeds_then_prescreen_yes(work, cache):
    routes = default_routes()
    feed = atom_feed([{"id": "2501.11111", "title": "Weak", "abstract": "weak maybe"},
                      {"id": "2501.22222", "title": "Strong", "abstract": "strong yes"}])
    routes["https://export.arxiv.org/api/query?search_query"] = (200, feed, {})
    for aid in ("2501.11111", "2501.22222"):
        routes[f"https://export.arxiv.org/api/query?id_list={aid}"] = (200, feed, {})
        routes[f"https://arxiv.org/html/{aid}"] = (200, SYNTH_HTML, {"Content-Type": "text/html"})

    def pres(user):
        return {"relevant": "yes" if "strong" in user else "unclear", "reason": "r", "confidence": 0.9}
    llm = FakeLLM(cache, make_responder("openai", {"prescreen": pres}), name="openai")
    p = _pipe(work, FakeFetcher(cache, routes), {"openai": llm})
    p.settings.keyword_gate = False
    from litmine.discovery import Discovery
    cands = Discovery(p.fetcher, cache).run([ARXIV_ID], arxiv_queries=["q"], free_queries=[], hf_queries=[],
                                           github_queries=[], citation_depth=0)
    p.discover = lambda *a, **k: cands
    p.run([ARXIV_ID], run_search=False, citation_depth=0, max_candidates=2)
    screened = [u.split("\n")[0] for st, _, u in llm.seen if st == "screen"]
    assert len(screened) == 2 and any("2501.22222" in s for s in screened) \
        and not any("2501.11111" in s for s in screened)      # seed + the "yes" paper, not the "unclear" one
    assert p.stats["papers_truncated_by_cap"] == 1


def test_ground_truths_json_links_each_teacher_to_trajectories(work, fetcher, fake_openai):
    p = _pipe(work, fetcher, {"openai": fake_openai})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    gt = json.loads(p.export(recs)["ground_truths"].read_text())
    assert [g["student"]["model"] for g in gt] == ["Stu-1B", "Stu-3B"]
    g = gt[1]
    assert g["task_distribution"]["hf_datasets"] == ["synth-org/synthbench-tasks"]
    assert [r["teacher"] for r in g["teacher_ranking"]] == ["T-Alpha", "T-Beta", "T-Gamma"]
    assert [r["rank"] for r in g["teacher_ranking"]] == [1, 1, 3]
    assert g["evaluation"]["scores"]["T-Alpha"] == 52.0 and "Table 1" in g["source"]["table_or_section"]
    links = g["trajectories"]["per_teacher_hf_url"]
    assert set(links) == {"T-Alpha", "T-Beta", "T-Gamma"}
    assert links["T-Alpha"].endswith("synth-org/synth-teacher-trajectories/resolve/main/data/t_alpha.jsonl")
    assert links["T-Gamma"].endswith("/resolve/main/data/t_gamma.jsonl")
    assert g["trajectories"]["all_teachers_public"] is True


def test_sibling_trajectory_datasets_are_discovered(work, fetcher, fake_openai):
    p = _pipe(work, fetcher, {"openai": fake_openai})
    recs = p.run([ARXIV_ID], run_search=False, citation_depth=0)
    checked = recs[0]["artifact_verification"]["checked"]
    sib = next(c for c in checked if c.get("id") == "synth-org/synth-teacher-trajectories-extra")
    assert sib["sibling_of"] == "synth-org/synth-teacher-trajectories" and sib["exists"]
    assert "https://huggingface.co/api/datasets?author=synth-org&limit=200" in fetcher.requested
    assert any("search=synth-teacher" in u for u in fetcher.requested)


def test_external_prescreen_file_avoids_llm_calls(work, cache, tmp_path):
    routes = default_routes()
    feed = atom_feed([{"id": "2501.44444", "title": "Agentic SFT", "abstract": "we fine-tune on trajectories"}])
    routes["https://export.arxiv.org/api/query?search_query"] = (200, feed, {})
    routes["https://export.arxiv.org/api/query?id_list=2501.44444"] = (200, feed, {})
    llm = FakeLLM(cache, make_responder("openai"), name="openai")
    p = _pipe(work, FakeFetcher(cache, routes), {"openai": llm})
    ext = tmp_path / "pres.json"
    ext.write_text(json.dumps({"arxiv:2501.44444": {"relevant": "no", "reason": "external judged off-topic",
                                                    "confidence": 0.8}}))
    p.settings.prescreen_file = str(ext)
    from litmine.discovery import Discovery
    cands = Discovery(p.fetcher, cache).run([ARXIV_ID], arxiv_queries=["q"], free_queries=[],
                                            hf_queries=[], github_queries=[], citation_depth=0)
    p.fill_metadata(cands)
    res = p.prescreen(next(c for c in cands if c.arxiv_id == "2501.44444"))
    assert res == {"relevant": "no", "reason": "external judged off-topic", "confidence": 0.8, "external": True}
    assert not any(st == "prescreen" for st, _, _ in llm.seen)
