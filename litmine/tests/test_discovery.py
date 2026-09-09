from conftest import ARXIV_ID, FakeFetcher, atom_feed, default_routes
from litmine.discovery import Discovery, build_arxiv_queries, build_free_text_queries


def test_queries_are_many_and_combinatorial():
    qs = build_arxiv_queries()
    assert len(qs) > 50 and len(set(qs)) == len(qs)
    assert any("teacher" in q and "agent" in q for q in qs)
    assert len(build_free_text_queries()) >= 5


def test_arxiv_search_parses_and_dedupes(cache):
    routes = default_routes()
    routes["https://export.arxiv.org/api/query?search_query"] = (
        200, atom_feed([{"id": "2501.11111", "title": "Paper A", "abstract": "abs a"},
                        {"id": "2501.22222", "title": "Paper B"}]), {})
    f = FakeFetcher(cache, routes)
    d = Discovery(f, cache, per_query=5)
    r1 = d.arxiv_search("q1")
    r2 = d.arxiv_search("q2")
    assert {c.arxiv_id for c in r1} == {"2501.11111", "2501.22222"}
    assert len(d.candidates) == 2
    assert d.candidates["arxiv:2501.11111"].sources == ["arxiv:q1", "arxiv:q2"]
    n = len(f.requested)
    d.arxiv_search("q1")                                    # cached: no new HTTP
    assert len(f.requested) == n


def test_hf_tags_yield_paper_candidates(cache):
    routes = default_routes()
    routes["https://huggingface.co/api/datasets?"] = (200, [
        {"id": "org/ds", "tags": ["arxiv:2501.33333", "task_categories:text-generation"]},
        {"id": "org/other", "tags": []}], {})
    d = Discovery(FakeFetcher(cache, routes), cache)
    out = d.hf_search("teacher trajectories")
    assert [c.arxiv_id for c in out] == ["2501.33333"]
    assert out[0].tags == ["hf_dataset:org/ds"]


def test_openalex_expansion_follows_citations_and_references(cache):
    routes = default_routes()
    work = {"id": "https://openalex.org/W1", "title": "Seed", "publication_year": 2026,
            "locations": [{"landing_page_url": f"https://arxiv.org/abs/{ARXIV_ID}"}],
            "referenced_works": ["https://openalex.org/W2"], "ids": {}}
    citing = {"id": "https://openalex.org/W3", "title": "Citing paper", "ids": {},
              "locations": [{"landing_page_url": "https://arxiv.org/abs/2502.00003"}],
              "abstract_inverted_index": {"We": [0], "cite": [1]}}
    ref = {"id": "https://openalex.org/W2", "title": "Referenced paper", "ids": {"doi": "https://doi.org/10.1/x"},
           "locations": []}

    def route(url):
        if "filter=locations.landing_page_url" in url:
            return 200, {"results": [work]}, {}
        if "filter=cites" in url:
            return 200, {"results": [citing]}, {}
        if "filter=openalex" in url:
            return 200, {"results": [ref]}, {}
        return 200, {"results": []}, {}
    routes["https://api.openalex.org/works?"] = route
    d = Discovery(FakeFetcher(cache, routes), cache)
    seed = d.add_seed(ARXIV_ID)
    d.expand_citations([seed], depth=1)
    keys = set(d.candidates)
    assert "arxiv:2502.00003" in keys and "url:https://doi.org/10.1/x" in keys
    assert d.candidates["arxiv:2502.00003"].abstract == "We cite" and d.candidates["arxiv:2502.00003"].hops == 1


def test_run_without_search_only_seeds(cache):
    d = Discovery(FakeFetcher(cache, default_routes()), cache)
    cands = d.run([ARXIV_ID], arxiv_queries=[], free_queries=[], hf_queries=[], github_queries=[],
                  citation_depth=0)
    assert [c.key for c in cands] == [f"arxiv:{ARXIV_ID}"] and cands[0].sources == ["seed"]


def test_keyword_gate_requires_agentic_and_training_vocabulary():
    from litmine.discovery import keyword_gate
    assert keyword_gate("Data Recipes for Agentic Models", "we fine-tune students on trajectories")
    assert keyword_gate("SWE-bench data", "training data from software engineering environments")
    assert not keyword_gate("Unrelated", "cats")
    assert not keyword_gate("Agents everywhere", "a survey of prompting strategies")     # agentic, no training
    assert not keyword_gate("Fine-tuning for math", "supervised fine-tuning on GSM8K")   # training, not agentic


def test_paged_search_stops_on_empty_page_and_filters_year(cache):
    routes = default_routes()
    calls = []

    def route(url):
        calls.append(url)
        if "start=0" in url:
            return 200, atom_feed([{"id": "2501.11111", "title": "New", "abstract": "a"},
                                   {"id": "2001.22222", "title": "Old", "abstract": "b", "year": 2020}]), {}
        return 200, atom_feed([]), {}
    routes["https://export.arxiv.org/api/query?search_query"] = route
    d = Discovery(FakeFetcher(cache, routes), cache)
    out = d.arxiv_search_paged("q", total=600, page=2, min_year=2023)
    assert [c.arxiv_id for c in out] == ["2501.11111"]
    assert len(calls) == 2                       # page 2 empty -> stop, page 3 never requested


def test_seed_titles_resolve_by_title_and_batch_metadata(cache):
    routes = default_routes()
    feed = atom_feed([{"id": "2501.33333", "title": "SWE-smith: Scaling Data for Software Engineering Agents",
                       "abstract": "we fine-tune agents on trajectories"}])
    routes["https://export.arxiv.org/api/query?search_query"] = (200, feed, {})
    routes["https://export.arxiv.org/api/query?id_list=2501.33333"] = (200, feed, {})
    d = Discovery(FakeFetcher(cache, routes), cache)
    seeds = d.resolve_seed_titles(["SWE-smith: Scaling Data for Software Engineering Agents", "Nonexistent Paper XYZ"])
    assert [c.arxiv_id for c in seeds] == ["2501.33333"] and seeds[0].sources == ["seed_title"]
    assert d.seed_resolution["Nonexistent Paper XYZ"] is None
    meta = d.arxiv_metadata(["2501.33333"])
    assert meta["2501.33333"]["abstract"].startswith("we fine-tune")


def test_run_without_search_mode_adds_no_title_seeds_or_broad_queries(cache):
    f = FakeFetcher(cache, default_routes())
    d = Discovery(f, cache)
    d.run([ARXIV_ID], arxiv_queries=[], free_queries=[], hf_queries=[], github_queries=[], citation_depth=0)
    assert not any("search_query" in u for u in f.requested)
