import json
import tempfile
import unittest
from pathlib import Path

import evaluate_ranking as er
import compute_proxies as cp


STUDENT = "Qwen/Qwen3-8B"


def _write_scores(root: Path, proxy: str, rows: list[dict]) -> None:
    score_dir = root / "proxy_scores" / STUDENT.replace("/", "__")
    score_dir.mkdir(parents=True)
    (score_dir / f"{proxy}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))


class DirectionTests(unittest.TestCase):
    def test_legacy_rsr_is_migrated_to_lower_better(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            # Ratio-of-means, deliberately different from mean(row ratios).
            values = {
                "t1": {"A": (1.0, 1.0), "B": (4.0, 1.0)},
                "t2": {"A": (9.0, 9.0), "B": (4.0, 1.0)},
            }
            for task, teachers in values.items():
                for teacher, (num, den) in teachers.items():
                    rows.append({
                        "task_id": task,
                        "teacher": teacher,
                        "score": num / den,
                        "ratio_num": num,
                        "ratio_den": den,
                        # Historical files carried this incorrect direction.
                        "meta": {"direction": "higher_better",
                                 "aggregation": "ratio_of_means"},
                    })
            _write_scores(root, "rsr", rows)

            per_task, meta = er.load_scores(root, STUDENT, "rsr")
            utility, n = er.aggregate(per_task, ["A", "B"])
            raw, _ = er.aggregate(per_task, ["A", "B"], raw=True)

            self.assertEqual(n, 2)
            self.assertEqual(meta["direction"], "lower_better")
            self.assertEqual(meta["legacy_declared_direction"], "higher_better")
            self.assertAlmostEqual(raw["A"], 1.0)
            self.assertAlmostEqual(raw["B"], 4.0)
            self.assertGreater(utility["A"], utility["B"])

            diag = er.trajectory_diagnostics(per_task, ["A", "B"])
            self.assertEqual(diag["raw_score_direction"], "lower_better")
            self.assertEqual(
                diag["task_winner_fractional_share"], {"A": 1.0, "B": 0.0})
            # Distribution summaries remain interpretable native RSR values.
            self.assertEqual(diag["within_teacher"]["A"]["mean"], 1.0)

    def test_explicit_lower_better_ordinary_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scores(root, "cost", [
                {"task_id": "t", "teacher": "A", "score": 2.0,
                 "meta": {"direction": "lower_better"}},
                {"task_id": "t", "teacher": "B", "score": 3.0,
                 "meta": {"direction": "lower_better"}},
            ])
            per_task, _ = er.load_scores(root, STUDENT, "cost")
            utility, _ = er.aggregate(per_task, ["A", "B"])
            self.assertEqual(utility, {"A": -2.0, "B": -3.0})

    def test_score_views_are_preoriented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scores(root, "error", [
                {"task_id": "t", "teacher": "A",
                 "score_views": {"fewer": -0.1, "more": 0.1},
                 "meta": {"direction": "multiple_predeclared"}},
                {"task_id": "t", "teacher": "B",
                 "score_views": {"fewer": -0.4, "more": 0.4},
                 "meta": {"direction": "multiple_predeclared"}},
            ])
            sets = er.load_score_sets(root, STUDENT, "error")
            fewer, meta = sets["error/fewer"]
            utility, _ = er.aggregate(fewer, ["A", "B"])
            self.assertGreater(utility["A"], utility["B"])
            self.assertEqual(meta["direction"], "higher_better")

    def test_ambiguous_direction_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scores(root, "bad", [
                {"task_id": "t", "teacher": "A", "score": 1.0,
                 "meta": {"direction": "unknown"}},
            ])
            with self.assertRaisesRegex(ValueError, "unsupported or ambiguous"):
                er.load_scores(root, STUDENT, "bad")

    def test_corrected_reported_rsr_order_has_negative_tau(self):
        # Historical raw order was Q35 > GLM > DS > CL. Since native RSR is
        # lower-is-better, the corrected prediction is its reverse.
        raw = {"Q35": 4.0, "GLM": 3.0, "DS": 2.0, "CL": 1.0}
        utility = {teacher: -value for teacher, value in raw.items()}
        published = [["DS"], ["GLM", "Q35"], ["CL"]]
        metrics = er.ranking_metrics(utility, published)
        self.assertEqual(metrics["predicted_order"],
                         ["CL", "DS", "GLM", "Q35"])
        self.assertAlmostEqual(metrics["kendall_tau_b"],
                               -0.18257418583505536)
        self.assertEqual(metrics["pairwise_accuracy"], 0.4)

    def test_grace_uses_same_generic_lower_better_path(self):
        per_task = er.PerTask("grace_teacher_level", "lower_better")
        per_task["t"]["A"] = [2.0]
        per_task["t"]["B"] = [3.0]
        old_fns = er._GRACE_FN[:]
        try:
            er._GRACE_FN[:] = [lambda vectors: vectors[0][0]]
            er._GRACE_MEMO.clear()
            raw, _ = er.aggregate(per_task, ["A", "B"], raw=True)
            utility, _ = er.aggregate(per_task, ["A", "B"])
        finally:
            er._GRACE_FN[:] = old_fns
            er._GRACE_MEMO.clear()
        self.assertEqual(raw, {"A": 2.0, "B": 3.0})
        self.assertEqual(utility, {"A": -2.0, "B": -3.0})


class ProxyComponentTests(unittest.TestCase):
    def test_parse_gpt_turn_accepts_exported_reasoning_prefix(self):
        turn = cp.parse_gpt_turn(
            'reasoning before payload</think>{"analysis":"a",'
            '"commands":[{"keystrokes":"ls -la\\n"}]}')
        self.assertEqual(turn["commands"][0]["keystrokes"], "ls -la\n")

    def test_tor_accepts_role_content_conversations(self):
        legacy = {"conversations": [
            {"from": "human", "value": "Working Directory: /testbed"},
            {"from": "gpt", "value": '{"commands":[{"keystrokes":"cat a.py\\n"}]}'},
            {"from": "human", "value":
             "(testbed) root@abc123-456:/testbed# cat a.py\nold\n"
             "(testbed) root@abc123-456:/testbed#"},
            {"from": "gpt", "value":
             'thinking</think>{"commands":[{"keystrokes":"sed -i s/old/new/ a.py\\n"}]}'},
            {"from": "human", "value":
             "(testbed) root@abc123-456:/testbed# sed -i s/old/new/ a.py\n"
             "(testbed) root@abc123-456:/testbed#"},
        ]}
        modern = {"conversations": [
            {"role": "user", "content": msg["value"]}
            if msg["from"] == "human"
            else {"role": "assistant", "content": msg["value"]}
            for msg in legacy["conversations"]
        ]}
        self.assertEqual(cp._trajectory_grounding_components(legacy),
                         cp._trajectory_grounding_components(modern))
        self.assertEqual(
            cp._trajectory_grounding_components(modern)["tor"], 1.0)

    def test_aslec_primary_uses_released_skip2_and_counts_steps(self):
        steps = [[-1.0, -2.0, -3.0], [-4.0, -5.0]]
        parts = cp._aslec_components(steps)
        self.assertEqual(cp.ASLEC_SKIP_TOKENS, 2)
        self.assertEqual(parts["n_steps"], 2)
        self.assertEqual(parts["n_head_tokens"], 4)
        self.assertEqual(parts["drop_score"], -3.0)
        self.assertEqual(parts["first_token_ratio"], 4 / 5)

    def test_legacy_changed_proxy_caches_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aslec = root / "aslec_drop.jsonl"
            aslec.write_text(json.dumps({
                "teacher": "A", "task_id": "t",
                "meta": {"skip_tokens": 1},
            }) + "\n")
            self.assertIn("cached skip_tokens=1",
                          cp.score_cache_incompatibility("aslec_drop", aslec))

            scas = root / "scas.jsonl"
            scas.write_text(json.dumps({
                "teacher": "A", "task_id": "t",
                "meta": {"direction": "higher_better"},
            }) + "\n")
            self.assertIn("predates",
                          cp.score_cache_incompatibility("scas", scas))


if __name__ == "__main__":
    unittest.main()
