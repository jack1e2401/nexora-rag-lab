import sys
from types import ModuleType

from app.config import Settings
from app.evaluation import BenchmarkCase
from scripts import evaluate
import pytest


def test_profiles_share_one_precomputed_query_vector(monkeypatch):
    case = BenchmarkCase("q1", "board-a", "Question?", "DIRECT", "SMOKE_60",
                         "ANSWER", "Answer", ["doc-a"], "scenario-1")
    calls = []

    monkeypatch.setattr(evaluate, "profile_metadata", lambda *args: {"corpus_fingerprint": "same"})
    monkeypatch.setattr(evaluate, "embed_query", lambda *args: [0.1] * 1024)

    def fake_run_case(case_id, question, board_id, profile_id, top_k, vector, settings, mode):
        calls.append((profile_id, vector))
        return {"case_id": case_id, "retrieved_document_ids": ["doc-a"],
                "cited_document_ids": [], "abstained": False, "answer": "", "citations": []}

    monkeypatch.setattr(evaluate, "run_case", fake_run_case)
    settings = Settings(database_url="postgresql://localhost/test")

    result = evaluate.evaluate_profiles([case], ["A", "B"], settings, 3, "retrieval")

    assert [item[0] for item in calls] == ["A", "B"]
    assert calls[0][1] is calls[1][1]
    assert result["A"][0]["scores"]["document_recall"] == 1.0


def test_profiles_from_different_corpora_are_rejected(monkeypatch):
    case = BenchmarkCase("q1", "board-a", "Question?", "DIRECT", "SMOKE_60",
                         "ANSWER", "Answer", ["doc-a"], "scenario-1")
    monkeypatch.setattr(evaluate, "profile_metadata", lambda profile, settings: {
        "corpus_fingerprint": profile,
    })
    monkeypatch.setattr(evaluate, "embed_query", lambda *args: pytest.fail("embedded before validation"))

    with pytest.raises(ValueError, match="different corpora"):
        evaluate.evaluate_profiles(
            [case], ["fingerprint-a", "fingerprint-b"],
            Settings(database_url="postgresql://localhost/test"), 3, "retrieval",
        )


def test_langfuse_missing_cases_are_reported(monkeypatch):
    case = BenchmarkCase("q1", "board-a", "Question?", "DIRECT", "SMOKE_60",
                         "ANSWER", "Answer", ["doc-a"], "scenario-1")
    metadata = {"corpus_fingerprint": "same"}
    monkeypatch.setattr(evaluate, "profile_metadata", lambda *args: metadata)
    monkeypatch.setattr(evaluate, "embed_query", lambda *args: [0.1] * 1024)

    class IncompleteLangfuse:
        def run_experiment(self, **kwargs):
            return None

    with pytest.raises(RuntimeError, match="did not finish cases"):
        evaluate.evaluate_profiles(
            [case], ["A", "B"], Settings(database_url="postgresql://localhost/test"),
            3, "retrieval", langfuse_client=IncompleteLangfuse(),
        )


def test_langfuse_runner_attaches_scores(monkeypatch):
    case = BenchmarkCase("q1", "board-a", "Question?", "DIRECT", "SMOKE_60",
                         "ANSWER", "Answer", ["doc-a"], "scenario-1")
    monkeypatch.setattr(evaluate, "profile_metadata", lambda *args: {"corpus_fingerprint": "same"})
    monkeypatch.setattr(evaluate, "embed_query", lambda *args: [0.1] * 1024)
    monkeypatch.setattr(evaluate, "run_case", lambda case_id, *args: {
        "case_id": case_id, "retrieved_document_ids": ["doc-a"],
        "cited_document_ids": ["doc-a"], "abstained": False,
        "answer": "Answer", "citations": [],
    })

    class Evaluation:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    langfuse_module = ModuleType("langfuse")
    langfuse_module.Evaluation = Evaluation
    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)

    class FakeLangfuse:
        def __init__(self):
            self.scores = []

        def run_experiment(self, **kwargs):
            for item in kwargs["data"]:
                output = kwargs["task"](item=item)
                self.scores.extend(kwargs["evaluators"][0](output=output))

    client = FakeLangfuse()
    results = evaluate.evaluate_profiles(
        [case], ["A", "B"], Settings(database_url="postgresql://localhost/test"),
        3, "retrieval", langfuse_client=client,
    )

    assert len(results["A"]) == 1
    assert len(client.scores) > 0
