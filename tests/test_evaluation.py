from pathlib import Path

import pytest

from app.evaluation import load_cases, score_case, compare_profiles


CASES = Path("data/benchmark/BoCauHoi/questions_300.jsonl")


def test_benchmark_split_counts_exclude_security():
    assert len(load_cases(CASES, "smoke")) == 51
    assert len(load_cases(CASES, "regression")) == 111
    assert len(load_cases(CASES, "holdout")) == 162
    assert len(load_cases(CASES, "all", include_security=True)) == 300


def test_document_recall_counts_distinct_expected_documents():
    case = next(case for case in load_cases(CASES, "smoke") if len(case.expected_document_ids) > 1)
    result = {"retrieved_document_ids": [case.expected_document_ids[0]],
              "cited_document_ids": [case.expected_document_ids[0]], "abstained": False}

    scores = score_case(case, result)
    assert scores["document_recall"] == 1 / len(case.expected_document_ids)
    assert scores["all_gold_retrieved"] is False


def test_comparison_requires_same_case_ids():
    base = {"category": "DIRECT", "latency_seconds": 0.2,
            "scores": {"document_recall": 1.0},
            "retrieved_document_ids": ["doc-a"]}
    with pytest.raises(ValueError, match="same case IDs"):
        compare_profiles({"A": [{**base, "case_id": "one"}],
                          "B": [{**base, "case_id": "two"}]})
