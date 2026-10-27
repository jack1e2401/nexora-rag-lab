"""Benchmark selection and deterministic RAG scores."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    board_id: str
    question: str
    category: str
    split: str
    expected_behavior: str
    expected_answer: str
    expected_document_ids: list[str]
    scenario_id: str


SPLITS = {
    "smoke": {"SMOKE_60"},
    "regression": {"SMOKE_60", "REGRESSION_120"},
    "holdout": {"HOLDOUT_180"},
    "all": {"SMOKE_60", "REGRESSION_120", "HOLDOUT_180"},
}


def load_cases(path: Path, split: str, include_security: bool = False) -> list[BenchmarkCase]:
    """Validate all IDs, then select one split and optionally skip SECURITY."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    selected = []
    seen = set()
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in seen:
                raise ValueError(f"Duplicate benchmark case ID: {item['id']}")
            seen.add(item["id"])
            if item["split"] not in SPLITS[split]:
                continue
            if item["category"] == "SECURITY" and not include_security:
                continue
            selected.append(BenchmarkCase(
                case_id=item["id"], board_id=item["board_id"], question=item["question"],
                category=item["category"], split=item["split"],
                expected_behavior=item["expected_behavior"],
                expected_answer=item["expected_answer"],
                expected_document_ids=item["expected_document_ids"],
                scenario_id=item["scenario_id"],
            ))
    if not selected:
        raise ValueError("Selected benchmark split is empty")
    return selected


def score_case(case: BenchmarkCase, result: dict) -> dict:
    """Score retrieval and citation IDs without calling an LLM judge."""
    expected = set(case.expected_document_ids)
    retrieved = set(result["retrieved_document_ids"])
    cited = result["cited_document_ids"]

    return {
        "document_recall": len(expected & retrieved) / len(expected) if expected else None,
        "any_gold_retrieved": bool(expected & retrieved) if expected else None,
        "all_gold_retrieved": expected <= retrieved if expected else None,
        "citation_validity": (all(document_id in retrieved for document_id in cited)
                               if cited else None),
        "citation_precision": (sum(document_id in expected for document_id in cited) / len(cited)
                               if cited and expected else None),
        "abstention_accuracy": result["abstained"] == (case.expected_behavior == "ABSTAIN"),
    }


def compare_profiles(results: dict[str, list[dict]]) -> dict:
    """Produce aggregates and per-question recall deltas for two profile runs."""
    if len(results) != 2:
        raise ValueError("Exactly two profiles are required for comparison")
    if any(not rows for rows in results.values()):
        raise ValueError("Profile results cannot be empty")
    profile_ids = list(results)
    expected_ids = {row["case_id"] for row in results[profile_ids[0]]}
    for profile_id in profile_ids[1:]:
        if {row["case_id"] for row in results[profile_id]} != expected_ids:
            raise ValueError("Profiles must evaluate the same case IDs")

    summary = {}
    category_summary = {}
    for profile_id, rows in results.items():
        metric_names = rows[0]["scores"].keys()
        summary[profile_id] = {}
        for name in metric_names:
            values = [row["scores"][name] for row in rows if row["scores"][name] is not None]
            summary[profile_id][name] = sum(values) / len(values) if values else None
        summary[profile_id]["mean_latency_seconds"] = sum(
            row["latency_seconds"] for row in rows
        ) / len(rows)

        categories = dict.fromkeys(row["category"] for row in rows)
        category_summary[profile_id] = {}
        for category in categories:
            category_rows = [row for row in rows if row["category"] == category]
            category_summary[profile_id][category] = {}
            for name in metric_names:
                values = [row["scores"][name] for row in category_rows
                          if row["scores"][name] is not None]
                category_summary[profile_id][category][name] = (
                    sum(values) / len(values) if values else None
                )
            category_summary[profile_id][category]["mean_latency_seconds"] = sum(
                row["latency_seconds"] for row in category_rows
            ) / len(category_rows)

    first = {row["case_id"]: row for row in results[profile_ids[0]]}
    second = {row["case_id"]: row for row in results[profile_ids[1]]}
    deltas = []
    for case_id in sorted(expected_ids):
        left = first[case_id]["scores"]["document_recall"]
        right = second[case_id]["scores"]["document_recall"]
        if left is not None and right is not None and left != right:
            deltas.append({"case_id": case_id, "category": first[case_id]["category"],
                           "document_recall_delta": right - left,
                           "first_retrieved": first[case_id]["retrieved_document_ids"],
                           "second_retrieved": second[case_id]["retrieved_document_ids"]})
    return {"profiles": summary, "by_category": category_summary,
            "paired_recall_differences": deltas}
