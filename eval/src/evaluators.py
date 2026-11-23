"""Evaluation runner and test case orchestrator for ChatBud.

The suite runs in one of two modes, and the mode is always recorded in the report:

``dataset_check`` (default, no model involved)
    Validates the golden dataset itself: schema, balance across categories and
    that each reference answer is supported by its own ground-truth context.
    It says nothing about ChatBud's model quality, so no faithfulness,
    relevance, precision or hallucination figures are produced.

``model_responses``
    Scores real model responses (see ``generate_responses.py``) against the
    ground-truth context and reference answers. This is the only mode that
    produces benchmark metrics. Context precision is reported only when the
    responses file also contains the chunks that were actually retrieved;
    otherwise it is "not measured".

All metrics are lexical heuristics (see ``metrics.py``), not LLM-judged
Ragas/DeepEval scores.
"""

import time
from typing import Any, Dict, List, Optional

from .metrics import (
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
    compute_hallucination_rate,
)

MODE_DATASET_CHECK = "dataset_check"
MODE_MODEL_RESPONSES = "model_responses"

EXPECTED_CATEGORY_SIZE = 25
REQUIRED_FIELDS = ("id", "category", "query", "ground_truth_context", "reference_answer")


class EvaluationItemResult:
    """Stores evaluation metrics and audit details for a single test case.

    Metrics that were not measured are ``None`` (never silently 1.0).
    """

    def __init__(
        self,
        item_id: str,
        category: str,
        persona_id: str,
        query: str,
        context: List[str],
        response: Optional[str],
        reference_answer: str,
        faithfulness: Optional[float],
        answer_relevance: Optional[float],
        context_precision: Optional[float],
        hallucination_rate: Optional[float],
        reference_support: Optional[float],
        passed: bool,
        notes: str = "",
    ):
        self.item_id = item_id
        self.category = category
        self.persona_id = persona_id
        self.query = query
        self.context = context
        self.response = response
        self.reference_answer = reference_answer
        self.faithfulness = faithfulness
        self.answer_relevance = answer_relevance
        self.context_precision = context_precision
        self.hallucination_rate = hallucination_rate
        self.reference_support = reference_support
        self.passed = passed
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.item_id,
            "category": self.category,
            "persona_id": self.persona_id,
            "query": self.query,
            "response": self.response,
            "reference_answer": self.reference_answer,
            "metrics": {
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
                "context_precision": self.context_precision,
                "hallucination_rate": self.hallucination_rate,
                "reference_support": self.reference_support,
            },
            "passed": self.passed,
            "notes": self.notes,
        }


def _mean(values: List[Optional[float]], digits: int = 4) -> Optional[float]:
    """Mean of the measured values, or ``None`` when nothing was measured."""
    measured = [v for v in values if v is not None]
    if not measured:
        return None
    return round(sum(measured) / len(measured), digits)


def validate_dataset(dataset: List[Dict[str, Any]]) -> List[str]:
    """Return a list of human-readable problems found in the golden dataset."""
    problems: List[str] = []
    seen_ids = set()
    category_counts: Dict[str, int] = {}

    for index, item in enumerate(dataset):
        label = item.get("id", f"#{index}")
        for field in REQUIRED_FIELDS:
            if not item.get(field):
                problems.append(f"{label}: missing or empty '{field}'")
        item_id = item.get("id")
        if item_id in seen_ids:
            problems.append(f"{label}: duplicate id")
        seen_ids.add(item_id)
        category = item.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in category_counts.items():
        if count != EXPECTED_CATEGORY_SIZE:
            problems.append(
                f"category '{category}' has {count} queries, expected {EXPECTED_CATEGORY_SIZE}"
            )
    return problems


class EvaluationSuite:
    """Executes benchmark evaluations across dataset queries."""

    def __init__(
        self,
        threshold_faithfulness: float = 0.85,
        threshold_relevance: float = 0.85,
        threshold_precision: float = 0.80,
        max_hallucination: float = 0.08,
        threshold_reference_support: float = 0.85,
    ):
        self.threshold_faithfulness = threshold_faithfulness
        self.threshold_relevance = threshold_relevance
        self.threshold_precision = threshold_precision
        self.max_hallucination = max_hallucination
        self.threshold_reference_support = threshold_reference_support

    # ------------------------------------------------------------------ items

    def evaluate_item(
        self,
        item: Dict[str, Any],
        response: Optional[str] = None,
        retrieved_context: Optional[List[str]] = None,
    ) -> EvaluationItemResult:
        """Evaluate one query.

        Without ``response`` this only checks the reference answer against the
        ground-truth context (dataset sanity check). With ``response`` it scores
        the model output.
        """
        query = item.get("query", "")
        ground_truth_context = item.get("ground_truth_context", [])
        reference_answer = item.get("reference_answer", "")

        common = dict(
            item_id=item.get("id", "unknown"),
            category=item.get("category", "general"),
            persona_id=item.get("persona_id", "default"),
            query=query,
            context=ground_truth_context,
            reference_answer=reference_answer,
        )

        if response is None:
            support, _, _ = compute_faithfulness(reference_answer, ground_truth_context)
            return EvaluationItemResult(
                response=None,
                faithfulness=None,
                answer_relevance=None,
                context_precision=None,
                hallucination_rate=None,
                reference_support=support,
                passed=support >= 0.80,
                notes="dataset check only: no model response was scored",
                **common,
            )

        faithfulness, _, _ = compute_faithfulness(response, ground_truth_context, query)
        answer_relevance = compute_answer_relevance(query, response, reference_answer)
        hallucination_rate = compute_hallucination_rate(faithfulness)

        context_precision: Optional[float] = None
        notes = "context precision not measured: no retrieved chunks supplied"
        if retrieved_context is not None:
            context_precision = compute_context_precision(retrieved_context, ground_truth_context)
            notes = ""

        # Individual item passes when meeting benchmark baseline tolerance.
        passed = (
            faithfulness >= 0.80
            and answer_relevance >= 0.70
            and hallucination_rate <= 0.20
            and (context_precision is None or context_precision >= 0.75)
        )

        return EvaluationItemResult(
            response=response,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            hallucination_rate=hallucination_rate,
            reference_support=None,
            passed=passed,
            notes=notes,
            **common,
        )

    # ---------------------------------------------------------------- dataset

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        responses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate every query.

        ``responses`` maps query id to ``{"response": str, "retrieved_context": [str] | None}``.
        When omitted the suite runs in ``dataset_check`` mode.
        """
        mode = MODE_MODEL_RESPONSES if responses is not None else MODE_DATASET_CHECK
        start_time = time.time()
        results: List[EvaluationItemResult] = []

        for item in dataset:
            if responses is None:
                results.append(self.evaluate_item(item))
                continue
            entry = responses[item["id"]]
            results.append(
                self.evaluate_item(
                    item,
                    response=entry["response"],
                    retrieved_context=entry.get("retrieved_context"),
                )
            )

        elapsed = round(time.time() - start_time, 2)
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)

        metrics = {
            "faithfulness": _mean([r.faithfulness for r in results]),
            "answer_relevance": _mean([r.answer_relevance for r in results]),
            "context_precision": _mean([r.context_precision for r in results]),
            "hallucination_rate": _mean([r.hallucination_rate for r in results]),
            "reference_support": _mean([r.reference_support for r in results]),
        }

        domains_formatted: Dict[str, Dict[str, Any]] = {}
        categories = sorted({r.category for r in results})
        for cat in categories:
            group = [r for r in results if r.category == cat]
            cnt = len(group)
            group_passed = sum(1 for r in group if r.passed)
            domains_formatted[cat] = {
                "total": cnt,
                "passed": group_passed,
                "pass_rate": round(group_passed / cnt * 100, 1),
                "faithfulness": _mean([r.faithfulness for r in group]),
                "answer_relevance": _mean([r.answer_relevance for r in group]),
                "context_precision": _mean([r.context_precision for r in group]),
                "hallucination_rate": _mean([r.hallucination_rate for r in group]),
                "reference_support": _mean([r.reference_support for r in group]),
            }

        dataset_problems = validate_dataset(dataset)

        if mode == MODE_DATASET_CHECK:
            support = metrics["reference_support"]
            ok = (
                not dataset_problems
                and support is not None
                and support >= self.threshold_reference_support
            )
            overall_status = "DATASET_OK" if ok else "DATASET_INVALID"
        else:
            gates = [
                (metrics["faithfulness"], self.threshold_faithfulness, ">="),
                (metrics["answer_relevance"], self.threshold_relevance, ">="),
                (metrics["context_precision"], self.threshold_precision, ">="),
                (metrics["hallucination_rate"], self.max_hallucination, "<="),
            ]
            ok = not dataset_problems
            for value, limit, op in gates:
                if value is None:
                    continue  # not measured: no claim either way
                ok = ok and (value >= limit if op == ">=" else value <= limit)
            overall_status = "PASSED" if ok else "FAILED"

        return {
            "summary": {
                "mode": mode,
                "total_queries": total,
                "passed_queries": passed_count,
                "pass_rate_percentage": round(passed_count / max(1, total) * 100, 1),
                "elapsed_seconds": elapsed,
                "overall_status": overall_status,
                "metrics": metrics,
                "thresholds": {
                    "faithfulness": self.threshold_faithfulness,
                    "answer_relevance": self.threshold_relevance,
                    "context_precision": self.threshold_precision,
                    "hallucination_rate": self.max_hallucination,
                    "reference_support": self.threshold_reference_support,
                },
                "dataset_problems": dataset_problems,
            },
            "domains": domains_formatted,
            "results": [r.to_dict() for r in results],
        }
