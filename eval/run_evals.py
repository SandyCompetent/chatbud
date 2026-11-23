#!/usr/bin/env python3
"""ChatBud Automated Evaluation Suite CLI Runner.

Two modes (always printed and stored in the reports):

* default, no ``--responses``  -> ``dataset_check``: validates the golden dataset
  (schema, balance, reference answers supported by their own context). This does
  NOT measure ChatBud or any model.
* ``--responses FILE``         -> ``model_responses``: scores real model answers
  (create the file with ``generate_responses.py``) for faithfulness, relevance
  and hallucination rate. Context precision is only reported when the file also
  contains the retrieved chunks.

The metrics are lexical heuristics inspired by Ragas/DeepEval, not LLM-judged.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from src.evaluators import MODE_DATASET_CHECK, EvaluationSuite
from src.report_generator import save_reports


def load_responses(path: str, queries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Load model responses; fail loudly if any golden query has no response."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    raw_map = raw.get("responses", raw) if isinstance(raw, dict) else {}
    responses: Dict[str, Dict[str, Any]] = {}
    for query_id, value in raw_map.items():
        if isinstance(value, str):
            responses[query_id] = {"response": value}
        elif isinstance(value, dict) and isinstance(value.get("response"), str):
            responses[query_id] = value

    missing = [q["id"] for q in queries if q["id"] not in responses]
    if missing:
        print(
            f"Error: {len(missing)} of {len(queries)} queries have no response in '{path}' "
            f"(first missing: {', '.join(missing[:5])}). "
            "Run eval/generate_responses.py to complete the file.",
            file=sys.stderr,
        )
        sys.exit(2)
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ChatBud evaluation suite (dataset check or model-response scoring)"
    )
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(here, "data", "golden_eval_dataset.json"),
        help="Path to golden ground truth dataset JSON",
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=None,
        help="JSON file of real model responses (from generate_responses.py). Omit for dataset check.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(here, "reports"),
        help="Output directory for generated evaluation reports",
    )
    parser.add_argument("--threshold-faithfulness", type=float, default=0.85)
    parser.add_argument("--threshold-relevance", type=float, default=0.85)
    parser.add_argument("--threshold-precision", type=float, default=0.80)
    parser.add_argument("--max-hallucination", type=float, default=0.08)
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at '{args.dataset}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading evaluation dataset from: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as handle:
        queries = json.load(handle).get("queries", [])
    print(f"Loaded {len(queries)} golden queries.")

    responses = None
    responses_metadata: Dict[str, Any] = {}
    if args.responses:
        if not os.path.exists(args.responses):
            print(f"Error: responses file not found at '{args.responses}'", file=sys.stderr)
            sys.exit(1)
        responses = load_responses(args.responses, queries)
        with open(args.responses, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            responses_metadata = raw.get("metadata", {})

    suite = EvaluationSuite(
        threshold_faithfulness=args.threshold_faithfulness,
        threshold_relevance=args.threshold_relevance,
        threshold_precision=args.threshold_precision,
        max_hallucination=args.max_hallucination,
    )
    eval_results = suite.evaluate_dataset(queries, responses=responses)
    summary = eval_results["summary"]
    summary["responses_metadata"] = responses_metadata
    metrics = summary["metrics"]

    def show(value: Any) -> str:
        return "not measured" if value is None else f"{value:.3f}"

    print("\n" + "=" * 65)
    print("   ChatBud Evaluation Suite")
    print("=" * 65)
    print(f"Mode:                 {summary['mode']}")
    if responses_metadata.get("model"):
        print(f"Model:                {responses_metadata['model']}")
    print(f"Status:               {summary['overall_status']}")
    print(f"Queries:              {summary['total_queries']}")
    print(f"Items passing:        {summary['passed_queries']} ({summary['pass_rate_percentage']}%)")
    print("-" * 65)

    if summary["mode"] == MODE_DATASET_CHECK:
        print("This is a dataset sanity check. It does not measure ChatBud's model.")
        print(
            f"Reference answers supported by their context: {show(metrics['reference_support'])} "
            f"(needs >= {args.threshold_faithfulness:.2f})"
        )
    else:
        print(f"Faithfulness:         {show(metrics['faithfulness'])}  (gate >= {args.threshold_faithfulness:.2f})")
        print(f"Answer Relevance:     {show(metrics['answer_relevance'])}  (gate >= {args.threshold_relevance:.2f})")
        print(f"Context Precision:    {show(metrics['context_precision'])}  (gate >= {args.threshold_precision:.2f})")
        print(f"Hallucination Rate:   {show(metrics['hallucination_rate'])}  (gate <= {args.max_hallucination:.2f})")

    if summary["dataset_problems"]:
        print("-" * 65)
        print("Dataset problems:")
        for problem in summary["dataset_problems"][:10]:
            print(f"  - {problem}")
    print("=" * 65 + "\n")

    save_reports(eval_results, args.output_dir)

    if summary["overall_status"] in ("PASSED", "DATASET_OK"):
        print("\nResult: OK")
        sys.exit(0)
    print("\nResult: FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
