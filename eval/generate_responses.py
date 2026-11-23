#!/usr/bin/env python3
"""Generate real model responses for the ChatBud golden dataset.

For every golden query this sends the grounding context and the query to the
Gemini API (using ``prompt_template.txt``) and stores the answer, so that
``run_evals.py --responses`` can score what a model actually said instead of
scoring the reference answers against themselves.

What this measures: the *generation* step (does the model stay faithful to the
context it is given?). It does NOT measure retrieval, because the golden
context is supplied directly rather than fetched from the live public APIs.

Usage:
    export GEMINI_API_KEY=...            # never commit this
    python3 eval/generate_responses.py --limit 5        # small trial run
    python3 eval/generate_responses.py                  # all 100 queries
    python3 eval/generate_responses.py --dry-run        # print prompts, no API calls

Ollama (local or remote, no key, no rate limits):
    python3 eval/generate_responses.py --provider ollama --model llama3.1:8b \
        --ollama-url http://<other-mac-ip>:11434 --output eval/data/model_responses_llama3.1-8b.json

The run is resumable: queries already present in the output file are skipped.
Standard library only.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "data", "golden_eval_dataset.json")
DEFAULT_TEMPLATE = os.path.join(HERE, "prompt_template.txt")
DEFAULT_OUTPUT = os.path.join(HERE, "data", "model_responses.json")
# Same default model the app ships with (lib/src/services/gemini_service.dart).
DEFAULT_MODEL = "gemini-3.5-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}


def render_prompt(template: str, context: list, query: str) -> str:
    return template.replace("{{context}}", "\n".join(context)).replace("{{query}}", query)


def call_gemini(api_key: str, model: str, prompt: str, temperature: float, retries: int = 5) -> str:
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )

    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            candidates = payload.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"no candidates returned: {json.dumps(payload)[:300]}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                raise RuntimeError(
                    f"empty response (finishReason={candidates[0].get('finishReason')})"
                )
            return text
        except urllib.error.HTTPError as error:
            if error.code in RETRY_STATUS and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"network error: {error.reason}") from error
    raise RuntimeError("unreachable")


def call_ollama(
    base_url: str, model: str, prompt: str, temperature: float, no_think: bool = False, retries: int = 3
) -> str:
    """Call a (possibly remote) Ollama server: POST /api/generate with stream=false."""
    payload_in = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": 0},
    }
    if no_think:
        payload_in["think"] = False  # reasoning models (qwen3, ...): answer directly
    body = json.dumps(payload_in).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            # Local models can be slow, especially on the first call while the model loads.
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = (payload.get("response") or "").strip()
            # Older servers inline the reasoning; keep only the answer.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if not text:
                raise RuntimeError(f"empty response from Ollama: {json.dumps(payload)[:200]}")
            return text
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"cannot reach Ollama at {base_url}: {error.reason}") from error
    raise RuntimeError("unreachable")


def load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"metadata": {}, "responses": {}}


def save(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    parser.add_argument("--model", default=None, help="Model name. Gemini default: the app's model. Ollama: required (or OLLAMA_MODEL).")
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Base URL of the Ollama server, e.g. http://192.168.1.20:11434",
    )
    parser.add_argument("--no-think", action="store_true", help="Ollama: disable reasoning mode (faster, cleaner answers)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N missing queries")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to wait between calls")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the API")
    args = parser.parse_args()
    if args.provider == "gemini":
        args.model = args.model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    else:
        args.model = args.model or os.environ.get("OLLAMA_MODEL")
        if not args.model:
            print("--model (or OLLAMA_MODEL) is required with --provider ollama, e.g. --model llama3.1:8b", file=sys.stderr)
            return 2

    with open(args.dataset, "r", encoding="utf-8") as handle:
        queries = json.load(handle)["queries"]
    with open(args.template, "r", encoding="utf-8") as handle:
        template = handle.read()

    state = load_existing(args.output)
    responses = state.setdefault("responses", {})
    previous_model = state.get("metadata", {}).get("model")
    previous_provider = state.get("metadata", {}).get("provider", "gemini")
    if responses and previous_model and (previous_model != args.model or previous_provider != args.provider):
        print(
            f"Refusing to continue: '{args.output}' already holds {len(responses)} responses from "
            f"'{previous_model}', but you asked for '{args.model}'. Mixing models in one file would "
            "mislabel the results. Use a different --output file (for example "
            f"eval/data/model_responses_{args.model}.json).",
            file=sys.stderr,
        )
        return 2
    todo = [q for q in queries if q["id"] not in responses]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"{len(queries)} queries in dataset, {len(responses)} already done, {len(todo)} to run.")

    if args.dry_run:
        for item in todo[:3]:
            print("-" * 60)
            print(item["id"])
            print(render_prompt(template, item["ground_truth_context"], item["query"]))
        print("-" * 60)
        print("Dry run: no API calls made.")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if args.provider == "gemini" and not api_key:
        print("GEMINI_API_KEY is not set. Export it in your shell (do not commit it).", file=sys.stderr)
        return 2

    state["metadata"] = {
        "provider": args.provider,
        "no_think": bool(args.no_think),
        "model": args.model,
        "temperature": args.temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "note": "Golden ground-truth context was supplied to the model. Measures generation faithfulness, not retrieval.",
    }

    failures = 0
    for number, item in enumerate(todo, start=1):
        prompt = render_prompt(template, item["ground_truth_context"], item["query"])
        try:
            if args.provider == "ollama":
                text = call_ollama(args.ollama_url, args.model, prompt, args.temperature, args.no_think)
            else:
                text = call_gemini(api_key, args.model, prompt, args.temperature)
        except RuntimeError as error:
            failures += 1
            print(f"[{number}/{len(todo)}] {item['id']} FAILED: {error}", file=sys.stderr)
            continue
        responses[item["id"]] = {"response": text}
        save(args.output, state)
        print(f"[{number}/{len(todo)}] {item['id']} ok")
        time.sleep(args.sleep)

    save(args.output, state)
    print(f"Done. {len(responses)}/{len(queries)} responses stored in {args.output}. Failures: {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
