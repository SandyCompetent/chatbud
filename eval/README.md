# ChatBud Evaluation Suite

A small, honest evaluation harness for ChatBud's grounded answers. It has a **100-query golden dataset** and two modes. Read "What this does and does not measure" before quoting any number from it.

---

## What this does and does not measure

| Question | Answered? |
| :--- | :--- |
| Is the golden dataset well formed and are the reference answers supported by their own context? | Yes, in **dataset check** mode. |
| Given the golden context, does a real Gemini model answer faithfully, with correct numbers? | Yes, in **model responses** mode (needs your own API key). |
| Does ChatBud retrieve the right data from the live public APIs? | **No.** The golden context is supplied directly. |
| Are the scores equivalent to Ragas or DeepEval? | **No.** They are lexical heuristics inspired by those frameworks, not LLM-judged. |

Known limits of the heuristics:
- Faithfulness counts a claim as supported when at least 30% of its words appear in the context, **and** every number in the claim also appears in the context. Without the number rule, answers with every figure changed still scored 0.99; with it they score 0.64 in a test.
- Two rules were added after the first real run on 2026-11-23 (so scores from that run are post-hoc and should be described that way): numbers repeated from the user's question are allowed, and self-introductions ("I am ChatBud AI") and remarks about the context itself ("the provided context does not mention X") are not scored as claims. Thresholds were not changed.
- Known false negatives that remain: a paraphrase with little word overlap (`eval_mem_012`), and list lead-ins that restate the question ("Based on the provided context, the key differences are:") which can be marked unsupported. Both lower faithfulness without the model being wrong.
- **Answer relevance is a weak proxy.** It is 70% word coverage of the *reference answer*, so a correct answer worded differently scores low (for example "Shibboleth Single Sign-On" vs the reference wording). Do not treat its gate as a quality verdict. An LLM-judged relevance score would be a better measure.
- The synonym map in `src/metrics.py` was written by hand for this dataset, which makes relevance scores optimistic on this data and not transferable.
- Subtle factual errors that reuse the right words and numbers are not detected.
- Reference answers scored against their own context are a **self-check**, so they always score near 1.0. That is why dataset-check mode does not report faithfulness or hallucination.

---

## Metrics (model responses mode)

| Metric | Gate | How it is computed |
| :--- | :---: | :--- |
| **Faithfulness** | >= 0.85 | Share of response claims supported by the golden context (word overlap plus number check). |
| **Hallucination rate** | <= 0.08 | `1 - faithfulness`. |
| **Answer relevance** | >= 0.85 | Word coverage of the query (30%) and of the reference answer (70%). |
| **Context precision** | >= 0.80 | Only when the responses file includes `retrieved_context`. Otherwise reported as `not measured` and not gated. |

---

## Running it

### 1. Dataset check (no key, no model)
```bash
python3 eval/run_evals.py
open eval/reports/index.html
```
Output says `Mode: dataset_check`. Passing means the dataset is sound, not that ChatBud is good.

### 2. Score a real model
```bash
export GEMINI_API_KEY=...                       # your own key; never commit it
python3 eval/generate_responses.py --dry-run    # preview prompts, no API calls
python3 eval/generate_responses.py --limit 5    # small trial
python3 eval/generate_responses.py              # all 100 (resumable)
python3 eval/run_evals.py --responses eval/data/model_responses.json
```
- `generate_responses.py` uses the same default model as the app (`gemini-3.5-flash-lite`), temperature 0, and the prompt in `prompt_template.txt`. Override with `--model` or `GEMINI_MODEL`.
- The responses file records the model, time and prompt hash so a published score can be traced to a run.
- `run_evals.py --responses` exits with an error if any golden query has no response. It never falls back to the reference answer.

### 2b. Score a local or remote Ollama model (no key, no rate limits)
```bash
curl http://<host>:11434/api/tags                  # list models on the server
python3 eval/generate_responses.py --provider ollama --model qwen3:14b --no-think \
  --ollama-url http://<host>:11434 --output eval/data/model_responses_qwen3-14b.json
python3 eval/run_evals.py --responses eval/data/model_responses_qwen3-14b.json \
  --output-dir eval/reports/qwen3-14b
```
- The remote server must listen on the network (`OLLAMA_HOST=0.0.0.0 ollama serve`).
- `--no-think` turns off reasoning mode for models such as qwen3, which is faster and keeps `<think>` text out of answers. Use one `--output` file per model; the script refuses to mix models in one file.
- The server address is not written to the responses file or the reports.

### 3. Promptfoo (optional)
```bash
export GOOGLE_API_KEY=...
npx --yes promptfoo eval --config eval/promptfooconfig.yaml
```
Uses a real Gemini provider with `context-faithfulness` and `answer-relevance` assertions graded by the same model. It has not been run as part of this repo's checks.

### 4. Dataset checks inside Flutter tests
```bash
flutter test test/services/rag_eval_suite_test.dart
```
These validate the dataset (schema, unique ids, reference answers related to their context, numbers consistent). They do not test model quality.

---

## Dataset

[`data/golden_eval_dataset.json`](data/golden_eval_dataset.json): 100 queries, 25 per category:
1. **live_public_api**: exchange rates, weather, scholarly papers, CVEs, release lifecycles, lexical lookups, nutrition, holidays.
2. **exeter_academic**: library, campus navigation, student services, exams.
3. **memory_vault**: facts about the app and user preferences.
4. **persona_specialist**: architecture, application security, HR and career coaching, stoic leadership.

Each item: `id`, `category`, `persona_id`, `query`, `ground_truth_context[]`, `reference_answer`, `expected_sources[]`.

---

## Reports

`eval/reports/` holds `eval_report.json`, `eval_report.md` and `index.html`. All values are generated from the run. Unmeasured metrics show `n/a`, never a number. The report header states the mode, and for model runs the model and generation time.

## CI

`.github/workflows/evals.yml` runs the Flutter dataset tests and the **dataset check**. It does not call a model. To gate on real model scores, add `GEMINI_API_KEY` as a repository secret and extend the workflow to run `generate_responses.py` and `run_evals.py --responses`.
