# ChatBud Evaluation Report

**Mode**: `model_responses` | **Status**: FAILED (62 / 100 queries) | **Queries**: 100 | **Execution Time**: 0.01s

> Scored real responses from qwen3:14b (generated 2026-11-23T14:27:46+00:00). Metrics are lexical heuristics (token overlap with a hand-written synonym map), not LLM-judged Ragas/DeepEval scores. Golden context is supplied to the model, so retrieval quality is not measured.

## Model response metrics

| Metric | Score | Gate | Result |
| :--- | :---: | :---: | :---: |
| Faithfulness | **0.925** | >= 0.85 | PASS |
| Answer Relevance | **0.760** | >= 0.85 | FAIL |
| Context Precision | **n/a** | >= 0.80 | not measured |
| Hallucination Rate | **0.075** | <= 0.08 | PASS |

## Domain breakdown

| Domain / Category | Queries | Items passing | Faithfulness | Relevance | Context Precision | Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exeter Academic & Campus (Forum, Guild, Navigation) | 25 | 60.0% | 0.960 | 0.725 | n/a | 0.040 |
| Live Public APIs (Frankfurter, Open-Meteo, OpenAlex, OSV) | 25 | 88.0% | 1.000 | 0.820 | n/a | 0.000 |
| Two-Layer Memory Vault (Episodic Facts & Preferences) | 25 | 68.0% | 0.960 | 0.764 | n/a | 0.040 |
| Specialist Personas (Athena, Cipher, Harper, Leo) | 25 | 32.0% | 0.779 | 0.733 | n/a | 0.221 |

> **Limitations**: Metrics are lexical heuristics (token overlap with a hand-written synonym map), not LLM-judged Ragas/DeepEval scores. Golden context is supplied to the model, so retrieval quality is not measured.