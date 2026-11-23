# ChatBud Evaluation Report

**Mode**: `model_responses` | **Status**: FAILED (54 / 100 queries) | **Queries**: 100 | **Execution Time**: 0.01s

> Scored real responses from gemini-3.5-flash-lite (generated 2026-11-23T14:02:26+00:00). Metrics are lexical heuristics (token overlap with a hand-written synonym map), not LLM-judged Ragas/DeepEval scores. Golden context is supplied to the model, so retrieval quality is not measured.

## Model response metrics

| Metric | Score | Gate | Result |
| :--- | :---: | :---: | :---: |
| Faithfulness | **0.972** | >= 0.85 | PASS |
| Answer Relevance | **0.735** | >= 0.85 | FAIL |
| Context Precision | **n/a** | >= 0.80 | not measured |
| Hallucination Rate | **0.028** | <= 0.08 | PASS |

## Domain breakdown

| Domain / Category | Queries | Items passing | Faithfulness | Relevance | Context Precision | Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exeter Academic & Campus (Forum, Guild, Navigation) | 25 | 48.0% | 1.000 | 0.688 | n/a | 0.000 |
| Live Public APIs (Frankfurter, Open-Meteo, OpenAlex, OSV) | 25 | 80.0% | 1.000 | 0.795 | n/a | 0.000 |
| Two-Layer Memory Vault (Episodic Facts & Preferences) | 25 | 52.0% | 0.960 | 0.736 | n/a | 0.040 |
| Specialist Personas (Athena, Cipher, Harper, Leo) | 25 | 36.0% | 0.929 | 0.721 | n/a | 0.071 |

> **Limitations**: Metrics are lexical heuristics (token overlap with a hand-written synonym map), not LLM-judged Ragas/DeepEval scores. Golden context is supplied to the model, so retrieval quality is not measured.