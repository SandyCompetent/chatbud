# ChatBud Evaluation Report

**Mode**: `dataset_check` | **Status**: DATASET CHECK: OK (100 / 100). Not a model benchmark | **Queries**: 100 | **Execution Time**: 0.0s

> Dataset check only. No model responses were scored, so this report says nothing about ChatBud's answer quality. Generate responses with eval/generate_responses.py and re-run with --responses to get real metrics.

## Dataset sanity check

| Check | Result | Requirement |
| :--- | :---: | :---: |
| Reference answers supported by their own context | **1.000** | >= 0.85 |
| Dataset problems found | **0** | 0 |

## Domain breakdown

| Domain / Category | Queries | Reference support |
| :--- | :---: | :---: |
| Exeter Academic & Campus (Forum, Guild, Navigation) | 25 | 1.000 |
| Live Public APIs (Frankfurter, Open-Meteo, OpenAlex, OSV) | 25 | 1.000 |
| Two-Layer Memory Vault (Episodic Facts & Preferences) | 25 | 1.000 |
| Specialist Personas (Athena, Cipher, Harper, Leo) | 25 | 1.000 |

> **Limitations**: Metrics are lexical heuristics (token overlap with a hand-written synonym map), not LLM-judged Ragas/DeepEval scores. Golden context is supplied to the model, so retrieval quality is not measured.