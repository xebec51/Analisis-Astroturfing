# IndoBERT V4 Artifact Audit

Generated at UTC: 2026-07-28T14:31:06.199958+00:00

This audit verifies the frozen base-reference artifact before any model promotion. It did not train a model, did not tune a threshold, did not rerun the locked test, and did not run full observational inference.

## Summary

| item | value |
| --- | --- |
| Audit commit | 545b565b5afe24153b91769e89dccbb0dc74c8b6 |
| Model directory | artifacts/rm2_sentiment/indobert_v4_final/base_reference |
| Model ID | indobenchmark/indobert-base-p2 |
| Model revision | 94b4e0a82081fa57f227fcc2024d1ea89b57ac1f |
| Model SHA-256 | 2b13c75df848ca0abe02cabf8a1ace0b26814730762a4a7d506095bb518c3c52 |
| Model size bytes | 497798148 |
| Checksum status | True |
| Manifest model hash match | True |
| Label order | Negative, Neutral, Positive |
| CPU load | True |
| CPU inference | True |
| Artifact status | PASS |
| Strict acceptance status | INDOBERT_V4_NOT_ACCEPTED_KEEP_V2 |

## Checksum Audit

| file | passed | sha256 |
| --- | --- | --- |
| config.json | True | 7903656314b55d379cf44d6a2179ee159e5f73f77084df91c2ec89b42f75b96b |
| cpu_inference_smoke_test.json | True | 75423d8f308289eca16c0d1c02be81fd304e213deeed0ed4087f8438b4dc06e1 |
| label_map.json | True | 999a6c5bb68d86de156468e38b523989fd88432bac3d9e897b1571fa26d8a7b5 |
| model.safetensors | True | 2b13c75df848ca0abe02cabf8a1ace0b26814730762a4a7d506095bb518c3c52 |
| remote_artifact_verification.json | True | a4343793a7706d2a363f59709de9fa6f5d33f16d15f1a73984c73c72b3c56a34 |
| selected_trial_config.json | True | f9ca87a32aca5bbfd4ec8b92cbdfadc28f91245f7d1f9f84b1c63cca5c24c856 |
| special_tokens_map.json | True | d2207e01f191626729e08582912c9bf23876883924839b2bbee97489f804e00e |
| tokenizer.json | True | cf77719b0af4f05ab66a4ca11b064be7687f375fd92d026aa1160d187d8192ee |
| tokenizer_config.json | True | 9f0c2c65a70ea18113ffa2e717103e7210ea826e7acc3442b69b346686b55a48 |

## CPU Smoke Test

| input | prediction | confidence | prob_sum |
| --- | --- | --- | --- |
| bagus banget, cocok dan mau beli lagi | Positive | 0.6023 | 1.000000 |
| ini bisa dipakai malam hari? | Neutral | 0.9771 | 1.000000 |
| kurang cocok, bikin kulit terasa perih | Negative | 0.9163 | 1.000000 |
| <empty> | Neutral | 0.6348 | 1.000000 |

All smoke-test predictions use the label order `Negative`, `Neutral`, `Positive`. The empty input check is a robustness check only; final full inference would assign `No Text` before model inference for invalid comment text.
