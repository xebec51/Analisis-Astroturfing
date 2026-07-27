# Sentiment V2 Positive-Recall Methodology

Status akhir pekerjaan ini:

`V2_POSITIVE_RECALL_CANDIDATE_NOT_ACCEPTED_KEEP_ORIGINAL_V2`

Kandidat V2 Positive-Recall berhasil dibuat sebagai kandidat beku, tetapi belum diterima sebagai model final. Penyebabnya metodologis: target development data manusia belum terpenuhi dan locked test baru masih berupa template pending anotasi/adjudikasi manusia.

## Tujuan

Tujuan perbaikan adalah meningkatkan kemampuan model V2 mengenali komentar Positive tanpa menerapkan aturan "komentar gagal dikenali = Positive". Aturan final kandidat tetap berbasis probabilitas/decision score development-only:

- `Positive` hanya diberikan bila probability Positive memenuhi threshold dan margin yang dipilih dari OOF development.
- Confidence rendah menjadi `Uncertain`.
- Komentar kosong menjadi `No Text`.
- Tidak ada perubahan massal `Neutral`, `Uncertain`, HCC, promosi, atau komentar berpola tertentu menjadi `Positive`.

## Sumber Data Manusia

Registry baru disimpan di:

`output/rm2_sentiment/validation/human_v2_positive_recall/human_label_registry.csv`

Sumber anotasi yang diaudit:

- `human_v1/sentiment_human_annotation_validated.csv`
- `human_v2/sentiment_human_annotation_v2_validated.csv`
- `human_v2/sentiment_v2_adjudication_template_final.csv`
- `human_v2/sentiment_v2_replacement_adjudication_final.csv`
- `human_v2/locked_test_v2_observational_final.csv`
- `human_v3/human_label_registry_v3.csv`

Dua file adjudikasi V2 yang sudah ada dipakai secara eksplisit:

- `sentiment_v2_adjudication_template_final.csv`
- `sentiment_v2_replacement_adjudication_final.csv`

Baris `development_v2` dari adjudication final dapat masuk development. Baris `locked_test_v2`, replacement locked-test, dan historical/final-test lama hanya dicatat sebagai `LEGACY_DIAGNOSTIC_TEST_ALREADY_OPENED`.

## Jumlah Data

Development evaluable setelah audit:

| Label | Count |
|---|---:|
| Negative | 147 |
| Neutral | 286 |
| Positive | 113 |

Target minimal belum terpenuhi:

- Positive target 250-350: belum terpenuhi.
- Negative minimal 250: belum terpenuhi.
- Neutral minimal 400: belum terpenuhi.

Active-learning blind package:

- 500 kandidat anotasi Positive-recall dibuat.
- 0 baris active-learning baru telah teradjudikasi saat kandidat ini dibekukan.

New locked-test template:

- 600 kandidat locked-test dibuat.
- 600 masih `PENDING_HUMAN_ADJUDICATION`.
- Tidak dipakai untuk training, threshold tuning, atau acceptance.

Agreement anotator V2 lama:

- All V2 sentiment raw agreement: 0.9667.
- All V2 Cohen kappa: 0.9467.
- Development V2 raw agreement: 0.9467.
- Development V2 Cohen kappa: 0.9178.

Agreement active-learning baru: pending, karena dua file annotator dan adjudication belum diisi.

## Split Dan Leakage

Split memakai `StratifiedGroupKFold` dengan hard group untuk exact/near-duplicate cluster. `video_id` diaudit sebagai soft grouping karena hard grouping video membuat fold sangat timpang.

Selected fold seed: 42.

Leakage audit:

- Hard duplicate/cv-group leakage: PASS.
- Video cross-fold: WARN, dicatat sebagai soft audit.
- Tidak ada `INJ` dalam development.
- Locked/historical test lama tidak dipakai untuk development selection.

## Kandidat Model

Kandidat yang diuji tetap dalam keluarga V2:

- TF-IDF character n-gram + LinearSVC.
- TF-IDF word n-gram + LinearSVC.
- TF-IDF word-character FeatureUnion + LinearSVC.
- Calibrated LinearSVC sigmoid/isotonic.
- Logistic Regression word-character.
- Ensemble top-3 seed probability averaging.
- V2 frozen baseline sebagai legacy diagnostic only.

Class-weight policies:

- `balanced`
- inverse frequency
- square-root inverse frequency
- manual moderate 1: Negative 1.10, Neutral 0.85, Positive 1.35
- manual moderate 2: Negative 1.00, Neutral 0.90, Positive 1.40

Preprocessing mempertahankan negasi, emoji/emoticon, pemanjangan kata, brand/product name, angka, tanda tanya/seru, dan campuran Indonesia-Inggris.

## Threshold Development-Only

Selected candidate:

- Base: `char_3_5_linearsvc_balanced`
- Ensemble seeds: 42, 52, 62
- Positive threshold: 0.30
- Positive-neutral margin: 0.00
- Positive-negative margin: 0.00
- Abstention threshold: 0.40

Threshold selection status:

`SAFETY_CONSTRAINTS_ONLY`

Artinya threshold memenuhi safety constraints development, tetapi tidak memenuhi seluruh full development constraints terhadap baseline legacy. Kandidat tidak boleh dipromosikan sebelum locked test baru selesai.

Development metrics kandidat ensemble:

| Metric | Value |
|---|---:|
| Accuracy | 0.7629 |
| Macro-F1 | 0.7194 |
| Weighted-F1 | 0.7574 |
| Balanced accuracy | 0.7076 |
| MCC | 0.5985 |
| Coverage | 0.9194 |
| Positive precision | 0.7111 |
| Positive recall | 0.6275 |
| Positive F1 | 0.6667 |
| Neutral recall | 0.8922 |
| Negative recall | 0.6031 |

Hard error counts on development OOF:

- Positive -> Neutral: 23
- Positive -> Negative: 15
- Neutral -> Positive: 9
- Negative -> Positive: 17

Stability across selected seeds:

- std macro-F1: 0.0000
- minimum Positive precision: 0.7111
- minimum Positive recall: 0.6275
- class collapse: none

## Baseline Legacy Diagnostic

V2 frozen baseline on the already-opened legacy locked test:

| Metric | Value |
|---|---:|
| Accuracy covered | 0.8359 |
| Macro-F1 covered | 0.7309 |
| Balanced accuracy | 0.7188 |
| MCC | 0.6369 |
| Coverage | 0.9343 |
| Positive precision | 0.7241 |
| Positive recall | 0.4773 |
| Positive F1 | 0.5753 |

Legacy Positive error summary:

- 47 true Positive.
- 21 predicted Positive.
- 15 predicted Neutral.
- 8 predicted Negative.
- 3 abstain/not covered.

This legacy test is marked `LEGACY_DIAGNOSTIC_TEST_ALREADY_OPENED` and was not used for new candidate selection.

## New Locked Test

Readiness result:

`V2_POSITIVE_RECALL_CANDIDATE_NOT_ACCEPTED_KEEP_ORIGINAL_V2`

Blocked checks:

- 600 rows still pending human adjudication.
- 0 final Positive labels available.
- 0 final Negative labels available.
- Non-evaluable/pending labels remain in `new_locked_test_final.csv`.

Therefore:

- No final locked-test comparison was run.
- No final model promotion was performed.
- No full inference was run.
- `output/rm2_sentiment/final/` remains untouched.

## Acceptance Result

Final status:

`V2_POSITIVE_RECALL_CANDIDATE_NOT_ACCEPTED_KEEP_ORIGINAL_V2`

Acceptance gate failed because the new locked test is not ready and target human development data is insufficient. The original V2 remains the validated/final model. The candidate is stored only as a frozen candidate for later evaluation after new human adjudication is complete.

## Policy Confirmation

No comment that failed recognition was automatically classified as `Positive`.

No blanket conversion was applied to:

- Neutral
- Uncertain
- No Text
- HCC comments
- Promotional comments
- keyword-matched comments
