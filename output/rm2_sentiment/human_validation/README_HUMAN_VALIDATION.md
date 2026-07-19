# RM2 Sentiment Human Validation Package

File utama untuk anotator adalah `sentiment_human_annotation_blind.csv`.
File tersebut tidak menyertakan prediksi model, heuristic reference label, confidence model, HCC goal result, atau hcc_id.

Isi `annotator_1_label` dan `annotator_2_label` menggunakan label pada `sentiment_human_annotation_codebook.csv`.
Setelah adjudication, isi `adjudicated_human_label`.

Validator menolak label di luar daftar allowed labels, menghitung agreement, Cohen's kappa, per-class disagreement, confusion matrix, dan adjudication coverage. Selama label manusia kosong, pipeline RM2 sentiment berjalan sebagai exploratory/provisional dan tidak boleh diklaim validated.
