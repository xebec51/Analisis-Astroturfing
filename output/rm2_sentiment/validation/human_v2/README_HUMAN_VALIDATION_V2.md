# Validasi Manusia Sentimen RM2 V2

Direktori ini berisi paket anotasi manusia tahap kedua untuk analisis sentimen RM2 pada level komentar.

Status fase: hanya pembuatan paket anotasi. Tidak ada retraining model, pemilihan threshold, pemilihan ensemble, inference ulang, pembaruan goal HCC, pembaruan actor type, atau perubahan topology Gephi.

Daftar eksklusi V1 saat ini berisi 579 nilai `comment_id` unik dari `output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv`. File V1 dipertahankan dan tidak ditimpa oleh paket ini.

File yang perlu diisi anotator:

- `sentiment_v2_annotator_1_blind.csv`
- `sentiment_v2_annotator_2_blind.csv`
- `sentiment_v2_adjudication_template.csv`

Jangan menambahkan prediksi model, label heuristik, ID HCC, output goal, probability, confidence, atau penanda error ke file blind anotator.

Gunakan `sentiment_human_annotation_v2_guideline.md` dan `sentiment_human_annotation_v2_codebook.csv` saat memberi label.

Setelah anotasi selesai, jalankan:

```powershell
python scripts/validate_rm2_sentiment_human_annotations_v2.py
```

ID locked-test V2 sudah dikunci di `locked_test_v2_manifest.csv`. ID tersebut tidak boleh digunakan untuk model selection, threshold selection, ensemble weighting, preprocessing selection, atau tuning berbasis error.
