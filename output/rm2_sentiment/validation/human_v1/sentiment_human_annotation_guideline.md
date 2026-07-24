# Sentiment Human Annotation Guideline

Tujuan anotasi ini adalah memberi reference labels manusia untuk evaluasi sentimen komentar TikTok skincare.
Label manusia tidak boleh diisi oleh pipeline, Codex, heuristic rules, atau model.

Allowed labels: Positive, Neutral, Negative, Uncertain, No Text.

Sentimen dipakai sebagai indikator orientasi pesan, bukan bukti niat, afiliasi, pembayaran, kontrol, pengaruh kausal, buzzer, bot, atau astroturfing.
Gunakan `Uncertain` untuk sarkasme, mixed sentiment yang tidak terselesaikan, target ambigu, atau konteks terlalu pendek.
Gunakan `No Text` hanya jika komentar tidak memiliki informasi yang dapat dievaluasi.
