# Analisis Pola Koordinasi Komentar TikTok pada Produk Skincare

## Ringkasan Metodologi

Repository ini berisi notebook analisis yang mengidentifikasi **pola koordinasi** pada komentar TikTok
di lima produk skincare, menggunakan pendekatan **Social Network Analysis (SNA)** yang mengacu pada
Weber & Neumann (2021), *Amplifying influence through coordinated behaviour in social networks*.
Pipeline-nya menggabungkan beberapa lapis analisis yang **sengaja dipisahkan** satu sama lain:

- **Latent Coordination Network (LCN)** dibangun dari tiga evidence koordinasi struktural
  (Co-conv, Co-reply, Co-temporal) untuk merepresentasikan hubungan koordinasi antar akun.
- **Louvain community detection** membagi LCN menjadi komunitas.
- **FSA_V (Algorithm 2, Weber & Neumann 2021)** mengekstrak **Highly Coordinating Community (HCC)** --
  komunitas dengan indikasi koordinasi struktural paling kuat -- dari komunitas Louvain.
- **Hashtag metadata video** dipakai untuk **auto-labeling brand** setiap HCC -- murni sebagai
  **asosiasi konteks brand/video**, bukan bukti koordinasi.
- **Co-Similarity (char 5-gram)** dipakai untuk mengukur **kesamaan narasi komentar** antar akun dalam
  HCC yang sama, terinspirasi analisis *consistency of content* Weber & Neumann.
- **Kandidat copypasta/near-copypasta** diidentifikasi sebagai indikasi kesamaan teks -- bukan bukti
  pasti adanya kampanye terkoordinasi.

Seluruh proses -- mulai dari data mentah hingga jaringan siap divisualisasikan di Gephi -- dilakukan
dalam satu notebook: `tiktok_coordination_analysis.ipynb`.

---

## Tujuan Analisis

Notebook ini digunakan untuk:

- Membangun **Latent Coordination Network (LCN)** dari Co-conv, Co-reply, dan Co-temporal.
- Mengekstrak **Highly Coordinating Community (HCC)** menggunakan Louvain + FSA_V.
- Memberi **label brand otomatis** pada tiap HCC berdasarkan hashtag metadata video yang dikomentari
  anggotanya (bukan mapping manual).
- Menganalisis **kesamaan narasi komentar** (Co-Similarity char 5-gram) antar akun dalam HCC, termasuk
  mendeteksi kandidat copypasta/near-copypasta.
- Mengekspor jaringan (LCN & HCC) ke format siap-pakai di **Gephi**.

**Notebook ini TIDAK digunakan untuk:**

- Membuktikan adanya *astroturfing*.
- Mengidentifikasi akun bot.
- Menyimpulkan niat atau motif manipulatif di balik suatu pola komentar.
- Menyatakan bahwa akun menuliskan hashtag tertentu di kolom komentarnya (hashtag bersumber dari
  metadata video, bukan teks komentar).

Seluruh hasil dalam notebook ini **hanya berupa indikasi** -- pola koordinasi struktural (HCC), konteks
brand/video (hashtag), dan kesamaan narasi (Co-Similarity) adalah tiga lapis bukti yang terpisah dan
harus dibaca bersama-sama, bukan sebagai satu kesimpulan tunggal (lihat [Catatan Interpretasi](#catatan-interpretasi)).

---

## Batasan Analisis

- Analisis ini bersifat **deskriptif-eksploratif** -- mengidentifikasi pola, bukan membuktikan niat.
- Pola koordinasi yang terdeteksi dapat bersifat organik (pengguna aktif, penggemar produk, komunitas
  diskusi wajar), bukan otomatis berarti terkoordinasi secara sengaja.
- Kandidat copypasta/near-copypasta adalah indikasi kesamaan teks, bukan bukti pasti kampanye.
- Notebook tidak mengklasifikasikan akun sebagai bot, buzzer, atau pelaku astroturfing.

---

## Pipeline Terbaru

```
Dataset komentar + metadata video
        ↓
Preprocessing (deduplikasi, normalisasi akun, parsing waktu)
        ↓
Evidence Construction: Co-conv, Co-reply, Co-temporal
        ↓
Evidence Filtering & Normalization (Min-Max / Max Normalization)
        ↓
Latent Coordination Network (LCN)
        ↓
Louvain Community Detection
        ↓
FSA_V / HCC Extraction (Algorithm 2, Weber & Neumann 2021)
        ↓
Hashtag Metadata Video → Asosiasi Konteks Brand/Video
        ↓
Auto Brand Labeling per HCC
        ↓
Co-Similarity (char 5-gram) → Kesamaan Narasi Komentar
        ↓
Copypasta Candidate Detection
        ↓
Gephi Export (LCN & HCC, ringkas untuk visualisasi)
```

Beberapa hal penting terkait alur di atas:

- **Evidence Filtering dilakukan sebelum normalisasi.** Pasangan akun hanya dipertahankan apabila
  memenuhi *minimal salah satu* ambang batas evidence (Co-conv/Co-reply/Co-temporal).
- **LCN memuat seluruh edge hasil Evidence Filtering** (dengan threshold persentil Final Weight sebagai
  pra-filter praktis) -- `final_weight` disimpan sebagai atribut (bobot Louvain/FSA_V, ketebalan garis di
  Gephi), **bukan** sebagai filter tambahan pada LCN itu sendiri.
- **HCC ditentukan lewat FSA_V (Algorithm 2)** -- kandidat tumbuh dari edge terberat tiap komunitas
  Louvain selama Mean Edge Weight tidak turun di bawah Global Mean Edge Weight; **tidak ada** ranking
  atau parameter "ambil N komunitas teratas".
- **Hashtag & Co-Similarity tidak membentuk LCN.** Keduanya murni dihitung **setelah** HCC final
  terbentuk, untuk karakterisasi/interpretasi -- tidak pernah mengubah struktur LCN, Louvain, atau FSA_V.
- **Auto brand labeling murni data-driven**: label brand HCC dihitung dari skor hashtag brand pada
  metadata video, tanpa mapping manual dari ketua tim maupun hardcode `community_id`.

---

## Penjelasan Evidence

| Evidence | Definisi | Peran |
|---|---|---|
| **Co-conv** | Pasangan akun muncul bersama pada video yang sama, berulang di beberapa video berbeda | Pembentuk LCN |
| **Co-reply** | Satu akun membalas komentar akun lain secara berulang di beberapa video berbeda | Pembentuk LCN |
| **Co-temporal** | Dua akun berkomentar dalam rentang waktu berdekatan, berulang di beberapa video berbeda | Pembentuk LCN |
| **Co-hashtag** | Hashtag pada metadata video yang dikomentari anggota HCC | **Bukan** pembentuk LCN -- hanya asosiasi konteks brand/video |
| **Co-Similarity** | Kesamaan narasi komentar antar akun dalam HCC (TF-IDF char 5-gram + cosine similarity) | **Bukan** pembentuk LCN -- hanya analisis kesamaan narasi setelah HCC terbentuk |
| **Co-mention** | Akun/kreator yang disebut pada metadata video | **Deprecated** -- tidak lagi dihitung ulang atau dipakai dalam interpretasi final (tidak informatif pada dataset TikTok ini) |

---

## Struktur Output

### `output/tables/` — lengkap untuk analisis dan pelaporan

| File | Isi |
|---|---|
| `co_conv_edges.csv`, `co_reply_edges.csv`, `co_temporal_edges.csv` | Evidence mentah per pasangan akun, setelah filtering |
| `focal_structures.csv` | Hasil FSA_V/HCC: Density, Average Degree, Average Weighted Degree, Candidate MEW, status HCC |
| `hcc_hashtag_profile.csv`, `hcc_hashtag_all_communities_complete.csv`, `hcc_hashtag_community_summary.csv`, `hcc_hashtag_matrix_frequency.csv`, `hcc_hashtag_matrix_users_exposed.csv` | Profil hashtag metadata video (konteks brand/video) per HCC |
| `hcc_brand_profile_auto.csv` | Skor & label brand otomatis per HCC (`BRAND_HASHTAG_LEXICON`, `primary_brand`, `brand_label_auto`, `brand_combo`, `brand_confidence`, skor per brand, dll.) |
| `hcc_cosimilarity_summary.csv` | Ringkasan kesamaan narasi per HCC (avg/median/max cosine, `narrative_similarity_level`, dll.) |
| `hcc_cosimilarity_pairs.csv` | Cosine similarity per pasangan akun dalam HCC yang sama |
| `hcc_copypasta_candidates.csv` | Kandidat exact/near-copypasta per HCC |
| `hcc_mention_profile.csv` | **Deprecated** -- artefak lama, tidak diregenerasi lagi |

### `output/gephi/` — ringkas untuk visualisasi

Prinsip: **Gephi export = ringkas untuk visualisasi, `output/tables/` = lengkap untuk analisis.** Detail
skor (brand, similarity) sengaja tidak dimasukkan ke file Gephi -- lihat tabel di atas.

**`gephi_lcn_nodes.csv`** -- node Full LCN untuk Gephi:

| Kolom | Fungsi |
|---|---|
| `id` | Identitas akun (username), dipakai sebagai Node ID di Gephi |
| `label` | Label tampilan node (sama dengan `id`) |
| `degree` | Jumlah tetangga langsung dalam LCN |
| `weighted_degree` | Total `final_weight` seluruh edge yang terhubung ke node |
| `betweenness` | Betweenness centrality -- peran sebagai penghubung (*bridge*) antar komunitas |
| `community` | ID komunitas hasil Louvain (seluruh LCN, bukan hanya HCC) |
| `brand_label_auto` | Label brand otomatis (Section 16.1.3); `"Non-HCC"` jika node bukan anggota HCC manapun |

**`gephi_lcn_edges.csv`** -- edge Full LCN untuk Gephi:

| Kolom | Fungsi |
|---|---|
| `source`, `target` | Pasangan akun yang membentuk edge |
| `co_conv_weight`, `co_reply_weight`, `co_temporal_weight` | Nilai mentah tiap evidence |
| `norm_co_conv`, `norm_co_reply`, `norm_co_temporal` | Hasil normalisasi (Max Normalization) tiap evidence |
| `final_weight` | Bobot gabungan (dasar Louvain & FSA_V; ketebalan garis di Gephi) |
| `n_evidence` | Jumlah evidence independen (>0) yang mendukung edge ini |
| `co_hashtag` | Jumlah hashtag video yang sama antara kedua akun (asosiasi konteks brand/video) |

**`gephi_hcc_nodes.csv`** -- node HCC untuk Gephi (subset LCN hasil FSA_V), termasuk `brand_label_auto`:

| Kolom | Fungsi |
|---|---|
| `id`, `label`, `degree`, `weighted_degree`, `betweenness`, `community` | Sama seperti `gephi_lcn_nodes.csv`, khusus subset anggota HCC |
| `primary_brand` | Brand dengan skor hashtag tertinggi pada HCC tsb |
| `brand_label_auto` | Label brand final HCC: nama brand tunggal / `Mixed_2_Brands` / `Mixed_3plus_Brands` / `Not identified` -- dipakai untuk pewarnaan node di Gephi |
| `brand_combo` | Gabungan brand aktif jika lebih dari satu (mis. `"Maryame + The Originote"`) |
| `brand_confidence` | Tingkat keyakinan label brand: `High` / `Medium` / `Low` / `None` |
| `narrative_similarity_level` | Level kesamaan narasi komentar dalam HCC: `High` / `Medium` / `Low` / `Insufficient text` |

**`gephi_hcc_edges.csv`** -- edge HCC untuk Gephi, termasuk `co_similarity_char5`:

| Kolom | Fungsi |
|---|---|
| `source`, `target` | Pasangan akun anggota HCC |
| `weight` | `final_weight` (bobot LCN) |
| `n_evidence` | Jumlah evidence independen (>0) |
| `co_conv`, `co_reply`, `co_temporal` | Nilai mentah tiap evidence |
| `co_hashtag` | Kesamaan hashtag video antara kedua akun |
| `co_similarity_char5` | Cosine similarity narasi komentar (TF-IDF char 5-gram, ala Weber & Neumann); `NaN` jika salah satu akun tidak punya teks bersih yang cukup |

### `output/visualisasi/`

Seluruh grafik ditampilkan inline di notebook; sebagian juga disimpan sebagai file PNG:

- `output/visualisasi/HCC_hashtag_*`, `output/visualisasi/hashtag_by_community/` -- visualisasi hashtag
  (konteks brand/video) per HCC, format **PNG saja**.
- `output/visualisasi/brand_similarity/` -- visualisasi Co-Similarity ala Weber & Neumann (matrix,
  bar/boxplot per HCC & per brand) dan kandidat copypasta, format **PNG saja**.

---

## Panduan Gephi

1. **Import**: `File → Import Spreadsheet` -- pilih `gephi_lcn_edges.csv`/`gephi_hcc_edges.csv` sebagai
   *Edge Table*, lalu `gephi_lcn_nodes.csv`/`gephi_hcc_nodes.csv` sebagai *Node Table*.
2. **Layout**: `ForceAtlas2` (LinLog Mode, weight = `final_weight`) atau `Fruchterman Reingold`.
3. **Node size**: mapped dari `weighted_degree`.
4. **Node color -- struktural**: Appearance → Nodes → Partition → `community`.
5. **Node color -- brand**: Appearance → Nodes → Partition → `brand_label_auto`. Warna **tidak**
   diekspor sebagai kolom CSV (tidak ada `brand_color_hex`) -- tentukan warna langsung di panel Partition
   Gephi. Rekomendasi warna (konsisten dengan visualisasi di notebook, Section 16.1.3/16.5):

   | brand_label_auto | Warna rekomendasi |
   |---|---|
   | Azarine | Hijau |
   | Daviena | Pink |
   | Maryame | Kuning |
   | The Originote | Biru |
   | Mixed_2_Brands | Abu-abu sedang |
   | Mixed_3plus_Brands | Abu-abu gelap |
   | Not identified | Abu-abu muda |
   | Non-HCC (khusus `gephi_lcn_nodes.csv`) | Abu-abu sangat muda |

6. **Edge thickness**: mapped dari `final_weight` (LCN) / `weight` (HCC).
7. **Filter tambahan (opsional)**: `co_hashtag` (LCN & HCC), `co_similarity_char5` (khusus HCC).
8. **Jangan** menjalankan modularity ulang di Gephi untuk mendapatkan pengelompokan brand -- struktur
   komunitas (Louvain/FSA_V) maupun label brand sudah final dan dihitung otomatis di notebook.

---

## Catatan Interpretasi

Pipeline ini memisahkan tiga lapisan analisis yang **tidak boleh disatukan begitu saja**:

1. **LCN/FSA_V** = koordinasi struktural berbasis perilaku (Co-conv/Co-reply/Co-temporal).
2. **Hashtag metadata video** = asosiasi konteks brand/video -- brand label menunjukkan konteks video
   yang dikomentari oleh anggota HCC berdasarkan hashtag metadata video, **bukan** berarti akun tersebut
   menulis hashtag itu dalam komentarnya.
3. **Co-Similarity** = kesamaan narasi komentar antar akun dalam HCC, **bukan** bukti pasti koordinasi
   terencana -- termasuk kandidat copypasta/near-copypasta, yang tetap disebut sebagai *kandidat*, bukan
   bukti kampanye pasti.

Dengan demikian, **HCC saja tidak langsung disebut sebagai bukti astroturfing**. HCC menunjukkan
indikasi koordinasi struktural; brand label menunjukkan konteks video; Co-Similarity menunjukkan apakah
akun-akun dalam HCC juga menyampaikan narasi komentar yang mirip. Ketiganya perlu dibaca bersama untuk
interpretasi yang proporsional.

**Co-Mention** dihapus dari analisis aktif karena tidak informatif pada dataset TikTok ini dan tidak
menjadi dasar interpretasi utama (lihat Section 16.2 & 19.11 pada notebook).

---

## Cara Menjalankan Notebook

1. **Install dependency** (Python 3.9+): `pandas`, `numpy`, `networkx`, `scikit-learn`, `matplotlib`,
   `seaborn`, `tqdm`, serta `python-louvain` (opsional -- jika tidak tersedia, notebook otomatis
   menggunakan algoritma Louvain bawaan NetworkX).
2. **Buka Jupyter Notebook/Lab** (atau Google Colab), lalu buka `tiktok_coordination_analysis.ipynb`.
3. **Siapkan data** -- pastikan `dataset.csv` dan `video_metadata_clean.csv` berada satu folder dengan
   notebook.
4. **Jalankan dari atas ke bawah** (*Run All*). Setiap tahap pipeline bergantung pada tahap sebelumnya,
   sehingga urutan sel tidak boleh diubah.

---

## Struktur Repository

| File / Folder | Keterangan |
|---|---|
| `dataset.csv` | Dataset mentah komentar TikTok dari 5 kategori produk skincare |
| `video_metadata_clean.csv` | Metadata video (hashtag & mention resmi per video) |
| `tiktok_coordination_analysis.ipynb` | Notebook utama -- seluruh pipeline analisis |
| `output/tables/` | Tabel hasil analisis lengkap (lihat [Struktur Output](#struktur-output)) |
| `output/gephi/` | Berkas jaringan siap-impor ke Gephi, ringkas untuk visualisasi |
| `output/visualisasi/` | Grafik & visualisasi (PNG saja; ekspor PDF/SVG duplikat tidak dipertahankan) |
| `README.md` | Berkas dokumentasi ini |

Seluruh hasil analisis **hanya berupa indikasi pola koordinasi** berdasarkan data yang teramati. Pola
koordinasi yang terdeteksi dapat pula bersifat organik, misalnya berasal dari pengguna aktif, penggemar
produk, atau komunitas diskusi yang wajar.

---

## RM1 ? Temporal Activity Profile

Section 19 pada notebook `tiktok_coordination_analysis.ipynb` berisi analisis temporal ringkas untuk RM1.
Fokusnya menampilkan grafik garis pola aktivitas komentar berdasarkan tiga cakupan:

- seluruh dataset;
- akun yang masuk LCN;
- akun yang masuk HCC.

Visual temporal ditampilkan menurut hari dalam pekan, jam lokal, serta kombinasi hari x jam. Zona waktu
utama output temporal adalah `Asia/Jakarta` / WIB. Analisis brand-context temporal, sentiment/goals
temporal, recurrence mingguan rinci, dan ekspor Gephi temporal tidak dimasukkan ke bagian inti RM1 agar
kode dan output tetap proporsional.

Output tabel ringkas disimpan di `output/rm1_temporal/tables/`:

- `temporal_data_quality_audit.csv`
- `temporal_method_parameters.csv`
- `temporal_scope_weekday_summary.csv`
- `temporal_scope_hour_summary.csv`
- `temporal_scope_weekday_hour_summary.csv`
- `hcc_weekday_summary.csv`
- `hcc_weekday_hour_matrix.csv`
- `hcc_temporal_profile.csv`
- `temporal_final_summary.csv`

Visualisasi PNG ringkas disimpan di `output/rm1_temporal/visualisasi/`:

- `temporal_scope_weekday_lines.png`
- `temporal_scope_hour_lines.png`
- `temporal_scope_weekday_hour_lines.png`
- `active_hcc_by_weekday_line.png`

Community Actor merupakan akun anggota HCC, bukan akun yang telah terbukti sebagai buzzer. Aktivitas yang
terkonsentrasi pada hari atau jam tertentu tidak membuktikan jadwal kerja, pembayaran, hubungan komersial,
atau koordinasi terencana.

## RM2 — Sentiment-based Goals Mapping

Rumusan Masalah 2 (RM2) menjawab pertanyaan *"Bagaimana three dimensions tipologi aktor digital
astroturfing pada produk skincare overclaim di platform TikTok?"* — terdiri dari tiga dimensi: **target**
dan **actor type** (dipetakan dengan SNA pada RM1, `tiktok_coordination_analysis.ipynb`), serta **goals**
(dipetakan dengan analisis sentimen pada notebook `02_rm2_sentiment_goals.ipynb`, terpisah dari RM1).

**`02_rm2_sentiment_goals.ipynb` hanya membaca** output RM1 sebagai input dan **tidak pernah** mengubah
notebook RM1, pipeline LCN/Louvain/FSA_V, hasil HCC, maupun file output RM1 yang sudah ada.

### Tujuan

1. Melakukan analisis sentimen komentar TikTok sebagai indikator **orientasi pesan** (Positive / Neutral /
   Negative), dengan status terpisah untuk `Uncertain` dan `No Text`.
2. Mengagregasi sentimen pada tiga level: **komentar**, **akun**, **HCC**.
3. Memetakan dimensi **`goals`** melalui `goal_orientation` berbasis distribusi sentimen teramati.
4. Menghubungkan hasil sentimen dengan hasil RM1 (`HCC`, `community`, `brand_label_auto`, `primary_brand`)
   tanpa mengubah LCN, Louvain, FSA_V, keanggotaan HCC, atau brand labeling RM1.

### Input

| File | Sumber | Peran |
|---|---|---|
| `dataset.csv` | Data mentah | Teks komentar untuk analisis sentimen |
| `output/gephi/gephi_hcc_nodes.csv` | RM1 (hanya dibaca) | Keanggotaan HCC (`id`=username), brand & atribut struktural |
| `output/tables/hcc_brand_profile_auto.csv` | RM1 (hanya dibaca) | Profil brand per HCC |
| `output/tables/focal_structures.csv` | RM1 (hanya dibaca, opsional) | Metadata FSA_V/HCC |
| `output/gephi/gephi_hcc_edges.csv` | RM1 (hanya dibaca, opsional) | Edge HCC untuk export Gephi RM2 |

### Model Sentimen

Pipeline berjalan bertahap. Model development dapat dibekukan setelah label manusia development tersedia,
tetapi **final locked-test evaluation** hanya boleh dilakukan satu kali setelah locked test V2 kembali
lengkap 300 komentar observasional. Sampai tahap itu selesai, status model adalah
`DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST`, bukan `FINAL_MODEL_VALIDATED`.

Model development V2 yang dibandingkan mencakup baseline V1 lama sebagai pembanding diagnostik, TF-IDF
word n-gram + Logistic Regression, TF-IDF word n-gram + LinearSVC, TF-IDF character n-gram + Logistic
Regression, TF-IDF character n-gram + LinearSVC, gabungan word + character features, calibrated LinearSVC,
dan ensemble development jika OOF development meningkat. Model human-supervised dibekukan dengan nama
`model_C_human_supervised`.

Label mapping dibaca dari konfigurasi model dan diuji dengan anchor sentences positif, netral, dan negatif.
Probabilitas tiga kelas disimpan untuk setiap komentar, sedangkan komentar `Uncertain` dan `No Text` tidak
diam-diam dipaksa menjadi Neutral.

### Validasi Domain

V1 human validation yang sudah dibersihkan berisi `579` comment_id unik. Paket V2 memiliki hasil anotasi
manusia lengkap pada file final, tetapi file kanonis observasional
`output/rm2_sentiment/human_validation_v2/sentiment_human_annotation_v2_validated.csv` berisi `592` baris:
`300` development V2 dan `292` locked-test V2 observasional. Delapan ID locked-test synthetic/injected
masih menahan evaluasi final sampai replacement dianotasi manusia.

Development training pool V2 memakai label manusia observasional yang sah dari V1 development, V1 historical
test, dan V2 development, setelah mengeluarkan seluruh synthetic/injected IDs dan seluruh 292 locked-test V2
observasional. Pool final: `806` komentar unik (`Negative=216`, `Neutral=433`, `Positive=157`; sumber
training: `V1=518`, `V2=288`).

Model development yang dibekukan saat ini adalah ensemble top-2 human-supervised:
`ensemble_top2_human_supervised_development_only`, terdiri dari
`tfidf_char_linearsvc_social_C1_balanced` dan
`calibrated_linearsvc_word_char_social_C1_balanced`. Threshold development dipilih `0.46` berdasarkan OOF
development saja. Pada OOF development, ensemble memiliki mean macro-F1 `0.6916`, accuracy `0.7407`,
balanced accuracy `0.6750`, MCC `0.5560`, ECE `0.1306`, dan Brier score `0.1278`. Pada threshold `0.46`,
coverage `0.8251`, abstention `0.1749`, macro-F1 covered `0.7495`, dan bootstrap 95% CI macro-F1 covered
`0.7084-0.7885`. Angka tersebut adalah **development diagnostics**, bukan locked-test performance.

Locked-test V2 belum dievaluasi, tidak dibuat prediksi per baris locked-test, dan full inference 33.847
komentar belum dijalankan. Karena itu `comment_sentiment.csv`, agregasi Goals, dan atribut sentimen Actor
Type lama belum di-refresh oleh model V2. Goal counts lama tetap dibaca sebagai hasil sementara; confidence
dan stability bukan akurasi.

### Output Tabel (`output/rm2_sentiment/tables/`)

| File | Level | Isi |
|---|---|---|
| `comment_sentiment.csv` | Komentar | `sentiment_label_final`, probabilitas kelas, confidence, uncertainty, HCC/brand per komentar |
| `account_sentiment_summary.csv` | Akun | Agregasi sentimen per akun + atribut struktural HCC |
| `hcc_sentiment_goals_summary.csv` | HCC | Agregasi sentimen per HCC + `goal_orientation`, `goal_confidence` |
| `hcc_vs_nonhcc_sentiment_summary.csv` | Grup | Perbandingan distribusi sentimen HCC vs Non-HCC |
| `brand_sentiment_summary.csv` | Brand | Agregasi sentimen & `dominant_goal_orientation` per `brand_label_auto` |
| `sentiment_development_heuristic_reference.csv` | Diagnostic | Development/challenge set dengan heuristic pseudo-label |
| `sentiment_holdout_heuristic_reference.csv` | Diagnostic | Locked-test sample dengan heuristic pseudo-label |
| `sentiment_deterministic_rule_reproducibility.csv` | Diagnostic | Reproducibility aturan deterministik, bukan reliability anotasi |
| `sentiment_development_human_reference.csv` | Human validation | Development/challenge set dengan adjudicated human labels |
| `sentiment_locked_test_human_reference.csv` | Human validation | Locked-test set dengan adjudicated human labels |
| `sentiment_model_selection.csv` | Model | Pipeline final, preprocessing, threshold, dan revision model |
| `sentiment_model_locked_test_metrics.csv` | Model | Metrik locked-test human-reference dan bootstrap CI |
| `sentiment_repeated_cv_summary.csv` | Model | Mean/std repeated stratified CV berbasis adjudicated human labels |
| `neutral_error_taxonomy.csv` | Model | Audit khusus kegagalan kelas Neutral |
| `hcc_goal_heuristic_review.csv` | Goals | Review heuristic seluruh 42 HCC pada level goal, diagnostic only |
| `hcc_goal_validation_metrics.csv` | Goals | Diagnostic agreement; nilai rendah menahan status validated |
| `hcc_goal_method_sensitivity.csv` | Goals | Sensitivity hard-label, soft probability, confidence-weighted, dan smoothed shares |
| `sentiment_final_validation_report.csv` | Audit | Gate status `PASS`/`WARNING`/`FAIL`/`NOT_AVAILABLE` dan overall status |

### Output Visualisasi (`output/rm2_sentiment/visualisasi/`, PNG saja)

Visualisasi utama dibatasi menjadi tiga file:

- `sentiment_validation_confusion_matrix.png`
- `sentiment_hcc_vs_nonhcc_100pct.png`
- `hcc_goal_orientation_confidence.png`

WordCloud dan grafik eksploratif lama tidak dipertahankan sebagai output utama karena tidak berfungsi sebagai
validasi sentimen. WordCloud eksploratif untuk paparan temuan tersedia terpisah di
`output/rm2_sentiment/visualisasi_exploratory/` dan tidak digunakan sebagai evidence validasi model.

### Output Gephi RM2 (`output/rm2_sentiment/gephi/`)

File terpisah dari Gephi RM1 (tidak menimpa): `gephi_hcc_nodes_sentiment.csv` (node HCC RM1 + atribut
sentimen/`goal_orientation`) dan `gephi_hcc_edges_sentiment.csv` (edge HCC RM1, disalin apa adanya —
sentimen hanya atribut node, edge tidak diubah). Dipakai untuk mewarnai node di Gephi berdasarkan
`dominant_sentiment` atau `goal_orientation`, disandingkan dengan `brand_label_auto` dari RM1.

## RM2 — Three Actor Types and Three Dimensions Typology

Notebook `03_rm2_actor_type_typology.ipynb` menambahkan tipologi tiga actor type dan mengintegrasikannya
dengan dimensi target serta goals yang sudah dihitung pada notebook sentimen RM2. Tiga actor type utama
yang dipakai hanya `Individual Actor`, `Community Actor`, dan `Mass Actor`.

- Actor universe merupakan gabungan akun komentator, seluruh creator pada `video_metadata_clean.csv`, dan
  actor tambahan dari `config/individual_actor_registry.csv` yang berstatus `Verified`; creator tanpa komentar
  tetap dimasukkan sebagai `Individual Actor`.
- `Individual Actor` otomatis mencakup seluruh akun pembuat video pada metadata. Subtype seperti influencer,
  akun resmi, owner/representative, atau expert tetap memerlukan registry `Verified`.
- `Community Actor` adalah akun anggota HCC final dari RM1 (`output/gephi/gephi_hcc_nodes.csv`) selama akun
  tersebut bukan `Individual Actor`.
- `Mass Actor` adalah komentator non-individual dan non-HCC. Keanggotaan LCN untuk Mass Actor hanya atribut
  posisi sekunder (`LCN Non-HCC` atau `Outside LCN`), bukan actor type utama.
- Community-Mass association aggregate dibaca melalui `Direct Interaction`, `Temporal Association`,
  `Shared-Video Context Only`, atau `No Observed Community Association`. Shared-video hanya konteks bersama,
  direct interaction harus lolos validasi parent comment, dan `temporal_mass_comment_count` dipisahkan dari
  `temporal_hcc_comment_pair_count`. Untuk layer account-level Community-Mass yang baru, edge dibentuk dari
  Co-conv, Co-reply, dan Co-temporal, bukan dari syarat reply langsung saja.
- Asosiasi HCC ambigu tidak dipaksa menjadi satu primary HCC; aktor ambigu dipertahankan sebagai many-to-many
  dan dihitung dengan fractional context weight pada ringkasan HCC.
- Dimensi `goals` memiliki account-level goals dan pooled actor-type goals; model sentimen tidak dijalankan
  ulang karena memakai output RM2 sentiment yang sudah tersedia.
- Sentiment alignment selalu dilaporkan bersama denominator dan coverage, serta hanya menunjukkan keselarasan
  sentimen yang teramati.
- Relasi Individual-Community berasal dari komentar HCC pada video yang dibuat Individual Actor.
- Dimensi `target` dipetakan sebagai konteks brand/video atau kategori produk, bukan korban atau sasaran
  serangan.
- Seluruh hubungan merupakan observed association, bukan bukti pengaruh kausal, perubahan opini, niat
  manipulatif, ataupun astroturfing.
- Output tabel disimpan di `output/rm2_actor_type/tables/`, visualisasi PNG di
  `output/rm2_actor_type/visualisasi/`, dan file Gephi baru di `output/rm2_actor_type/gephi/`.

## RM2 — Actor Type Visualization in Gephi

Visualisasi Type Actor untuk RM2 membedakan **actor type** dari **posisi jaringan**:

- `actor_type_primary`: `Individual Actor`, `Community Actor`, atau `Mass Actor`.
- `network_position`: `HCC`, `LCN Non-HCC`, atau `Outside LCN`.

Definisi utama:

- `Individual Actor`: akun creator video atau registry individual yang terverifikasi.
- `Community Actor`: akun anggota HCC final yang bukan Individual Actor.
- `Mass Actor`: residual seluruh actor universe setelah Individual Actor dan Community Actor dikeluarkan.

LCN bukan actor type. Mass Actor juga bukan hanya akun LCN: Mass Actor mencakup aktor residual yang berada
di `LCN Non-HCC` maupun `Outside LCN`. Visualisasi Gephi hanya menampilkan subset akun yang masuk LCN,
karena akun `Outside LCN` tidak mempunyai edge koordinasi yang memenuhi kriteria pembentukan jaringan.
Mass Actor di luar LCN tetap dihitung dalam statistik actor universe.

Untuk ekspor aggregate actor type, `Non-HCC` diperlakukan sebagai kategori residual, bukan ID HCC valid.
Node atau edge seperti `HCC_Non-HCC` dan `MASS_HCC_Non_HCC_*` adalah artefak konstruksi aggregate dan
dihapus di dalam notebook sebelum file Gephi diekspor. Notebook menyimpan audit pembersihan sehingga
proses dapat dijalankan ulang tanpa pembersihan manual.

Bobot mentah edge tetap disimpan untuk analisis sebagai `edge_weight_raw`. File khusus Gephi memakai
`Weight = log1p(edge_weight_raw)` agar edge `Ambiguous` dengan nilai fraksional besar tidak mendominasi
ketebalan visual. Edge `Ambiguous` menunjukkan alokasi fraksional, bukan hubungan langsung yang pasti;
ketebalan edge bukan bukti pengaruh, kendali, atau intensi koordinasi.

Output utama:

- `output/rm2_actor_type/tables/actor_type_universe_summary.csv`
- `output/rm2_actor_type/tables/actor_type_edges_analysis.csv`
- `output/rm2_actor_type/tables/lcn_actor_type_summary.csv`
- `output/rm2_actor_type/tables/actor_type_network_position_matrix.csv`
- `output/rm2_actor_type/tables/lcn_actor_type_edge_summary.csv`
- `output/rm2_actor_type/tables/gephi_lcn_edge_integrity_audit.csv`
- `output/rm2_actor_type/tables/actor_type_gephi_validation_report.csv`
- `output/rm2_actor_type/audit/actor_type_gephi_cleaning_audit.csv`
- `output/rm2_actor_type/gephi/gephi_actor_type_nodes.csv`
- `output/rm2_actor_type/gephi/gephi_actor_type_edges.csv`
- `output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv`
- `output/rm2_actor_type/gephi/gephi_lcn_edges_actor_type.csv`

### Community-Mass Account Evidence Network

Layer ini membangun **observed Community-Mass interaction/coordination relation** pada level akun:

- satu node = satu akun;
- satu edge = satu pasangan akun `Community Actor` dan `Mass Actor`;
- edge dibentuk dari gabungan evidence RM1: `co_conv_weight`, `co_reply_weight`, dan
  `co_temporal_weight`;
- `parent_comment_id` hanya relevan untuk evidence Co-reply, bukan syarat utama edge;
- sentiment/goal tidak dipakai untuk membentuk edge.

Layer ini berbeda dari graf aggregate 396 node/497 edge dan berbeda dari HCC-Mass segment association.
Mass Actor tetap berasal dari seluruh residual actor universe; Mass Actor `Outside LCN` dapat muncul pada
pasangan pre-LCN, tetapi tidak pernah dipromosikan menjadi edge LCN final. Direct-reply output lama tetap
dipertahankan sebagai diagnostic tambahan dengan status
`analysis_scope = OPTIONAL_DIRECT_REPLY_DIAGNOSTIC`.

Ringkasan output saat ini:

- total pasangan account-level Community-Mass: `434823`;
- pasangan yang merupakan edge LCN final: `305`;
- pasangan pre-LCN multi-evidence: `2667`;
- pasangan pre-LCN single-evidence: `431851`;
- Mass Actor `Outside LCN` yang memiliki evidence: `25660`;
- status atribut sentimen sementara: `DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST`;
- final locked-test evaluation: `BLOCKED_WAITING_FOR_8_HUMAN_ANNOTATED_REPLACEMENTS`.

File utama:

- `output/rm2_actor_type/account_interaction/community_mass_account_pairs.csv`
- `output/rm2_actor_type/account_interaction/community_mass_account_summary.csv`
- `output/rm2_actor_type/account_interaction/community_mass_by_network_position.csv`
- `output/rm2_actor_type/account_interaction/community_mass_by_interaction_scope.csv`
- `output/rm2_actor_type/account_interaction/community_mass_by_evidence_combination.csv`
- `output/rm2_actor_type/account_interaction/community_mass_by_hcc.csv`
- `output/rm2_actor_type/account_interaction/community_mass_integrity_report.csv`
- `output/tables/pre_filter_combined_evidence_edges.csv`

File Gephi account-level:

- Full analytical: `output/rm2_actor_type/gephi/gephi_community_mass_account_nodes.csv` dan
  `output/rm2_actor_type/gephi/gephi_community_mass_account_edges_all_evidence.csv`
- Recommended visual: `output/rm2_actor_type/gephi/gephi_community_mass_account_nodes_visual.csv` dan
  `output/rm2_actor_type/gephi/gephi_community_mass_account_edges_visual.csv`
- Alias visual: `output/rm2_actor_type/gephi/gephi_community_mass_account_edges.csv`

Panduan import Gephi:

- Graph type: `Undirected`
- Node color: `actor_type_primary`
- Node size: `degree` atau `weighted_degree`
- Edge thickness: `Weight` (`final_weight`)
- Gunakan file visual untuk presentasi publik; file all-evidence sangat padat dan lebih cocok untuk audit.

Batas interpretasi: edge menunjukkan keterhubungan struktural berdasarkan evidence LCN, bukan bukti
pengaruh, kendali, pembayaran, perubahan opini, afiliasi, atau intensi koordinasi.

### Comment-Level Exact and Near-Similarity Analysis

Analisis ini mencari **identical normalized comments**, **near-identical comments**, dan **high textual
similarity** pada level komentar individual. Unit analisisnya berbeda dari
`output/tables/hcc_cosimilarity_pairs.csv`: file lama membandingkan narasi gabungan akun dalam HCC,
sedangkan pipeline ini membandingkan dua `comment_id` observasional yang berbeda.

Metode:

- sumber utama: `dataset.csv`;
- synthetic/injected comment ID berawalan `INJ` dikeluarkan dari output similarity;
- normalisasi teks memakai Unicode NFKC, lowercase, trim whitespace, collapse whitespace, dan normalisasi
  tanda baca/karakter dekoratif berulang;
- kata, angka, negasi, emoji bermakna, dan nama produk dipertahankan;
- near-similarity dihitung memakai TF-IDF character 5-gram (`analyzer=char_wb`, `ngram_range=(5,5)`)
  dengan cosine similarity;
- pencarian pasangan memakai unique normalized texts dan sparse top-k nearest neighbors, bukan dense matrix
  `33847 x 33847`;
- threshold pasangan disimpan mulai `cosine_similarity_char5 >= 0.70`.

Kategori interpretasi:

- `EXACT_DUPLICATE`: normalized text identik;
- `NEAR_EXACT`: cosine similarity `>= 0.92`;
- `HIGH_SIMILARITY`: `0.85-0.9199`;
- `MODERATE_SIMILARITY`: `0.75-0.8499`;
- `WEAK_CANDIDATE`: `0.70-0.7499`, hanya untuk audit/manual review.

Ringkasan output saat ini:

- komentar input: `33847`, unique `comment_id`: `33847`;
- synthetic/injected IDs dikeluarkan: `784`;
- komentar observasional dianalisis: `33063`;
- unique normalized texts: `28891`;
- exact duplicate groups: `800`;
- exact duplicate pairs: `230535`;
- near-exact pairs: `10193`;
- high-similarity pairs: `14674`;
- moderate-similarity pairs: `28753`;
- weak candidates: `16618`;
- similarity pairs yang akun-akunnya juga LCN edge: `13`;
- similarity pairs yang terkait pre-LCN Community-Mass evidence pair: `2758`;
- kandidat awal PowerPoint: `30`, seluruhnya berstatus `PENDING_MANUAL_REVIEW`.

Output utama:

- `output/rm2_comment_similarity/comment_similarity_pairs_all.csv`
- `output/rm2_comment_similarity/exact_duplicate_comment_groups.csv`
- `output/rm2_comment_similarity/exact_duplicate_comment_group_members.csv`
- `output/rm2_comment_similarity/near_similar_comment_clusters.csv`
- `output/rm2_comment_similarity/near_similar_comment_cluster_members.csv`
- `output/rm2_comment_similarity/comment_similarity_threshold_manual_audit.csv`
- `output/rm2_comment_similarity/comment_similarity_integrity_report.csv`
- `output/rm2_comment_similarity/presentation/ppt_comment_similarity_examples.csv`
- `output/rm2_comment_similarity/presentation/ppt_comment_similarity_example_members.csv`
- `output/rm2_comment_similarity/presentation/ppt_comment_similarity_examples.md`
- `output/rm2_comment_similarity/presentation/ppt_comment_similarity_manual_review.csv`

Actor type, network position, brand, HCC, dan Community-Mass evidence status hanya menjadi konteks tambahan.
Similarity tidak membentuk atau mengubah LCN, Louvain, FSA_V, HCC, actor type, sentiment, atau Goals.
Kemiripan tekstual tidak boleh dibaca sebagai bukti pasti copy-paste, bot, instruksi, kampanye, pembayaran,
pengaruh, atau koordinasi yang disengaja. Contoh PowerPoint tetap perlu review manual sebelum dipresentasikan.

Visualisasi PNG:

- `actor_type_account_count.png`
- `actor_type_comment_count.png`
- `actor_type_network_position_stacked.png`
- `lcn_edge_count_by_actor_type_pair.png`
- `lcn_edge_weight_by_actor_type_pair.png`

### Panduan Visualisasi Actor Type di Gephi

Import aggregate actor type:

- Nodes: `output/rm2_actor_type/gephi/gephi_actor_type_nodes.csv`
- Edges: `output/rm2_actor_type/gephi/gephi_actor_type_edges.csv`
- Graph type: `Directed`
- Node color: `actor_type_primary`
- Node size: `Degree`
- Edge color: `edge_type`
- Edge thickness: `Weight` dari `gephi_actor_type_edges.csv`
- Edge opacity: gunakan `edge_visual_opacity`; `Ambiguous` dibuat lebih transparan daripada `Observed`
  dan `Verified`.
- Jangan memakai `edge_weight_raw` sebagai thickness utama pada visualisasi publik.

Import LCN actor type:

- Nodes: `output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv`
- Edges: `output/rm2_actor_type/gephi/gephi_lcn_edges_actor_type.csv`
- Graph type: `Undirected`

Pastikan tipe kolom:

- `Id`: String
- `Label`: String
- `actor_type_primary`: String
- `network_position`: String
- `community`: String atau Integer sesuai sumber
- `degree`: Integer
- `weighted_degree`: Double
- `betweenness`: Double
- `Weight`: Double

Tampilan 1 — Actor Type:

- Partition / Color: `actor_type_primary`
- `Individual Actor` = `#E69F00`
- `Community Actor` = `#009E73`
- `Mass Actor` = `#8A8A8A`
- Ranking / Size: `weighted_degree`
- Minimum node size = `5`, maximum node size = `30`
- Edge color = light gray, opacity = `20–35%`, thickness = `Weight`
- Judul: `Struktur Latent Coordination Network Berdasarkan Type Actor`

Tampilan 2 — Network Position:

- Partition / Color: `network_position`
- Kategori: `HCC`, `LCN Non-HCC`
- Judul: `Posisi Aktor dalam Latent Coordination Network`

Tampilan 3 — Community dan Mass Actor:

- Filter actor type: `Community Actor`, `Mass Actor`
- Gunakan warna actor type dan ukuran `weighted_degree`
- Judul: `Relasi Community Actor dan Mass Actor dalam LCN`

Tampilan 4 — Edge Pair:

- Filter: `actor_type_pair`
- Tampilkan kategori edge secara terpisah, misalnya `Community–Community`, `Community–Mass`, `Mass–Mass`,
  `Individual–Community`, dan `Individual–Mass`.
- Jangan memakai semua kategori edge dengan warna berbeda sekaligus jika visualisasi menjadi terlalu padat.

Layout:

1. OpenOrd
2. ForceAtlas2
3. Noverlap

Parameter awal ForceAtlas2:

- Edge Weight Influence = `1`
- Scaling = `5–15`
- Gravity = `1`
- Prevent Overlap = aktif
- Barnes-Hut Optimization = aktif

Jalankan sampai struktur relatif stabil, bukan sampai komponen terpisah terlalu jauh.

Caption standar:

> Visualisasi hanya mencakup akun yang masuk Latent Coordination Network. Mass Actor di luar LCN tidak
> ditampilkan karena tidak mempunyai edge koordinasi yang memenuhi kriteria pembentukan jaringan. Seluruh
> Mass Actor tetap dihitung dalam ringkasan statistik actor universe.

Batas interpretasi:

- Individual Actor tidak otomatis merupakan pengendali komunitas.
- Community Actor adalah anggota HCC final, bukan akun yang terbukti sebagai buzzer.
- Mass Actor adalah kategori residual komentator umum, bukan aktor yang terbukti melakukan amplifikasi massal.
- Mass Actor dalam LCN hanya merupakan subset Mass Actor yang mempunyai posisi pada jaringan koordinasi laten.
- Edge menunjukkan hubungan berdasarkan evidence RM1, bukan bukti pembayaran, instruksi, afiliasi,
  pengaruh kausal, atau astroturfing.
- Node besar menunjukkan weighted degree tinggi, bukan kekuasaan atau kendali.
- Posisi pusat atau betweenness tinggi menunjukkan posisi struktural, bukan kepemimpinan.

### Interpretasi `goal_orientation`

Dimensi *goals* dioperasionalisasikan melalui distribusi sentimen komentar pada akun dan HCC. Sentimen
positif, netral, dan negatif diperlakukan sebagai **indikator orientasi pesan**, bukan sebagai bukti
langsung niat aktor. `goal_orientation` per HCC diklasifikasikan menjadi `Promotional / Supportive`,
`Critical / Complaint`, `Neutral Engagement`, `Polarized / Contested`, `Mixed Goals`, atau
`Insufficient Text` (HCC dengan `<5` komentar bertext) — masing-masing disertai `goal_confidence`
(`High`/`Medium`/`Low`/`None`). `goal_confidence` adalah stabilitas bootstrap, bukan akurasi atau
kebenaran label.

### Batasan Analisis

- **Analisis sentimen digunakan untuk memetakan orientasi pesan atau dimensi goals. Hasil ini tidak
  dimaknai sebagai bukti niat aktor, bukti bot, atau bukti pasti astroturfing.**
- Sentimen **tidak** digunakan untuk membentuk HCC (HCC sudah final dari RM1) dan **tidak** digunakan
  untuk membuktikan astroturfing.
- Akurasi model bergantung pada domain teks; komentar TikTok yang pendek, bahasa gaul, dan campur kode
  berpotensi menurunkan akurasi dibanding korpus pelatihan model. Human validation 600 komentar sudah
  tersedia dan comment-level sentiment model melewati gate validasi.
- HCC dengan sedikit komentar bertext diberi label `Insufficient Text`/`goal_confidence` rendah secara
  eksplisit agar tidak disalahartikan sebagai pola yang kuat.
- Goal counts tetap perlu dibaca hati-hati karena diagnostic agreement goal terhadap heuristic HCC review
  masih gagal; confidence/stability bukan bukti kebenaran label goal.
- Perbandingan HCC vs Non-HCC dipakai untuk melihat perbedaan orientasi sentimen, **bukan** untuk
  membuktikan koordinasi.
