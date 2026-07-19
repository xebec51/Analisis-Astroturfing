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

Seluruh grafik ditampilkan inline di notebook; sebagian juga disimpan sebagai file:

- `output/visualisasi/HCC_hashtag_*`, `output/visualisasi/hashtag_by_community/` -- visualisasi hashtag
  (konteks brand/video) per HCC.
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
| `output/visualisasi/` | Grafik & visualisasi (PNG/PDF/SVG untuk grafik lama, PNG-only untuk grafik baru) |
| `README.md` | Berkas dokumentasi ini |

Seluruh hasil analisis **hanya berupa indikasi pola koordinasi** berdasarkan data yang teramati. Pola
koordinasi yang terdeteksi dapat pula bersifat organik, misalnya berasal dari pengguna aktif, penggemar
produk, atau komunitas diskusi yang wajar.

---

## RM2 — Sentiment-based Goals Mapping

Rumusan Masalah 2 (RM2) menjawab pertanyaan *"Bagaimana three dimensions tipologi aktor digital
astroturfing pada produk skincare overclaim di platform TikTok?"* — terdiri dari tiga dimensi: **target**
dan **actor type** (dipetakan dengan SNA pada RM1, `tiktok_coordination_analysis.ipynb`), serta **goals**
(dipetakan dengan analisis sentimen pada notebook `02_rm2_sentiment_goals.ipynb`, terpisah dari RM1).

**`02_rm2_sentiment_goals.ipynb` hanya membaca** output RM1 sebagai input dan **tidak pernah** mengubah
notebook RM1, pipeline LCN/Louvain/FSA_V, hasil HCC, maupun file output RM1 yang sudah ada.

### Tujuan

1. Melakukan analisis sentimen komentar TikTok (Positive / Neutral / Negative) menggunakan model IndoBERT
   fine-tuned untuk sentimen Bahasa Indonesia.
2. Mengagregasi sentimen pada tiga level: **komentar**, **akun**, **HCC**.
3. Memetakan dimensi **`goals`** melalui `goal_orientation` (orientasi pesan berbasis distribusi sentimen).
4. Menghubungkan hasil sentimen dengan hasil RM1 (`HCC`, `community`, `brand_label_auto`, `primary_brand`).

### Input

| File | Sumber | Peran |
|---|---|---|
| `dataset.csv` | Data mentah | Teks komentar untuk analisis sentimen |
| `output/gephi/gephi_hcc_nodes.csv` | RM1 (hanya dibaca) | Keanggotaan HCC (`id`=username), brand & atribut struktural |
| `output/tables/hcc_brand_profile_auto.csv` | RM1 (hanya dibaca) | Profil brand per HCC |
| `output/tables/focal_structures.csv` | RM1 (hanya dibaca, opsional) | Metadata FSA_V/HCC |
| `output/gephi/gephi_hcc_edges.csv` | RM1 (hanya dibaca, opsional) | Edge HCC untuk export Gephi RM2 |

### Model Sentimen

Model utama: [`mdhugol/indonesia-bert-sentiment-classification`](https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification)
(IndoBERT *fine-tuned* untuk sentimen Bahasa Indonesia). Jika model gagal dimuat (mis. offline), notebook
otomatis menggunakan fallback: baca ulang hasil prediksi sebelumnya jika tersedia, atau rule-based baseline
sementara (bukan hasil final) — lihat Section 6 pada notebook. `MODEL_NAME` dan `LABEL_MAP` dapat diubah
di Section 2 jika model diganti.

### Output Tabel (`output/rm2_sentiment/tables/`)

| File | Level | Isi |
|---|---|---|
| `comment_sentiment.csv` | Komentar | `sentiment_label`/`score`/`confidence` + keterkaitan HCC/brand per komentar |
| `account_sentiment_summary.csv` | Akun | Agregasi sentimen per akun + atribut struktural HCC |
| `hcc_sentiment_goals_summary.csv` | HCC | Agregasi sentimen per HCC + `goal_orientation`, `goal_confidence` |
| `hcc_vs_nonhcc_sentiment_summary.csv` | Grup | Perbandingan distribusi sentimen HCC vs Non-HCC |
| `brand_sentiment_summary.csv` | Brand | Agregasi sentimen & `dominant_goal_orientation` per `brand_label_auto` |
| `sentiment_validation_sample.csv` | Validasi | Sampel stratified (hingga 300 komentar) untuk anotasi manual |
| `sentiment_validation_metrics.csv` | Validasi | Accuracy/macro precision/recall/F1 (setelah `manual_label` diisi) |

### Output Visualisasi (`output/rm2_sentiment/visualisasi/`, PNG saja)

`sentiment_distribution_overall.png`, `sentiment_hcc_vs_nonhcc.png`, `sentiment_by_brand_label_auto.png`,
`hcc_sentiment_heatmap.png`, `top_hcc_positive_ratio.png`, `top_hcc_negative_ratio.png`,
`goal_orientation_by_brand.png`.

### WordCloud Narasi Komentar

WordCloud digunakan sebagai visualisasi **eksploratif** untuk membaca kata dominan dalam komentar (Section
14.8 pada notebook) — dibangun dari `comment_sentiment` yang sudah ada (tidak ada prediksi sentimen ulang).
Token WordCloud telah dibersihkan dengan stopwords khusus percakapan TikTok dan penghapusan emoji/simbol agar visualisasi lebih fokus pada tema substantif seperti bahan aktif, kondisi kulit, pengalaman pemakaian, harga, keamanan, dan keluhan.

- Output disimpan di: `output/rm2_sentiment/visualisasi/wordcloud/` (dan subfolder `by_brand/` untuk
  WordCloud per brand).
- Token frequency disimpan di: `output/rm2_sentiment/tables/wordcloud_token_frequency.csv`.
- WordCloud + bar chart top words dibuat untuk: keseluruhan (*overall*), per sentimen (Positive / Neutral
  / Negative), dan HCC / Non-HCC. WordCloud (tanpa bar chart) juga dibuat per `brand_label_auto`.
- Kata brand (azarine, daviena, maryame, originote, theoriginote) dihapus dari WordCloud secara default
  (`REMOVE_BRAND_TERMS_FROM_WORDCLOUD = True`) supaya tema narasi lebih terlihat, tidak didominasi nama
  brand.
- **WordCloud bukan bukti sentimen atau astroturfing** — hanya alat bantu eksploratif untuk membaca pola
  kata; harus dibaca bersama hasil analisis sentimen, HCC, dan `brand_label_auto`, bukan berdiri sendiri.

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
- Community-Mass association dibaca melalui `Direct Interaction`, `Temporal Association`,
  `Shared-Video Context Only`, atau `No Observed Community Association`. Shared-video hanya konteks bersama,
  direct interaction harus lolos validasi parent comment, dan `temporal_mass_comment_count` dipisahkan dari
  `temporal_hcc_comment_pair_count`.
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

## RM2 — Temporal Activity Profile

Notebook `04_rm2_temporal_activity_profile.ipynb` menambahkan analisis hari dalam pekan, jam, daypart,
recurrence mingguan, dan konteks brand untuk tiga actor type RM2. Notebook ini hanya membaca output RM1
dan RM2 yang sudah tersedia; tidak menjalankan ulang LCN, Louvain, FSA_V, model sentimen, WordCloud,
keanggotaan HCC, atau aturan actor type.

- Zona waktu utama output temporal adalah `Asia/Jakarta` / WIB. Notebook tetap menghitung sensitivity
  WIB-WITA untuk melihat berapa komentar yang berubah hari bila dibaca dengan `Asia/Makassar`.
- `weekday_activity_lift` membandingkan share komentar Community Actor pada suatu hari dengan baseline
  seluruh komentar pada hari yang sama.
- Normalisasi per-video tersedia melalui `comments_per_active_video` dan
  `community_accounts_per_active_video`, sehingga raw count tidak dibaca sendirian.
- Profil dibuat pada comment-level, account-level, HCC-level, brand-context level, video-normalized level,
  dan week-level recurrence.
- HCC-level profile memakai weekday entropy dan `weekday_concentration_score`; HCC dengan sedikit komentar
  atau minggu aktif terbatas diberi reliability `Insufficient observations` atau `Limited recurrence`.
- Visualisasi weekday x hour, weekday x daypart, weekday lift, HCC concentration, recurrence, dan konteks
  brand disimpan sebagai PNG di `output/rm2_temporal/visualisasi/`.
- Output tabel disimpan di `output/rm2_temporal/tables/`, termasuk
  `actor_type_weekday_summary.csv`, `community_weekday_activity_lift.csv`,
  `account_temporal_profile.csv`, `hcc_weekday_profile.csv`, `hcc_temporal_goal_profile.csv`,
  `weekday_actor_type_chi_square.csv`, dan tabel audit timestamp/checksum.
- Output Gephi temporal disimpan di `output/rm2_temporal/gephi/`; edge HCC disalin tanpa perubahan dan
  hash edge divalidasi sama dengan sumbernya.

Analisis brand-HCC pada notebook temporal memakai `output/tables/hcc_brand_profile_auto.csv`, yaitu hasil
auto-labeling RM1 berdasarkan hashtag metadata video:

- `brand_label_auto` adalah klasifikasi eksklusif per HCC. Exclusive count selalu berjumlah total HCC valid.
- HCC mixed dapat diasosiasikan dengan beberapa brand pada analisis multi-label melalui `brand_combo`.
  Multi-label incidence dapat mempunyai total lebih besar daripada jumlah HCC.
- Jumlah komunitas HCC berbeda dari jumlah node pada legenda Gephi; satu HCC dapat berisi beberapa akun.
- Jika satu HCC memiliki `brand_combo = Maryame + The Originote + Azarine`, maka pada ringkasan eksklusif
  HCC tersebut dihitung satu kali sebagai `Mixed_3plus_Brands`. Pada analisis multi-label, HCC yang sama
  dihitung masing-masing satu kali pada Maryame, The Originote, dan Azarine.
- Konteks brand berasal dari hashtag metadata video, bukan nama username, teks komentar, hashtag komentar,
  dugaan manual, atau warna visualisasi Gephi.

Community Actor merupakan akun anggota HCC, bukan akun yang telah terbukti sebagai buzzer. Aktivitas yang
terkonsentrasi pada hari atau jam tertentu tidak membuktikan jadwal kerja, pembayaran, atau koordinasi
terencana.

Pengelompokan brand pada HCC menunjukkan konteks video yang dikomentari oleh anggota komunitas berdasarkan
hashtag metadata video. Label tersebut tidak berarti seluruh anggota HCC menulis hashtag, mendukung brand,
berafiliasi dengan brand, atau membahas brand secara eksplisit dalam setiap komentar.

### Interpretasi `goal_orientation`

Dimensi *goals* dioperasionalisasikan melalui distribusi sentimen komentar pada akun dan HCC. Sentimen
positif, netral, dan negatif diperlakukan sebagai **indikator orientasi pesan**, bukan sebagai bukti
langsung niat aktor. `goal_orientation` per HCC diklasifikasikan menjadi `Promotional / Supportive`,
`Critical / Complaint`, `Neutral Engagement`, `Polarized / Contested`, `Mixed Goals`, atau
`Insufficient Text` (HCC dengan `<5` komentar bertext) — masing-masing disertai `goal_confidence`
(`High`/`Medium`/`Low`/`None`).

### Batasan Analisis

- **Analisis sentimen digunakan untuk memetakan orientasi pesan atau dimensi goals. Hasil ini tidak
  dimaknai sebagai bukti niat aktor, bukti bot, atau bukti pasti astroturfing.**
- Sentimen **tidak** digunakan untuk membentuk HCC (HCC sudah final dari RM1) dan **tidak** digunakan
  untuk membuktikan astroturfing.
- Akurasi model bergantung pada domain teks; komentar TikTok yang pendek, bahasa gaul, dan campur kode
  berpotensi menurunkan akurasi dibanding korpus pelatihan model. Validasi manual (Section 13 notebook)
  disediakan untuk mengukur potensi penyimpangan ini.
- HCC dengan sedikit komentar bertext diberi label `Insufficient Text`/`goal_confidence` rendah secara
  eksplisit agar tidak disalahartikan sebagai pola yang kuat.
- Perbandingan HCC vs Non-HCC dipakai untuk melihat perbedaan orientasi sentimen, **bukan** untuk
  membuktikan koordinasi.
