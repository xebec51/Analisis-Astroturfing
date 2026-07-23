# Panduan Anotasi Manusia Sentimen RM2 V2

## Tujuan

Paket ini mengumpulkan label manusia tambahan untuk analisis sentimen RM2 pada level komentar. Tujuannya adalah memperbaiki diagnosis lanjutan terhadap recall kelas Positive, precision kelas Negative, pembedaan pertanyaan atau kondisi kulit dari evaluasi produk, serta coverage model.

Saat memberi label, jangan menggunakan prediksi model, label heuristik, output goal HCC, probability, confidence, atau hasil otomatis sebelumnya. Nilai komentar hanya berdasarkan teks komentar dan konteks video/brand yang tersedia di file anotasi.

## Label Sentimen Utama

- Positive: gunakan jika komentar jelas memuji, merekomendasikan, mendukung, mempromosikan, atau melaporkan hasil membaik. Contoh: "Aku cocok, bekas jerawat makin pudar."
- Neutral: gunakan untuk pertanyaan, pernyataan faktual, tagging, nama produk saja, logistik harga/pembelian tanpa evaluasi, atau deskripsi kondisi kulit tanpa evaluasi produk. Contoh: "Ini dipakai pagi atau malam?"
- Negative: gunakan jika komentar jelas mengeluhkan efek produk/brand, keamanan, keaslian, nilai harga, atau hasil pemakaian. Contoh: "Setelah pakai ini wajahku perih dan merah."
- Uncertain: gunakan jika sinyal positif dan negatif sama kuat atau konteks tidak cukup untuk menentukan polaritas dominan. Contoh: "Bagus sih, tapi di aku bikin kering."
- No Text: gunakan jika teks kosong, terhapus, tidak terbaca, atau hanya berisi konten yang tidak dapat ditafsirkan.

## Target Sentimen

- Product / Brand: evaluasi terutama diarahkan pada produk skincare atau brand.
- Skin condition: komentar terutama menjelaskan jerawat, kusam, iritasi, berminyak, kering, atau kondisi kulit lain.
- Usage question: komentar menanyakan apakah, kapan, atau bagaimana memakai produk atau bahan tertentu.
- Creator / Seller: evaluasi diarahkan pada creator, penjual, layanan, atau akun.
- Price / Purchase: komentar membahas harga, link, keranjang, checkout, pengiriman, stok, atau ketersediaan.
- Promotion / CTA: komentar terutama berupa rekomendasi, ajakan membeli, promosi, atau call to action.
- General discussion: percakapan skincare umum tanpa target evaluasi spesifik.
- Other / unclear: target tidak dapat ditentukan dengan cukup jelas.

## Complaint Scope

- product_effect: keluhan atau pujian berhubungan dengan efek produk atau hasil pemakaian.
- skin_condition: kondisi kulit disebut sebagai masalah tanpa hubungan sebab-akibat produk yang jelas.
- price_value: harga, nilai, ongkir, atau syarat pembelian menjadi isu utama.
- safety_concern: iritasi, keamanan bahan, bahaya, atau risiko kesehatan menjadi isu utama.
- authenticity_concern: keaslian produk, official store, originalitas, atau kepercayaan menjadi isu utama.
- usage_confusion: urutan pemakaian, frekuensi, kompatibilitas, atau cara pakai menjadi sumber kebingungan.
- not_applicable: tidak ada cakupan keluhan yang relevan.
- unclear: cakupan tidak dapat ditentukan dari komentar.

## Aturan Keputusan

1. Pertanyaan tentang kondisi kulit tanpa evaluasi produk cenderung Neutral dengan target Skin condition atau Usage question.
2. Keluhan kondisi kulit tidak otomatis berarti Negative terhadap Product / Brand. Beri Negative terhadap Product / Brand hanya jika komentar mengaitkan efek buruk dengan pemakaian produk.
3. Efek buruk setelah menggunakan produk dapat diberi Negative, target Product / Brand, scope product_effect atau safety_concern.
4. Hasil membaik, rekomendasi, dukungan, dan promosi yang jelas dapat diberi Positive.
5. Jika sinyal positif dan negatif sama kuat, gunakan Uncertain.
6. Emoji tidak boleh mengalahkan makna utama teks.
7. Kata seperti jerawat, bruntusan, kusam, mahal, murah, aman, dan cocok harus dinilai berdasarkan konteks kalimat.

## Contoh

| Contoh komentar | Sentimen | Target | Complaint scope | Catatan |
|---|---|---|---|---|
| "Kak ini aman buat kulit sensitif?" | Neutral | Usage question | usage_confusion | Pertanyaan, bukan keluhan produk. |
| "Jerawatku lagi parah banget" | Neutral | Skin condition | skin_condition | Kondisi kulit tanpa sebab-akibat produk. |
| "Pakai ini malah breakout" | Negative | Product / Brand | product_effect | Efek produk disalahkan. |
| "Aku cocok banget, jadi lebih cerah" | Positive | Product / Brand | product_effect | Ada hasil membaik yang jelas. |
| "Mahal tapi worth it" | Positive | Product / Brand | price_value | Sinyal positif dominan meskipun harga disebut. |
| "Bagus tapi bikin kering" | Uncertain | Product / Brand | product_effect | Sentimen campuran tanpa polaritas dominan. |
| "Checkout sekarang, lagi promo" | Positive | Promotion / CTA | not_applicable | Ajakan promosi yang jelas. |
| "[emoji tertawa]" | No Text | Other / unclear | unclear | Emoji saja tanpa sentimen yang dapat ditafsirkan. |
