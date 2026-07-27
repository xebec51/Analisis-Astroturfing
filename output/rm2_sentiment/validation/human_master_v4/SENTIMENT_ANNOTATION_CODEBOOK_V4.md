# SENTIMENT ANNOTATION CODEBOOK V4

Positive: Komentar menunjukkan dukungan, kepuasan, rekomendasi, pengalaman baik, pujian, harapan positif, atau orientasi positif terhadap produk/konteks video.

Neutral: Komentar bersifat informasional, pertanyaan tanpa evaluasi, pernyataan faktual, tagging, permintaan informasi, atau tidak menunjukkan evaluasi positif/negatif yang cukup.

Negative: Komentar menunjukkan ketidakpuasan, kritik, keluhan, penolakan, pengalaman buruk, kekhawatiran negatif, atau evaluasi negatif terhadap produk/konteks.

Uncertain: Teks tersedia tetapi makna sentimen tidak dapat ditentukan secara memadai, terlalu ambigu, sarkastik tanpa konteks cukup, atau konflik interpretasi tidak dapat diselesaikan.

No Text: Komentar kosong, hanya whitespace, hanya artefak teknis, atau tidak mempunyai teks yang dapat dinilai.

Aturan khusus:

- Pertanyaan tidak otomatis Neutral; nilai orientasi evaluatifnya.
- Promosi tidak otomatis Positive.
- Emoji harus dibaca bersama teks.
- Sarkasme harus menggunakan konteks yang tersedia.
- Nama brand tidak menentukan label.
- HCC tidak menentukan label.
- Kata 'bagus' dalam negasi bukan Positive.
- 'Belum coba' umumnya Neutral kecuali memiliki evaluasi lain.
- 'Mau coba' dapat Neutral bila hanya menyatakan niat.
- 'Wajib beli' dapat Positive bila menunjukkan rekomendasi.
- 'Aman nggak?' umumnya Neutral/Uncertain tergantung konteks.
- Label didasarkan pada isi komentar, bukan dugaan identitas atau motif akun.

Contoh sintetis:

## Positive
1. Aku cocok pakai serum ini, kulit terasa lebih halus.
2. Rekomendasi banget buat yang cari pelembap ringan.
3. Hasilnya pelan tapi bekas jerawatku mulai pudar.
4. Produknya nyaman dan tidak bikin lengket.
5. Wajib beli lagi kalau sudah habis.
6. Suka sama teksturnya, cepat meresap.
7. Akhirnya nemu sunscreen yang pas.
8. Kulitku jadi terlihat lebih cerah setelah rutin pakai.
9. Mantap, packaging aman dan isinya bagus.
10. Aku percaya produk ini karena sejauh ini cocok.
11. Bagus untuk kulitku yang mudah kering.
12. Worth it dengan harga segitu.
13. Temanku pakai dan hasilnya kelihatan baik.
14. Semoga brand ini terus keluarin produk seperti ini.
15. Ini penyelamat kulitku waktu lagi kusam.

## Neutral
1. Harganya berapa?
2. Beli di mana ya kak?
3. Ini dipakai pagi atau malam?
4. Kandungan utamanya apa?
5. Aku baru mau coba minggu depan.
6. Ukuran botolnya berapa ml?
7. Ada link produknya?
8. Untuk umur 17 boleh tidak?
9. Aku pakai merek lain sekarang.
10. Ini varian yang mana?
11. Kak spill urutan pemakaiannya.
12. Belum coba, masih cari info.
13. Tag teman yang tanya produk ini.
14. Ini sama dengan yang di video sebelumnya?
15. Pengiriman ke luar kota bisa?

## Negative
1. Aku tidak cocok, malah muncul jerawat.
2. Baunya mengganggu dan bikin pusing.
3. Sudah pakai lama tapi tidak ada perubahan.
4. Kulitku terasa perih setelah pakai.
5. Kecewa karena isinya bocor.
6. Produk ini bikin wajahku makin berminyak.
7. Aku kapok beli lagi.
8. Teksturnya berat dan lengket.
9. Klaimnya tidak sesuai di kulitku.
10. Muka jadi gatal setelah pemakaian.
11. Warnanya berubah dan aku ragu aman.
12. Tidak worth it untuk harganya.
13. Packaging rusak waktu sampai.
14. Aku merasa efeknya buruk.
15. Bukannya membaik, kulitku malah kusam.

## Uncertain
1. Hmm begitu ya.
2. Yang ini gimana sih sebenarnya?
3. Aduh kok bisa gitu.
4. Aku kira beda.
5. Mungkin iya mungkin tidak.
6. Tergantung kulit masing-masing kali ya.
7. Baru lihat, belum paham.
8. Komentarnya bercanda tapi konteksnya kurang jelas.
9. Wah menarik juga.
10. Entah ini bagus atau tidak.
11. Aku bingung maksud videonya.
12. Kalimatnya terpotong dan ambigu.
13. Nada komentarnya bisa sarkas, tapi tidak jelas.
14. Tidak tahu ini pujian atau kritik.
15. Butuh konteks video untuk menilai.

## No Text
1. 
2.    
3. [deleted]
4. deleted
5. <teks tidak tersedia>
6. <komentar kosong>
7. <hanya artefak teknis>
8. <file tidak memuat isi komentar>
9. <teks rusak tidak terbaca>
10. <hanya whitespace>
11. <baris tanpa comment text>
12. <konten hilang saat ekspor>
13. <placeholder kosong>
14. <komentar tidak terscrape>
15. <tidak ada teks yang dapat dinilai>

