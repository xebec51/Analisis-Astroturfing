# Final RM2 Sentiment V2 Analysis Report

## Status Validasi Model

Model Sentimen V2 berstatus `FINAL_MODEL_VALIDATED` berdasarkan evaluasi locked test final 300 komentar observasional. Evaluasi dilakukan satu kali setelah locked test dibekukan, dengan threshold abstention `0.42` dan model hash `477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc`.

## Locked-Test Metrics

- Coverage evaluable: `0.9343`
- Abstention rate: `0.0657`
- Macro-F1 covered: `0.7309`
- Balanced accuracy covered: `0.7188`
- MCC covered: `0.6369`

| class_label | precision | recall | f1 | support | covered_predicted_count | abstained_true_count |
| --- | --- | --- | --- | --- | --- | --- |
| Negative | 0.6756756756756757 | 0.7352941176470589 | 0.704225352112676 | 34 | 37 | 7 |
| Neutral | 0.8842105263157894 | 0.9438202247191011 | 0.9130434782608695 | 178 | 190 | 8 |
| Positive | 0.7241379310344828 | 0.4772727272727273 | 0.5753424657534246 | 44 | 29 | 3 |

## Distribusi Sentimen Observasional

Denominator utama adalah `33063` komentar observasional/non-INJ. Sebanyak `784` komentar INJ disimpan terpisah sebagai diagnostic dan tidak dicampurkan dalam denominator utama.

| label | count | percentage_of_total |
| --- | --- | --- |
| Positive | 2718 | 8.220669630705018 |
| Neutral | 23977 | 72.51913014547983 |
| Negative | 4771 | 14.43002752321326 |
| Uncertain | 1593 | 4.818074584883404 |
| No Text | 4 | 0.012098115718476847 |

## HCC vs Non-HCC

Perbandingan ini menunjukkan perbedaan distribusi sentimen teramati menurut posisi jaringan, bukan efek kausal, niat, atau pengaruh.

| group | total_comments | evaluable_comments | coverage | uncertain_count | no_text_count | negative_count | negative_ratio_evaluable | negative_ratio_total | negative_ci_low | negative_ci_high | neutral_count | neutral_ratio_evaluable | neutral_ratio_total | neutral_ci_low | neutral_ci_high | positive_count | positive_ratio_evaluable | positive_ratio_total | positive_ci_low | positive_ci_high | negative_hcc_minus_nonhcc | neutral_hcc_minus_nonhcc | positive_hcc_minus_nonhcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HCC | 945 | 917 | 0.9703703703703703 | 28 | 0 | 87 | 0.09487459105779716 | 0.09206349206349207 | 0.07756191228744493 | 0.11556750323852608 | 541 | 0.5899672846237731 | 0.5724867724867725 | 0.5578219617579645 | 0.6213619500027551 | 289 | 0.31515812431842966 | 0.3058201058201058 | 0.28591231747498436 | 0.3459461910893549 | -0.058452850102306275 | -0.1771936699082901 | 0.23564652001059633 |
| Non-HCC | 32118 | 30549 | 0.951148888473753 | 1565 | 4 | 4684 | 0.15332744116010344 | 0.14583722523195716 | 0.14933064102182603 | 0.1574114199195693 | 23436 | 0.7671609545320632 | 0.7296842891836354 | 0.7623880831569213 | 0.7718666422693381 | 2429 | 0.07951160430783331 | 0.07562737405816053 | 0.07653043961882364 | 0.08259851027237075 | -0.058452850102306275 | -0.1771936699082901 | 0.23564652001059633 |

## Account-Level HCC vs Non-HCC

| group | n_accounts | mean_negative_ratio | median_negative_ratio | iqr_negative_ratio | mean_negative_ratio_ci_low | mean_negative_ratio_ci_high | mean_neutral_ratio | median_neutral_ratio | iqr_neutral_ratio | mean_neutral_ratio_ci_low | mean_neutral_ratio_ci_high | mean_positive_ratio | median_positive_ratio | iqr_positive_ratio | mean_positive_ratio_ci_low | mean_positive_ratio_ci_high | median_coverage | mean_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HCC | 207 | 0.09321946169772256 | 0.0 | 0.16666666666666666 | 0.07066942719116631 | 0.1152414021164021 | 0.5954911433172303 | 0.6666666666666666 | 0.8 | 0.5433008396595354 | 0.649781314699793 | 0.31128939498504715 | 0.0 | 0.775 | 0.25410570508396596 | 0.3618842017483322 | 1.0 | 0.9717966413618586 |
| Non-HCC | 25983 | 0.14953151754663857 | 0.0 | 0.0 | 0.14542150301975754 | 0.15345127637130354 | 0.7772202924231403 | 1.0 | 0.0 | 0.7725164487428114 | 0.7822998208689603 | 0.07324819003022118 | 0.0 | 0.0 | 0.07009001246751856 | 0.07625063336012483 | 1.0 | 0.9536847530626248 |

## Goal Orientation HCC

Goal orientation dibaca sebagai orientasi pesan deskriptif berbasis pola sentimen teramati.

| goal_orientation | n_hcc |
| --- | --- |
| Neutral Engagement | 31 |
| Promotional / Supportive | 11 |

## Brand Context

Konteks brand berasal dari label brand HCC berdasarkan metadata hashtag video. Kategori mixed bukan brand tunggal dan tidak membuktikan afiliasi.

| brand_label_auto | n_comments | n_accounts | n_hccs | evaluable_comments | coverage | support_status | negative_count | negative_ratio | negative_ci_low | negative_ci_high | neutral_count | neutral_ratio | neutral_ci_low | neutral_ci_high | positive_count | positive_ratio | positive_ci_low | positive_ci_high | dominant_sentiment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Azarine | 102 | 18 | 3 | 99 | 0.9705882352941176 | AVAILABLE | 8 | 0.08080808080808081 | 0.041513905489623 | 0.1514196933653151 | 50 | 0.5050505050505051 | 0.40822998685489836 | 0.6014937047251627 | 41 | 0.41414141414141414 | 0.32209342077368147 | 0.5126038223652817 | Neutral |
| Daviena | 206 | 40 | 11 | 202 | 0.9805825242718447 | AVAILABLE | 16 | 0.07920792079207921 | 0.049340663768739024 | 0.12478157391305127 | 111 | 0.5495049504950495 | 0.48060428966299035 | 0.616557800021505 | 75 | 0.3712871287128713 | 0.3076415304924698 | 0.4397370363278426 | Neutral |
| Maryame | 163 | 35 | 6 | 154 | 0.9447852760736196 | AVAILABLE | 16 | 0.1038961038961039 | 0.06496831920387043 | 0.1621048985029952 | 64 | 0.4155844155844156 | 0.34072752354135816 | 0.4945503753142033 | 74 | 0.4805194805194805 | 0.4030476633173572 | 0.5589395441108491 | Positive |
| The Originote | 228 | 54 | 9 | 222 | 0.9736842105263158 | AVAILABLE | 19 | 0.08558558558558559 | 0.055474129279163994 | 0.1297955464581669 | 124 | 0.5585585585585585 | 0.49279209985098454 | 0.6223328372730882 | 79 | 0.35585585585585583 | 0.2958168029709911 | 0.4207987364159065 | Neutral |
| Mixed_2_Brands | 51 | 13 | 3 | 51 | 1.0 | AVAILABLE | 4 | 0.0784313725490196 | 0.030921696126931502 | 0.18500198224860823 | 40 | 0.7843137254901961 | 0.6537337712118704 | 0.8750618875581144 | 7 | 0.13725490196078433 | 0.06811015331542725 | 0.25721952342631627 | Neutral |
| Mixed_3plus_Brands | 177 | 41 | 9 | 172 | 0.9717514124293786 | AVAILABLE | 21 | 0.12209302325581395 | 0.0812588499238568 | 0.17943941487810128 | 141 | 0.8197674418604651 | 0.7555393738198233 | 0.8700236328862121 | 10 | 0.05813953488372093 | 0.03188308865655954 | 0.10370257480419148 | Neutral |
| Not identified | 18 | 6 | 1 | 17 | 0.9444444444444444 | LOW_SUPPORT | 3 | 0.17647058823529413 | 0.06191013712548989 | 0.41029929017375777 | 11 | 0.6470588235294118 | 0.4129996672121183 | 0.8269051385609512 | 3 | 0.17647058823529413 | 0.06191013712548989 | 0.41029929017375777 | Neutral |

## Legacy V1 vs Final V2

| comparison_scope | label | legacy_count | final_v2_count | count_delta_final_minus_legacy |
| --- | --- | --- | --- | --- |
| all_observational | Positive | 2791 | 2718 | -73 |
| all_observational | Neutral | 13933 | 23977 | 10044 |
| all_observational | Negative | 11585 | 4771 | -6814 |
| all_observational | Uncertain | 3034 | 1593 | -1441 |
| all_observational | No Text | 1720 | 4 | -1716 |
| hcc_comments | Positive | 293 | 289 | -4 |
| hcc_comments | Neutral | 323 | 541 | 218 |
| hcc_comments | Negative | 212 | 87 | -125 |
| hcc_comments | Uncertain | 68 | 28 | -40 |
| hcc_comments | No Text | 49 | 0 | -49 |
| nonhcc_comments | Positive | 2498 | 2429 | -69 |
| nonhcc_comments | Neutral | 13610 | 23436 | 9826 |
| nonhcc_comments | Negative | 11373 | 4684 | -6689 |
| nonhcc_comments | Uncertain | 2966 | 1565 | -1401 |
| nonhcc_comments | No Text | 1671 | 4 | -1667 |
| all_observational | changed_comment_id |  | 13547 |  |
| hcc_goals | changed_hcc_goal_orientation |  | 21 |  |

## Batas Interpretasi

- Sentimen adalah indikator orientasi pesan, bukan bukti niat, pembayaran, afiliasi, kendali, atau pengaruh kausal.
- HCC menunjukkan pola koordinasi struktural, bukan bukti bahwa akun adalah bot atau buzzer.
- Confidence dan stability adalah indikator ketidakpastian model/agregasi, bukan akurasi aktual pada setiap komentar.
