# IndoBERT V4 Initial Audit

Created UTC: 2026-07-28T08:21:40.157406+00:00

## Summary
- OOF rows: 55059
- Metrics rows: 181
- Trial summary rows: 10
- OOF duplicate key count: 0
- Metrics duplicate key count: 0
- Development evaluable rows: 1824
- Locked evaluable rows: 672
- Dev/locked overlap: {'comment_id': 0, 'text_cluster_id': 0, 'exact_duplicate_group_id': 0, 'near_duplicate_cluster_id': 12}

## Top Trials

- indobenchmark__indobert-base-p2__context_sep_comment__len256__lr3p0em05__warm0p06__wd0p05__drop0p3__focal_loss__ls0p05: macro_f1=0.8793080026199229, selection=0.8796612825019942, accuracy=0.8832236842105264, n_seeds=3
- indolem__indobertweet-base-uncased__comment_only__len128__lr2p0em05__warm0p1__wd0p01__drop0p2__weighted_cross_entropy__ls0p0: macro_f1=0.8779185284926121, selection=0.8776660860473072, accuracy=0.8801169590643275, n_seeds=3
- indolem__indobertweet-base-uncased__context_sep_comment__len192__lr2p0em05__warm0p1__wd0p05__drop0p2__weighted_cross_entropy__ls0p03: macro_f1=0.8756084517638394, selection=0.8758431743911482, accuracy=0.8775584795321637, n_seeds=3
- indobenchmark__indobert-base-p2__context_sep_comment__len192__lr2p0em05__warm0p1__wd0p05__drop0p2__weighted_cross_entropy__ls0p03: macro_f1=0.8733647935477727, selection=0.8746902035846005, accuracy=0.8777412280701755, n_seeds=3
- indolem__indobertweet-base-uncased__context_sep_comment__len128__lr1p0em05__warm0p06__wd0p01__drop0p1__cross_entropy__ls0p0: macro_f1=0.8707873118781215, selection=0.8712939354496743, accuracy=0.8733552631578947, n_seeds=3
- indobenchmark__indobert-base-p2__comment_only__len192__lr2p0em05__warm0p1__wd0p01__drop0p1__weighted_cross_entropy__ls0p0: macro_f1=0.8705901116573634, selection=0.8714688992841394, accuracy=0.8744517543859649, n_seeds=3
- indobenchmark__indobert-base-p2__comment_only__len128__lr3p0em05__warm0p06__wd0p01__drop0p2__focal_loss__ls0p03: macro_f1=0.8697353629574406, selection=0.8702138289978016, accuracy=0.8735380116959064, n_seeds=3
- indolem__indobertweet-base-uncased__comment_only__len256__lr3p0em05__warm0p06__wd0p05__drop0p3__focal_loss__ls0p05: macro_f1=0.867393698211905, selection=0.8680517150907442, accuracy=0.8702485380116959, n_seeds=3
- indobenchmark__indobert-base-p2__context_sep_comment__len256__lr1p0em05__warm0p1__wd0p05__drop0p1__cross_entropy__ls0p05: macro_f1=0.8662542197076748, selection=0.8672039614978191, accuracy=0.8709795321637426, n_seeds=3
- indobenchmark__indobert-base-p2__context_sep_comment__len128__lr1p0em05__warm0p06__wd0p01__drop0p1__cross_entropy__ls0p0: macro_f1=0.8633635682626707, selection=0.8641116614447646, accuracy=0.8684210526315789, n_seeds=3

## Large Partial Metrics

- indobenchmark__indobert-large-p2__context_sep_comment__len128__lr8p0em06__warm0p06__wd0p01__drop0p1__cross_entropy__ls0p0 seed 42: folds=['1'], has_all_oof=False
