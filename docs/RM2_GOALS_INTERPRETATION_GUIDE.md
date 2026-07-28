# RM2 Goals Interpretation Guide

Generated at UTC: 2026-07-28T14:31:06.311393+00:00

IndoBERT V4 was audited as a candidate sentiment model, but the current strict decision is `INDOBERT_V4_NOT_ACCEPTED_KEEP_V2`. Until a candidate passes all locked-test gates, RM2 Goals remains based on the accepted V2 sentiment outputs.

Use these interpretation constraints in all RM2 reporting:

- IndoBERT V4, if accepted in a future run, would classify comment sentiment orientation into Negative, Neutral, and Positive.
- Goals are message orientations derived from aggregated comment sentiment, not evidence of intent, payment, or commercial relationships.
- HCC indicates groups of accounts with stronger structural connectivity, but it does not automatically prove buzzer activity or paid coordination.
- Community Actor means membership in an HCC from RM1, not a finding that an account is a bot, buzzer, or paid promoter.
- Sentiment is an RM2 attribute only. It must not change LCN, Louvain community membership, FSA_V, HCC membership, edges, nodes, or modularity from RM1.
- Brand and video context are context associations only, not sentiment labels and not evidence of brand involvement.

Canonical V2 goal mapping should remain unchanged unless a newly accepted sentiment model explicitly replaces the canonical model pointer through a separate validated run.
