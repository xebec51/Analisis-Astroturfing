# Positive-Recall Sentiment Annotation Codebook

Labels allowed for evaluable sentiment:

- Negative: complaint, disappointment, harm, distrust, rejection, or clearly unfavorable sentiment.
- Neutral: question, factual request, transaction/logistics, unclear stance, or informational comment without sentiment.
- Positive: support, recommendation, favorable experience, trust, satisfaction, or clearly favorable sentiment.

Non-evaluable labels:

- No Text: empty, deleted, or text cannot be evaluated.
- Uncertain: insufficient evidence, needs context that is not available, or annotators cannot adjudicate reliably.
- INJ: injected/synthetic diagnostic comment; exclude from training and locked tests.

Positive recall focus:

- Testimony may be Positive even without the word 'bagus' when it reports beneficial results.
- Implicit support can be Positive when the favorable stance is clear.
- Short recommendations can be Positive only when the recommendation is explicit enough.
- Emoji can support interpretation, but emoji alone should not override ambiguous text.

Forbidden shortcuts:

- Do not label all unknown comments as Positive.
- Do not convert all Neutral or Uncertain comments to Positive.
- Do not use HCC status, promotion suspicion, model prediction, lexicon, or goal orientation as ground truth.
- Do not infer sentiment from sampling reason; sampling reason is not a label.
