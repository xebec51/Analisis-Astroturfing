# Sentiment AI Annotation Guideline

AI-assisted semantic adjudication labels comments as Positive, Neutral, Negative, Uncertain, or No Text.
Positive covers praise, suitability, support, satisfaction, recommendation, and negation of bad effects.
Negative covers complaints, adverse reactions, rejection, distrust, harmful price/value judgments, and product failure.
Neutral covers questions, factual information, tagging, product names, and comments without evaluative stance.
Uncertain covers unresolved mixed sentiment, sarcasm, very low-confidence semantics, or insufficient context.
No Text is reserved for comments without evaluable information; it is not equivalent to Neutral.

The two-pass kappa measures AI self-consistency, not human inter-annotator agreement. manual_label remains blank for future independent human validation.
