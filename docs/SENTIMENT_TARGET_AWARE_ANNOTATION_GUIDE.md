# Sentiment Target-Aware Annotation Guide V5

Status: `SENTIMENT_TARGET_AWARE_ANNOTATION_GUIDE_V5`

This guide defines the next human annotation pass for RM2 sentiment development. It does not change any V4 labels and does not reinterpret the already opened V4 locked test.

## Primary Label

The primary model label for RM2 Goals V5 is:

`sentiment_toward_target`

The target is the product, brand, or video context represented by `product_category` and `brand_or_video_context`. RM2 Goals should describe message orientation toward that target, not general emotion and not inferred actor intent.

`sentiment_overall` is collected as an auxiliary annotation for ambiguity analysis. It is not the canonical training label unless a separate preregistered protocol explicitly changes the target definition before training.

## Columns

Required V5 annotation columns:

- `sentiment_overall`
- `sentiment_toward_target`
- `target_brand`
- `mixed_sentiment_flag`
- `question_flag`
- `comparison_brand_flag`
- `sarcasm_or_irony_flag`
- `insufficient_context_flag`
- `annotator_1`
- `annotator_2`
- `adjudicated_label`
- `adjudication_note`

Allowed labels: `Negative`, `Neutral`, `Positive`, `Uncertain`, `No Text`.

Allowed flags: `Yes`, `No`.

## Ambiguity Rules

Positive versus Neutral:
A comment is Positive only when it expresses support, satisfaction, recommendation, praise, successful experience, or clearly favorable orientation toward the target. A plain question, purchase intent, tag, or request for information is Neutral unless it contains evaluative support.

Positive versus Negative:
Negation must be read carefully. A phrase such as “tidak bikin perih” can be Positive if it praises product safety or comfort. A phrase such as “bagus sih tapi bikin bruntusan” is mixed; choose the target-aware dominant orientation and mark `mixed_sentiment_flag=Yes`.

General sentiment versus target sentiment:
Annotate toward the target brand/product/video context. “Brand lain lebih bagus daripada ini” is Negative toward the target even if it praises another brand.

Buying interest versus praise:
“Mau coba” is usually Neutral because it states intent without evaluation. “Wajib beli” is Positive when it functions as a recommendation.

Short answers:
Short replies such as “cocok”, “bagus”, “worth it”, or “rekomen” can be Positive if the target context is clear. Short replies such as “spill”, “berapa”, or “link” are Neutral.

Brand comparison:
Mark `comparison_brand_flag=Yes` when two or more brands/products are compared. The label remains toward the target context, not necessarily toward every mentioned brand.

Mixed comments:
Use `mixed_sentiment_flag=Yes` when positive and negative evidence both matter. Do not average automatically; choose the dominant target-aware orientation or `Uncertain` if dominance cannot be resolved.

Irony:
Use `sarcasm_or_irony_flag=Yes` if sarcasm changes the literal polarity. If the sentiment cannot be determined, choose `Uncertain`.

Emoji-only comments:
Emoji can carry sentiment when target context is clear. If emoji meaning is unclear or too context-dependent, choose `Uncertain`; if there is no interpretable text or symbol, choose `No Text`.

## Prohibited Inference

Do not use HCC, actor type, network position, previous model predictions, brand involvement assumptions, hashtags alone, or suspected buzzer status as labels. Sentiment is only a comment-level message orientation attribute.
