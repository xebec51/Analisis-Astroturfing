# RM2 Sentiment Human Annotation V2 Guideline

## Objective

This package collects additional human labels for RM2 comment-level sentiment analysis. The goal is to improve future diagnosis of Positive recall, Negative precision, questions about skin condition, and model coverage. Do not use model predictions, heuristic labels, HCC goal outputs, or prior automated results when labeling.

## Main Sentiment Labels

- Positive: use when the comment clearly praises, recommends, supports, promotes, or reports improvement. Example: "Aku cocok, bekas jerawat makin pudar."
- Neutral: use for questions, factual statements, tagging, product names only, price or purchase logistics without evaluation, or skin condition descriptions without product evaluation. Example: "Ini dipakai pagi atau malam?"
- Negative: use when the comment clearly complains about product or brand effect, safety, authenticity, value, or usage result. Example: "Setelah pakai ini wajahku perih dan merah."
- Uncertain: use when positive and negative signals are balanced or context is insufficient. Example: "Bagus sih, tapi di aku bikin kering."
- No Text: use when text is empty, deleted, unreadable, or only contains content that cannot be interpreted.

## Sentiment Target

- Product / Brand: the evaluation is directed at a skincare product or brand.
- Skin condition: the comment mainly describes acne, dullness, irritation, oiliness, or another skin condition.
- Usage question: the comment asks whether, when, or how to use a product or ingredient.
- Creator / Seller: the evaluation is directed at the creator, seller, service, or account.
- Price / Purchase: the comment concerns price, link, cart, checkout, shipping, stock, or availability.
- Promotion / CTA: the comment is mainly a recommendation, sales prompt, affiliate-style cue, or call to purchase.
- General discussion: general skincare conversation without a specific target.
- Other / unclear: the target cannot be determined.

## Complaint Scope

- product_effect: product effect or result is the object of complaint or praise.
- skin_condition: skin condition is mentioned as a problem without clear product causality.
- price_value: price, value, shipping, or purchase terms are central.
- safety_concern: irritation, ingredient safety, danger, or health risk is central.
- authenticity_concern: fake product, official store, originality, or trust is central.
- usage_confusion: sequence, frequency, compatibility, or how-to confusion is central.
- not_applicable: no complaint scope applies.
- unclear: scope cannot be determined.

## Decision Rules

1. Questions about skin condition without product evaluation tend to be Neutral with target Skin condition or Usage question.
2. Skin complaints are not automatically Negative toward Product / Brand. Mark Negative toward Product / Brand only when the comment links the bad effect to product usage.
3. Bad effects after product use can be Negative, target Product / Brand, scope product_effect or safety_concern.
4. Improvement, recommendation, support, and clear promotion can be Positive.
5. If positive and negative signals are equally strong, use Uncertain.
6. Emoji should not override the main text meaning.
7. Words such as jerawat, bruntusan, kusam, mahal, murah, aman, and cocok must be judged by sentence context.

## Examples

| Comment example | Sentiment | Target | Complaint scope | Note |
|---|---|---|---|---|
| "Kak ini aman buat kulit sensitif?" | Neutral | Usage question | usage_confusion | A question, not a product complaint. |
| "Jerawatku lagi parah banget" | Neutral | Skin condition | skin_condition | Skin condition without product causality. |
| "Pakai ini malah breakout" | Negative | Product / Brand | product_effect | Product effect is blamed. |
| "Aku cocok banget, jadi lebih cerah" | Positive | Product / Brand | product_effect | Clear improvement. |
| "Mahal tapi worth it" | Positive | Product / Brand | price_value | Positive dominates despite price mention. |
| "Bagus tapi bikin kering" | Uncertain | Product / Brand | product_effect | Mixed sentiment with no dominant polarity. |
| "Checkout sekarang, lagi promo" | Positive | Promotion / CTA | not_applicable | Clear promotional CTA. |
| "😂😂😂" | No Text | Other / unclear | unclear | Emoji-only with no interpretable sentiment. |
