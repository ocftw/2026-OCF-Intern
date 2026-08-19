# Metrics

## Official OmniDocBench v1.6 metrics

The main results are produced only by the pinned official Docker evaluator
using `end2end`, `quick_match`, the official ignore/attribute logic, and the
external immutable 1,651-page v1.6 JSON ground truth.

- Text normalized Edit Distance (lower is better)
- Formula CDM (higher is better)
- Table TEDS and evaluator-provided TEDS-S (higher is better)
- Reading Order Edit Distance (lower is better)
- Overall, exactly as emitted/defined by v1.6
- Available page/category/language/layout/attribute artifacts

Raw result directories are retained. The supervisor reproduction config is the
primary configuration. The generated immutable run config differs only in
mounted paths and conservative worker counts (match 4, CDM 3, TEDS 3); the
source config and its hash remain recorded.

## Supplementary OCR diagnostics — not leaderboard metrics

When official MGAM matching-pair artifacts expose text pairs, the adapter must
use those pairs. The pinned reproduction image currently does not document a
stable complete text-pair artifact. In that case the implementation reports
`method=diagnostic_page_flattened_visible_text`: non-ignored canonical GT
blocks are ordered by the official `order` field and compared with the complete
page prediction. This is diagnostic and does not replace official structured
evaluation.

Normalization is identical for both models:

1. Unicode NFC.
2. CRLF/CR become LF.
3. Remove only outermost whitespace.
4. Preserve punctuation, internal whitespace, Markdown, HTML tables, formulas,
   JSON-looking content, and repeated characters.

Definitions:

- EM: normalized prediction exactly equals normalized GT.
- CM: directional `gt in prediction`.
- ANLS: `1 - distance/max(lengths)`, set to zero below 0.5.
- Character F1: Unicode character multiset overlap precision/recall/F1.

Every aggregation includes its denominator, unmatched count and empty count.
Reports separate page macro averages from GT-character-weighted micro averages.
