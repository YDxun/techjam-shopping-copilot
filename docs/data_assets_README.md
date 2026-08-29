# TechJam2026 Shopping Copilot - Data Optimization Assets

## 1. Project Overview

This directory contains the static data assets produced for the
TechJam2026 Shopping Copilot Agent retrieval pipeline.

The optimization work focused only on assets with direct expected
contribution to retrieval quality, structured constraint matching,
private-set paraphrase robustness, and first-turn category routing. The
main goals were:

-   improve HR@10 / MRR through cleaner attribute normalization and
    field-aware matching;
-   reduce unnecessary search space and MTTC through category routing;
-   improve robustness to natural-language paraphrases using real
    customer review language;
-   preserve reproducibility by keeping official source data read-only
    and generating derived assets separately.

The final runtime assets are:

-   `vocab_v2_clean.json`
-   `field_mapping.json`
-   `review_paraphrases.json`
-   `category_mapping.json`

These four files have different responsibilities and should not be
merged blindly.

------------------------------------------------------------------------

## 2. Recommended Final Delivery Structure

``` text
data/
├── raw/                                  # Official/upstream data; read-only, not modified
│   ├── catalog.jsonl                     # If present in project
│   ├── public_set.jsonl                  # If present in project
│   ├── meta_Clothing_Shoes_and_Jewelry.jsonl
│   └── Clothing_Shoes_and_Jewelry.jsonl.gz
│
├── assets/                               # Final runtime assets
│   ├── vocab_v2_clean.json
│   ├── field_mapping.json
│   ├── review_paraphrases.json
│   └── category_mapping.json
│
├── analysis/                             # Audit/statistical outputs
│   ├── vocab_cleanup_report.json
│   ├── final_assets_report.json
│   ├── review_paraphrase_stats_full.json
│   ├── category_mapping_report.json
│   ├── category_stats.json
│   ├── attribute_candidates.json
│   ├── vocab_candidate_review.json
│   ├── field_mapping_stats.json          # If retained
│   ├── size_field_stats.json             # If retained
│   └── brand_field_stats.json            # If retained
│
└── intermediate/                         # Optional: reproducibility/debugging only
    ├── vocab_metadata_v2.json
    ├── vocab_metadata_v2_report.json
    ├── vocab_v2.json
    └── review_context_candidates_sample.json

scripts/
├── build_index.py                        # Existing project index builder, if applicable
├── build_size_stats.py
├── normalize_attribute_candidates.py
├── build_vocab_metadata_v2.py
├── extract_review_contexts.py
├── mine_reviews_full.py
├── build_final_assets.py
├── clean_vocab_v2.py
├── build_category_stats.py
└── build_category_mapping.py

tests/
└── data/
    └── adversarial_public_set.jsonl       # Optional evaluation asset; not required at runtime

README.md
```

### Submission packaging note

Do **not** put the 17 GB upstream metadata or 7 GB review corpus into
the submission package unless the competition explicitly requires them.
They are source corpora used to derive the compact runtime assets.

For the Agent itself, the four files under `data/assets/` are the
important deliverables.

------------------------------------------------------------------------

## 3. Final Runtime Assets

### 3.1 `vocab_v2_clean.json`

**Purpose:** canonical attribute vocabulary and safe lexical
normalization.

It is the final vocabulary derived from the original vocab plus
structured metadata evidence and conservative review-derived literal
aliases.

Typical responsibilities:

``` text
cotton / polyester / stainless steel  -> material canonical values
navy / rose gold / gray               -> color canonical values
xl / xxl / 3xl                        -> size canonical values
real leather -> leather               -> safe lexical alias
grey -> gray                           -> spelling normalization
```

The final cleanup removes confirmed legacy `public_set` extraction noise
only when the term is explicitly classified as noise and has no
catalog/structured-metadata support.

Examples of removed noise include terms such as:

``` text
comfortable
quality
material
colors
shoes
socks
super
```

The cleanup removed 68 confirmed noisy canonicals: 43 Material entries
and 25 Color entries.

**Runtime rule:** use this file for literal/canonical normalization. Do
not treat every synonym as a hard semantic equivalence without
considering attribute context.

------------------------------------------------------------------------

### 3.2 `field_mapping.json`

**Purpose:** tells retrieval and structured filtering **where to look**
for each constraint and how strict matching should be.

Conceptually:

``` json
{
  "material": {
    "lookup_fields": [
      "title",
      "features",
      "details.Material"
    ],
    "bm25_weight": {
      "title": 5.0,
      "features": 3.0,
      "details.Material": 6.0
    },
    "filter_mode": "soft"
  }
}
```

Core design:

-   `material`: prefer structured Material fields when present, while
    retaining title/features recall.
-   `color`: search structured Color plus title/features.
-   `size`: use context-aware matching; avoid confusing physical
    dimensions with wearable sizes.
-   `budget`: only use `price`; missing price must be handled leniently
    rather than automatically rejecting the item.
-   `brand`: `store` is the primary high-coverage brand signal, with
    structured Brand fields as supporting evidence.

**Runtime rule:** this asset should drive structured filtering and field
weighting rather than performing a full-text search for every
constraint.

------------------------------------------------------------------------

### 3.3 `review_paraphrases.json`

**Purpose:** natural-language interpretation learned from the full
customer review corpus.

The full review mining pass processed:

``` text
66,033,346 reviews
0 malformed JSON lines
62,175,766 verified purchases
94.16% verified-purchase rate
```

The most useful signal was size/fit language:

``` text
true_to_size
runs_small
runs_large
size_up
size_down
tight_or_snug
loose
```

These expressions are intentionally separated from literal size values.

For example:

``` text
"runs small" != size = "S"
"size up"    != size = "XL"
```

Instead, they represent fit/query intent.

The asset also contains conservative material language and
color-language rules, including:

``` text
real leather / genuine leather
made of {material}
dark navy -> navy
muted purple -> purple
bluish black -> soft expansion only
```

**Runtime rule:** use this asset for query interpretation and soft
expansion. Do not blindly insert all review phrases into
`vocab_v2_clean.json`.

------------------------------------------------------------------------

### 3.4 `category_mapping.json`

**Purpose:** normalize noisy Amazon category hierarchies into stable
coarse categories for first-turn routing.

The category builder uses the full category path rather than leaf-only
mapping:

``` text
Women > Clothing > Dresses > Formal
    -> women_dresses

Boys > Shoes > Sneakers
    -> boys_sneakers
```

The final V2 mapping statistics are:

``` text
Input path frequency:     7,218,331
Mapped path frequency:    6,650,845
Unmapped path frequency:    362,903
Noise-only frequency:       204,583

Mapped rate:                92.14%
Unmapped rate:               5.03%
Noise-only rate:             2.83%

Canonical categories:          392
Known path mappings:         1,757
```

The remaining high-frequency unmapped paths are intentionally dominated
by broad/ambiguous nodes such as:

``` text
Women
Women > Clothing
Men
Men > Clothing
Girls
Women > Accessories
Women > Jewelry
```

These are not reliable product families and therefore should **not** be
converted into hard filters merely to increase mapping coverage.

The category pipeline also filters merchandising/infrastructure nodes
such as promotion, sale, test, Prime Wardrobe, Shopbop, MFN and similar
taxonomy noise.

**Runtime rule:** if category mapping is confident, use it for coarse
routing. If a path remains broad or ambiguous, fall back to
lexical/semantic retrieval rather than hard filtering.

------------------------------------------------------------------------

## 4. Data Optimization Pipeline

The complete optimization process was:

``` text
Original vocab
     |
     v
Structured metadata inspection
     |
     +--> field coverage / Brand / Size analysis
     |
     +--> attribute candidate extraction
     |
     v
vocab_candidate_review.json
     |
     v
Conservative metadata vocabulary expansion
     |
     v
vocab_metadata_v2.json
     |
     +------------------------------+
     |                              |
     v                              v
66M review mining              Category statistics
     |                              |
     v                              v
review paraphrase stats       category_stats.json
     |                              |
     v                              v
safe review aliases           path-aware category mapping
     |                              |
     v                              v
vocab_v2.json                 category_mapping.json
     |
     v
legacy public_set noise cleanup
     |
     v
vocab_v2_clean.json
```

The source corpora remain read-only throughout this process.

------------------------------------------------------------------------

## 5. Metadata Optimization

### 5.1 Attribute-field analysis

The metadata stage was used to determine where attributes actually
appear instead of assuming a schema from individual examples.

This was necessary because a single product having no
`details.Material`, `details.Color`, or `details.Size` does not imply
that those fields are absent from the corpus.

The analysis therefore focused on corpus-level coverage and consistency.

### 5.2 Brand

Brand inspection showed that `store` has very high coverage and agrees
strongly with structured Brand fields when those fields are present.

This led to the routing decision:

``` text
brand -> store first
         + details.Brand
         + details.Brand Name
         + title as supporting text
```

### 5.3 Size

Size required special handling because structured `details.Size` can
contain both wearable sizes and physical dimensions.

Examples of physical dimensions must not become apparel sizes:

``` text
8 inch
10 inch
pack of 1
```

Therefore size matching should remain context-aware.

### 5.4 Material / Color

Structured metadata was used as the primary evidence for adding new
literal attribute values.

The pipeline did not automatically accept every frequent string.
Candidates were separated into:

``` text
AUTO_ACCEPT
REVIEW
REJECT
```

Only conservative, metadata-backed values were promoted into the final
vocabulary.

------------------------------------------------------------------------

## 6. Review-Corpus Optimization

A 1,000,000-review pilot was run before the full scan.

The pilot demonstrated two things:

1.  real review language contains strong paraphrase signals;
2.  noisy legacy vocab anchors can create huge false-positive counts.

For example, words such as `comfortable` and `quality` appeared
frequently in reviews but are not Material values.

Therefore the full 66M review pass used a restricted strategy:

-   trusted Material anchors;
-   trusted Color anchors;
-   explicit size/fit phrase patterns;
-   negation tracking;
-   bounded examples;
-   frequency aggregation instead of storing the full corpus.

Review statistics are evidence for language usage, **not automatically
product truth**.

For example, `felt` is both a material and the past tense of "feel", so
raw review frequency for `felt` cannot be treated as reliable material
evidence.

------------------------------------------------------------------------

## 7. Category Optimization

The category stage processed the complete metadata category
distribution.

Key design decision:

> Category normalization is path-aware, not leaf-only.

This is required because audience and product family depend on
hierarchy.

The V2 audience detector also prioritizes specific audience nodes:

``` text
Baby > Baby Girls -> baby_girls
Baby > Baby Boys  -> baby_boys
Kids & Baby > Girls -> girls
Kids & Baby > Boys  -> boys
```

The category taxonomy is intentionally coarse. The goal is retrieval
routing, not reproduction of Amazon's complete browse-node tree.

------------------------------------------------------------------------

## 8. Asset Responsibilities

  --------------------------------------------------------------------------------------------
  Asset                       Primary role                Hard filtering?     Query expansion?
  --------------------------- ---------------------- -------------------- --------------------
  `vocab_v2_clean.json`       Canonical attribute       Context-dependent                  Yes
                              normalization                               

  `field_mapping.json`        Field routing,                          Yes                   No
                              weights, strictness                         

  `review_paraphrases.json`   Natural-language/fit             Usually no                  Yes
                              interpretation                              

  `category_mapping.json`     Coarse category         Only when confident         Yes/fallback
                              routing                                     
  --------------------------------------------------------------------------------------------

A key principle is **separation of semantics**.

Do not collapse all four assets into a single synonym dictionary.

------------------------------------------------------------------------

## 9. Recommended Retriever Integration

Recommended processing order:

``` text
User query
   |
   v
1. Category interpretation
   |-- category_mapping.json
   |
   v
2. Attribute/value normalization
   |-- vocab_v2_clean.json
   |
   v
3. Natural-language intent interpretation
   |-- review_paraphrases.json
   |
   v
4. Field-aware candidate retrieval/filtering
   |-- field_mapping.json
   |
   v
5. BM25 / embedding / fusion / reranking
```

### Suggested behavior

**Category**

Use category mapping as an early routing signal. If the mapping is broad
or uncertain, degrade gracefully to lexical/semantic retrieval.

**Material / Color / Brand**

Use `field_mapping.json` to prioritize the appropriate structured fields
and text fields.

**Budget**

Missing price should not automatically reject a product if the
configured policy is lenient.

**Size**

Use context-aware size matching. Numeric tokens require category/size
context.

**Review paraphrases**

Use fit expressions as intent signals rather than literal structured
values.

------------------------------------------------------------------------

## 10. Known Ambiguities

The final vocab audit may still contain semantic ownership overlaps
inherited from earlier taxonomy decisions, such as:

``` text
pu leather
tpu
eva
golden
numeric shoe sizes
gym
```

These should not be resolved by a global last-write-wins dictionary.

Recommended runtime behavior:

-   keep attribute-specific term ownership;
-   use category context for numeric sizes;
-   use field context for ambiguous material/color terms;
-   use soft matching when semantic equivalence is uncertain.

------------------------------------------------------------------------

## 11. Files for Runtime vs. Audit

### Runtime - required

``` text
vocab_v2_clean.json
field_mapping.json
review_paraphrases.json
category_mapping.json
```

### Audit - recommended to retain

``` text
vocab_cleanup_report.json
final_assets_report.json
review_paraphrase_stats_full.json
category_mapping_report.json
category_stats.json
attribute_candidates.json
vocab_candidate_review.json
```

### Intermediate - not loaded by production Agent

``` text
vocab_metadata_v2.json
vocab_metadata_v2_report.json
vocab_v2.json
review_context_candidates_sample.json
```

------------------------------------------------------------------------

## 12. Reproducibility Rules

1.  Official/upstream data must remain read-only.
2.  Derived assets must be generated into separate paths.
3.  Do not manually overwrite raw corpora.
4.  Keep reports together with final assets so every vocabulary/category
    change is auditable.
5.  Large source corpora should not be bundled into the runtime
    submission unless explicitly required.
6.  Prefer deterministic scripts and versioned outputs.
7.  When changing normalization rules, regenerate the corresponding
    report and compare coverage/conflicts before replacing the runtime
    asset.

------------------------------------------------------------------------

## 13. Completion Status

Core data-asset work:

``` text
Field mapping / field-aware filtering       COMPLETE
Metadata attribute vocabulary expansion     COMPLETE
Full review paraphrase mining               COMPLETE
Legacy vocab noise cleanup                  COMPLETE
Private-set paraphrase robustness asset     COMPLETE
Path-aware category normalization           COMPLETE
```

Final runtime asset set:

``` text
vocab_v2_clean.json
field_mapping.json
review_paraphrases.json
category_mapping.json
```

The optional adversarial rewrite set for the public 200 examples is an
**evaluation asset**, not a required runtime dependency. It can be
generated separately if the team wants a dedicated paraphrase stress
test.

------------------------------------------------------------------------

## 14. Next Stage

The data optimization stage is complete.

The next engineering stage is integration and parameter search:

``` text
1. Load the four runtime assets in the retrieval pipeline.
2. Implement attribute-specific normalization.
3. Implement field-aware structured filtering.
4. Add category routing with safe fallback.
5. Add review-paraphrase query interpretation.
6. Tune BM25 field weights.
7. Tune lexical/embedding fusion alpha.
8. Tune hard/soft/lenient filtering behavior.
9. Evaluate HR@10, MRR and MTTC on the public set.
10. Use paraphrase stress tests to check private-set robustness.
```

The four runtime assets should be treated as **inputs to retrieval
tuning**, not as the end of retrieval optimization.
