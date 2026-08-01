# Figure and table inventory

Every artifact below regenerates from `cache/` alone:

```bash
python scripts/run_analysis.py all   # populates cache/figure_data and cache/tables
python figures/render_paper.py       # writes figures/generated/{ID}.pdf and .png
```

No number in any figure, table, or results file is entered by hand. Each figure
is written as both **PDF** (vector, for the paper) and **PNG** (for slides and
review). Tables are emitted as Markdown alongside the figures.

## Reading the visual language

| element | meaning |
|---|---|
| blue ordinal ramp, light → dark | the Vendi family ordered by q: VS_0, VS_0.1, VS_0.5, VS_1, VS_2, VS_∞ |
| orange | coverage — the pseudo log-determinant (never a member of the family) |
| aqua | facility location, the representativeness reference |
| grey band + dashed grey line | the 20-seed random baseline (5–95% band and mean) |
| dashed grey horizontal line | known ground truth, where one exists |
| shaded region around a curve | bootstrap or across-replicate uncertainty |
| pink | the ε = 1 log-determinant arm, kept only to show it is an artifact |

## Context and stratification

| ID | What it shows |
|---|---|
| **P-0a** | Snell hardness bin × MATH level occupancy, with Spearman ρ — confirms the two hardness definitions are correlated but not redundant |
| **P-0b** | Answer-entropy distribution per model and dataset |
| **P-0c** | **Effective-number axiom on synthetic pools with known ground truth.** Left: N balanced dissimilar classes score exactly N for every order q. Right: as one class takes over, the orders separate — low q holds near the class count, high q tracks the dominant mass. This is the property the raw embedding kernel violated on real pools |

## Measurement: how the two functionals behave (D1)

| ID | What it shows |
|---|---|
| **P-1a** | Rare-mode sensitivity. Synthetic sweep of a vanishing class, plus the real-pool effect of deleting the correct chains on minority/tail questions |
| **P-1b** | Redundancy, split by duplication regime. **Uniform** duplication leaves every functional exactly flat (harness T7); **skewed** duplication moves VS_q *and* coverage. Reporting only one regime would misstate what coverage does |
| **P-1c** | Sample-size behaviour: bias, standard deviation, and rank stability against subsample size n, per functional |
| **P-1d** | PCA dimensionality sweep: functional values, selector rank stability, and the same-vs-different answer cosine gap |
| **P-1e** | **The two log functionals.** Scatter of log VS_1 against coverage coloured by λ_min, plus the correlation-versus-q curve for within-budget, pooled, and the ε = 1 arm |
| **P-1f** | Reversal onset: pooled correlation against the λ_min decile, with the sign-flip threshold marked |

## The winner map (D2)

| ID | What it shows |
|---|---|
| **P-2a** | **Lead figure.** Accuracy against selection budget for greedy VS_1, greedy VS_∞, greedy coverage, and facility location, against the random band, under all three aggregation rules. Also emitted per model |
| **P-2b** | Hypothesis strip: H0–H6 each as one row with point estimate, CI, and verdict |
| **P-2c** | Coverage and VS_∞ deltas grouped by tail-heaviness stratum |
| **P-2d** | α threshold: accuracy against answer-awareness mixing, with α\* marked — the arm closest to the diversity-quality trade-off that Vendi-RAG shows works |
| **P-2e** | Four-panel tail conditioning: every objective's delta within modal / minority / tail / absent |
| **P-2f** | **q-inertness on the answer kernel.** At budgets at or below the number of distinct answers, every order q selects identically — the Similarity-Eigenvalue-Prevalence theorem made visible |
| **P-2g** | **What bounds the achievable gain.** Best delta against the winnable share (correct answer present but not modal) versus against raw headroom. The winnable share tracks it more closely, because headroom counts absent questions no selector can win |

## Payoff: verifier-free signals (D3)

| ID | What it shows |
|---|---|
| **P-3a** | Risk–coverage curves per signal: answer entropy, vote margin, mean logprob, embedding VS_1 |
| **P-3b** | The pooled-lift confound: embedding-diversity lift pooled across datasets versus within dataset |
| **P-3c** | Entropy-gated escalation: answered-set accuracy against fraction answered, and accuracy against total generated tokens |

## Relationships among the measures

| ID | What it shows |
|---|---|
| **P-4a** | **Correlation matrix** over every diversity order, coverage, answer entropy, vote margin, mean logprob, pass@1, and majority-vote correctness. All at one common subsample budget — see the note below |
| **P-4b** | Each functional scattered against question difficulty, annotated with its correlation |

## Appendix

| ID | What it shows |
|---|---|
| **P-A3** | τ threshold sensitivity for coverage: mean, median, and worst change, plus rank stability, at τ ∈ {1e-8, 1e-12} against the 1e-10 default |
| **P-A4** | **Why the embedding kernel is question-centred.** Spectral concentration, effective modes, and rate of identical selections across the raw, corpus-corrected, and question-centred kernels |
| **P-A4b** | Effect sizes under each kernel-correction arm |

## Tables

| ID | Contents |
|---|---|
| TB-0 | Stratum occupancy per cell; underpowered cells flagged |
| TB-1 | Smallest subsample n reaching rank stability τ ≥ 0.9, per functional |
| TB-2 | Within-budget and pooled correlations between log VS_q and coverage |
| TB-3 | **The winner map**: winning objective, delta, CI, and Holm-corrected p per cell, rule, and stratum |
| TB-4 | α\* per rule and objective |
| TB-5 | Risk–coverage AUC and lift per signal |
| TB-6 | Escalation operating points at 90 / 95 / 99% answered-set accuracy |
| TB-7 | Encoder rank stability (bge-large vs mxbai), per functional |
| TB-8 | Generation-seed variance across g ∈ {0, 1, 2} |

## Two conventions that matter for correctness

**Cross-question comparisons use one common budget.** Unparsed chains shrink
MATH pools unevenly, so each question's largest available subsample budget
ranges 128–1024. Comparing questions at their own maxima makes VS_0 an exact
proxy for pool size (corr = 1.000) and manufactures a −0.99 "anticorrelation"
with coverage. P-4a and P-4b therefore fix the budget to the largest value
available for at least 90% of a cell's questions.

**VS_0 is excluded from winner claims on continuous kernels.** Richness equals
the subset size whenever items are distinct, so every candidate ties and the
tie-break selects by index rather than content (verified: the eight
lowest-indexed chains on 20 of 20 pools). It remains meaningful on the answer
kernel, where exact ties make the count informative.
