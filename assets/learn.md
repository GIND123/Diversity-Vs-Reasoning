# Learn: diversity, coverage, and what this study actually did

A self-contained walkthrough. It assumes you know that the Vendi Score measures
diversity from a similarity matrix and roughly how it is computed, and builds
from there to every decision, result, and pitfall in this project.

**Contents**

1. [The measures, precisely](#1-the-measures-precisely)
2. [Why coverage is *not* part of the family](#2-why-coverage-is-not-part-of-the-family)
3. [The research question](#3-the-research-question)
4. [How the data was made](#4-how-the-data-was-made)
5. [The kernel problem — the study's central methodological finding](#5-the-kernel-problem)
6. [The statistical protocol](#6-the-statistical-protocol)
7. [Results: when does diversity help, when does coverage](#7-results)
8. [Every error we found, and what it teaches](#8-every-error-we-found)
9. [How to defend each claim](#9-how-to-defend-each-claim)
10. [References](#10-references)

---

## 1. The measures, precisely

### Hill numbers: diversity before similarity

Ecologists asked "how many species are in this ecosystem?" long before ML did.
For a probability vector `p = (p_1, ..., p_S)` over species, the **Hill number
of order q** is the exponential of the **Rényi entropy** of order q
[[Hill 1973](#refs)]:

```
H_q(p) = 1/(1-q) · log Σ_{i ∈ supp(p)} p_i^q
D_q(p) = exp( H_q(p) )
```

`q` controls how much rare species count:

| q | what it measures | behaviour |
|---|---|---|
| 0 | richness — just how many species exist | every species counts equally, prevalence ignored |
| 1 | exponential of Shannon entropy | each species weighted by its prevalence |
| 2 | inverse Simpson index | common species dominate |
| ∞ | 1 / (largest prevalence) | only the single most common species matters |

Crucially these are **effective numbers**: `N` equally abundant, completely
dissimilar species score exactly `N`, for every q. That is the interpretability
property the whole family rests on.

The limitation: Hill numbers assume you already know the species labels and
their prevalences, and that different species are *completely* dissimilar. Real
data is not like that — two reasoning chains can be 90% similar.

### The Vendi Score: diversity with similarity, no labels

The Vendi Score [[Friedman & Dieng 2023](#refs)] removes both assumptions. You
supply a similarity function `k(x, x')` with `k(x,x) = 1`. Build the similarity
matrix `K` over your `n` items, normalise it as `K/n`, take its eigenvalues
`λ_1, ..., λ_n` (which then sum to 1, because `trace(K)/n = 1`), and compute the
exponential of the Shannon entropy of that spectrum:

```
VS(x, k) = exp( − Σ λ_i log λ_i )
```

The eigenvalue spectrum plays the role the prevalence vector `p` played before —
but you never had to label anything.

### The Cousins: the order-q family

[[Pasarkar & Dieng 2024](#refs)] extends this to every Hill order. Their
**Theorem 4.1 (Similarity–Eigenvalue–Prevalence)** is the bridge: for a block
kernel where `N` groups of identical items have multiplicities `M_i` and
`C = Σ M_i`, the normalised kernel `K/C` has exactly `N` nonzero eigenvalues and
`λ_i = M_i / C`. The spectrum *is* the prevalence vector. So you can define, for
any q:

```
VS_q(x, k) = exp( 1/(1-q) · log Σ_{i ∈ supp(λ)} λ_i^q )        (Eq. 6)
```

Two properties this study checks empirically on real data:

```
VS_∞ ≤ ... ≤ VS_1 ≤ VS_0            (Eq. 7, monotone decreasing in q)
sqrt(VS_2) ≤ VS_∞ ≤ VS_2            (Eq. 8)
```

Both hold with **zero violations across 1,728 real chain spectra** in our runs.

### The two corrections that matter

Adji corrected two things early, and both are load-bearing:

> **"It's 'Rényi entropy' not 'von Neumann entropy'. The latter doesn't have a
> parameter q to be tuned, it corresponds to q = 1."**

The *original* Vendi Score is the von Neumann (equivalently Shannon-of-the-
spectrum) case. The **family** is Rényi. Saying "von Neumann entropy
parameterised by q" is a category error.

> **"Low q limit is still diversity and not coverage."**

`VS_0` is richness — the count of nonzero eigenvalues. It is a *diversity*
measure, the q → 0 member of the family. It is not coverage, not a volume, not
a different kind of quantity.

---

## 2. Why coverage is *not* part of the family

**Coverage**, in this study, is the **pseudo log-determinant**: the sum of the
logs of the *nonzero* eigenvalues of the same kernel.

```
coverage(x, k) = Σ_{i ∈ supp(λ)} log λ_i
```

This comes from the determinantal point process literature
[[Kulesza & Taskar 2012](#refs)], where `log det(K_S)` measures the **squared
volume** spanned by the selected items' feature vectors. Maximising it selects
sets that span a large region — "broad coverage" in the geometric sense.

Why it is genuinely a different animal from `VS_q`:

| | VS_q | coverage |
|---|---|---|
| functional form | exp of Rényi entropy of the spectrum | sum of logs of the spectrum |
| units | an effective *number* of items | a log *volume* (unbounded below) |
| bounded? | between 1 and n | goes to −∞ as any eigenvalue → 0 |
| duplicates | penalised (score falls) | zero eigenvalues are excluded, so exact duplicates are invisible |
| in Dieng's papers? | yes, it is the object of study | **no — the word "determinant" appears zero times in Cousins** |

### A terminology hazard you must handle explicitly

The word "coverage" is **overloaded**, and Adji's own papers use the *other*
sense. In *Cousins*, "coverage" appears nine times, always meaning the
**reference-based generative-evaluation metric** (the precision/recall/coverage
family that needs training data) [[Naeem et al. 2020](#refs)].

So in any write-up: say **"coverage (pseudo log-determinant)"** on first use, and
state that it is not the reference-based coverage metric. This is a naming
collision, not a disagreement — but left unflagged it reads as confusion about
her own literature.

---

## 3. The research question

> For each aspect of LLM reasoning, does selecting reasoning chains to maximise
> **diversity** (VS_q) or **coverage** (pseudo log-det) work better, and under
> what conditions?

The setup: a model answers a maths question 1024 times, producing 1024 chains of
reasoning. You can only afford to *use* k of them (say 8). Which 8?

- **Random** — pick 8 at random. This is the baseline in every comparison.
- **Greedy VS_q** — pick the 8 that maximise diversity at order q.
- **Greedy coverage** — pick the 8 that maximise the log-volume.
- **Facility location** — pick the 8 most *representative* (a different idea:
  coverage-of-the-set-by-proxies, not volume). Kept separate on purpose.

Then aggregate those 8 chains into an answer three ways:

- **majority vote** — the most common answer wins [[Wang et al. 2023](#refs)]
- **pass@k** — did *any* of the 8 get it right [[Chen et al. 2021](#refs)]
- **verifier best-of-n** — trust the highest-scoring chain

This matters practically because test-time compute is a real lever: sampling
more and choosing well can beat using a much larger model
[[Snell et al. 2024](#refs)].

---

## 4. How the data was made

| axis | value |
|---|---|
| models | Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-3B (pass@1 spanning 0.27 → 0.71) |
| datasets | GSM8K test [[Cobbe et al. 2021](#refs)], MATH levels 1–5 [[Hendrycks et al. 2021](#refs)] |
| chains | 1024 per question, temperature 1.0, top-p 0.95 |
| tokens | 400 (GSM8K), 1024 (MATH — see §8) |
| questions | 192 / 96 / 96 on GSM8K; 180 / 120 / 60 on MATH |
| hardware | one A100-40GB via Modal, vLLM |

Roughly **900,000 chains**, all published with embeddings and provenance.

Three kernels are built per question pool:

- **K_ans** — 1 if two chains reach the same answer, else 0. A block kernel, so
  Theorem 4.1 applies exactly.
- **K_emb** — cosine similarity of sentence embeddings of the chain text.
- **K_α = α·K_ans + (1−α)·K_emb** — a mixture, sweeping α from 0 to 1.

---

## 5. The kernel problem

**This is the study's most important methodological finding, and you should be
able to explain it cold.**

`VS_q` is a functional *of a kernel*. Choose a bad kernel and the number you get
is meaningless, however correctly you compute it.

### The symptom

We computed `VS_1` on real pools of 40 genuinely different reasoning chains and
got **≈ 1.4**. Recall the effective-number axiom: 1.0 means "all items are
identical". So the measure was reporting that 40 visibly different chains were
essentially one chain.

Diagnostics:

| quantity | raw K_emb |
|---|---|
| top eigenvalue's share of the spectrum | **0.94** |
| mean VS_1 among 40 distinct chains | **1.4** |
| rate at which two *different* objectives pick the identical set | 0.22 |

A kernel whose leading eigenvalue holds 94% of the mass is nearly rank-1: every
chain looks like every other. With such a kernel, the winner map is not
measuring diversity versus coverage — every objective picks nearly the same
chains, so the comparison is empty.

### The wrong explanation (which we tested and rejected)

The blueprint anticipated *anisotropy*: sentence-embedding spaces are known to
occupy a narrow cone, so the standard fix is to remove the top few directions
shared across the whole corpus. We implemented that sweep (c = 0,1,2,3).

**It barely helped.** VS_1 went 1.49 → 2.15 and then flattened. Corpus-wide
directions were not the cause.

### The right explanation

Every chain in a pool answers **the same question**. So the pool's dominant
direction is *that question's own content* — not an encoder artifact, and not
something shared across the corpus. With 96 different questions in a cell, no
small set of corpus directions can remove 96 different topics.

The fix is per-pool: re-express each chain as its **deviation from its own
question's centroid**. Then the kernel measures how chains differ *from each
other in solving this problem*, which is what "diversity of reasoning" means.

| arm | top-eig share | mean VS_1 | identical-selection rate |
|---|---|---|---|
| raw | 0.940 | 1.49 | 0.220 |
| corpus c=1 | 0.875 | 2.15 | 0.192 |
| corpus c=3 | 0.873 | 2.17 | 0.242 |
| **question-centred** | **0.514** | **9.29** | **0.060** |

Replicated on **3 models, 2 model families, 2 datasets** with nearly identical
numbers (raw top-eig 0.94–0.95 → 0.513–0.514 everywhere).

**Why this is a contribution, not just bookkeeping.** Anyone applying the Vendi
Score to "diversity of model outputs for one prompt" — reasoning chains, RAG
retrievals, code samples — hits this. The measure will read ≈1 and they will
conclude the outputs are not diverse, when in fact the measurement is being
swamped by the shared prompt. It follows directly from the effective-number
axiom, which is why the axiom check (figure P-0c) sits next to it.

---

## 6. The statistical protocol

Understanding these choices is how you defend the results.

**Paired bootstrap against random.** For each question, compute the objective's
outcome and random's outcome, take the difference, and bootstrap the mean of
those per-question differences (1000 replicates). Pairing removes
question-difficulty variance, which is far larger than the effect.

**Everything averaged over 5 subsample draws.** The 40-chain pool is drawn from
1024 with 5 different seeds; *both* the objectives and the random baseline are
evaluated on all five and averaged. Skipping this was a real bug — see §8.

**Holm correction within families**, with an explicit caveat: Holm controls
error *inside* a family of objectives for one (cell, rule, stratum), and the
sweep spans ~36 such families. So an isolated Holm-significant cell is weak
evidence. **The evidence standard used here is replication across models.**

**Practical-null threshold.** Any |Δ| < 0.01 is labelled practically null
regardless of p.

**Two stratifications**, both fixed before seeing outcomes:

- *Snell bins* — questions binned by the model's own pass@1, following the
  finding that the best test-time strategy depends on question difficulty
  [[Snell et al. 2024](#refs)].
- *Tail-heaviness* — where the correct answer ranks in the pool's answer
  distribution: **modal** (rank 1), **minority** (2–5), **tail** (>5),
  **absent** (never produced).

That second one turns out to explain the results, and it was pre-registered.

---

## 7. Results

### 7.1 The two functionals disagree depending on scope

Within a fixed budget, log VS_q and coverage correlate **strongly positively**
(r up to +0.998). Pooled across budgets, the correlation **flips negative**.
This is a Simpson's-paradox-style reversal: budget is a confounder driving both.

Two refinements we added:

- The reversal is **specific to continuous kernels**. On K_ans the two are
  anti-correlated at *every* scope.
- The `ε = 1` variant (`Σ log(1 + λ)`, sometimes used to avoid `log 0`) is
  **positive everywhere** — it destroys the effect and is not a substitute for
  the pseudo log-determinant.

### 7.2 The winner map

At k = 8 on the question-centred kernel, across 6 cells:

| finding | effect | evidence |
|---|---|---|
| VS_2 beats random on pass@k (Qwen-0.5B/GSM8K, 192q) | **+0.033** | CI [+0.014, +0.054], Holm p = 0.028 |
| VS_∞ beats random on pass@k (Qwen-1.5B/GSM8K) | +0.020 | CI [+0.006, +0.035] |
| **Coverage** helps pass@k on the **tail** stratum | **+0.068** | CI [+0.014, +0.121], n = 27 |
| VS_1 helps pass@k on MATH (Qwen-0.5B) | +0.039 | CI [+0.010, +0.075] |
| **Coverage** leads majority vote on MATH (Qwen-0.5B) | +0.034 | CI [−0.001, +0.071] |
| Everything on Llama-3.2-3B | ≈ 0 | all CIs include zero |

**The shape of the answer**: diversity orders tend to win **pass@k** (you want
to *hit* the answer, so spread out), while coverage does relatively better on
**majority vote** and on **tail-heavy** questions (where the answer is rare and
you need volume to reach it). This is the qualitative separation the study was
built to find — but the effects are small (2–7 points) and mostly not
Holm-significant on their own.

### 7.3 What bounds how much any selector can gain

Effects vanish on Llama, which at first looked like "stronger models don't
benefit". That explanation was **incomplete** — Llama/MATH has plenty of
headroom yet still shows nothing.

The better predictor is the **winnable share**: the fraction of questions whose
correct answer is *present in the pool but not the mode*.

- If the answer is already **modal**, majority vote gets it without help.
- If the answer is **absent**, no objective can recover it — a capability bound.
- Only **minority + tail** questions are contestable.

| predictor | corr with best pass@k gain |
|---|---|
| **winnable share** | **r = +0.61** |
| headroom (1 − random accuracy) | r = +0.48 |

Headroom overcounts, because absent questions inflate it while offering nothing
to win. This is descriptive across 6 cells, not an estimated law — but the
partition was fixed in advance, so it is not data dredging.

### 7.4 q-inertness on the answer kernel

On K_ans, at budgets at or below the number of distinct answers, **every order q
selects identically**. This is Theorem 4.1 made visible: the spectrum is the
answer-prevalence vector, and picking one chain per answer class maximises every
order simultaneously. Useful as a sanity check that the implementation is right.

### 7.5 VS_0 is unusable on continuous kernels

Three independent lines of evidence, all agreeing:

1. *Cousins* itself: with q = 0 "all species are given equal weight and D_0(p) is
   equal to the size of the support. **This is an uninformative measure of
   diversity.**"
2. Greedy VS_0 selection picked the **eight lowest-indexed chains on 20 of 20
   pools** — it is index selection, not content selection, because richness ties
   for every candidate when items are distinct.
3. Cross-encoder rank stability **τ = 0.34**, versus τ ≈ 0.83–0.90 for every
   other order and for coverage.

It remains meaningful on K_ans, where exact ties make the count informative.

### 7.6 Robustness

| check | result |
|---|---|
| generation-seed variance (g ∈ {0,1,2}, full regeneration) | pass@1 sd **0.0096**, tail-label agreement **0.88–0.92** |
| cross-encoder rank stability | τ ≈ 0.83–0.90 for coverage and q ≥ 0.1 |
| Eq. 7 monotonicity, Eq. 8 bound | **0 violations in 1,728 real spectra** |
| effective-number axiom on synthetic pools | exact for every q |

Seed noise (0.0096) is several times smaller than the effects (0.02–0.07), which
is the specific comparison a reviewer will want.

---

## 8. Every error we found

Each of these would have corrupted the results. They are worth knowing because
being able to say "we checked for X" is most of what makes a result credible.

**1. The symbolic answer oracle failed silently.** It ran sympy in a spawned
subprocess; `spawn` re-imports `__main__`, so in some entry points the worker
died and *every* equivalence check returned `False` — marking nearly every MATH
answer wrong, with no error. Fixed by evaluating in-process with a signal-based
timeout, plus a startup self-check that refuses to run if sympy cannot prove
`2/4 == 1/2`. Accuracy on hand-checked pairs went 2/10 → 9/10.

**2. Unequal noise between treatment and baseline.** Random was averaged over 20
draws; each objective got *one* arbitrary 40-chain subsample. On a 13-question
stratum this produced Δ = **+0.185, p = 0.001** at seed 0, **+0.104** at seed 1,
and **−0.169** at seed 2. The "significant" result was noise. Fixed by averaging
every arm over the same 5 draws.

**3. MATH chains were truncated.** At 400 tokens, **67.4% of MATH chains were
unparsed**, and 98.4% of those sat exactly at the cap. The surviving third were
exactly the short solutions — a length-biased pool that would bias every
diversity measurement. Raised to 1024 tokens and regenerated.

**4. A narrow run overwrote the shared question bank.** A 24-question
seed-variance run replaced the Hub's 96-question file. Fixed by making the
question file **merge by qid instead of overwrite**, and repairing the artifact.

**5. VS_0 appeared as a "winner".** See §7.5 — excluded from winner claims on
continuous kernels.

**6. Per-question budgets confounded a correlation.** Comparing each question at
*its own* largest budget made VS_0 an exact proxy for pool size (corr = 1.000)
and produced a spurious **−0.99** diversity–coverage anticorrelation on MATH. At
a common budget the relationship is consistent everywhere (+0.88 to +0.96).

---

## 9. How to defend each claim

Anticipated questions, and the answer:

**"Is this just generation noise?"** No — seed variance is 0.0096 in pass@1
against effects of 0.02–0.07, measured by fully regenerating banks at g = 1, 2.

**"Is it an artifact of your encoder?"** Rank stability across encoder families
is τ ≈ 0.83–0.90 for coverage and every q ≥ 0.1.

**"Did you implement VS_q correctly?"** 36 tests transcribe Eq. 1, Eq. 6, Eq. 7,
Eq. 8 and Theorem 4.1 directly from the paper and check our code against them;
plus Eq. 7 and 8 verified on 1,728 real spectra.

**"Your effects are tiny."** Correct, and stated plainly. They are bounded by the
winnable share, which we measure. Independent work reports the same shape: DPP-
based diverse selection gives "slightly higher diversity but insignificant
improvements in downstream performance" [[Deprez et al. 2026](#refs)], and
over-optimising a diversity objective can make it a poor downstream proxy
[[Bilmes et al. 2026](#refs)].

**"Why does facility location keep winning majority vote?"** Consistent with
[[Bilmes et al. 2026](#refs)], who find facility location performs best among
the submodular objectives they test. Representativeness amplifies the mode,
which is exactly what a vote rewards.

**"Why should diversity help at all?"** Because it is a *trade-off*, not a
maximisation: Vendi-RAG improves RAG by adaptively balancing diversity against
answer quality [[Rezaei & Dieng 2025](#refs)]. Our α sweep (mixing K_ans into
K_emb) is the analogue, and α\* is where that balance sits.

---

## 10. References

<a name="refs"></a>

- **Friedman & Dieng (2023).** *The Vendi Score: A Diversity Evaluation Metric
  for Machine Learning.* TMLR. [arXiv:2210.02410](https://arxiv.org/abs/2210.02410)
- **Pasarkar & Dieng (2024).** *Cousins of the Vendi Score: A Family of
  Similarity-Based Diversity Metrics for Science and Machine Learning.* AISTATS.
  [arXiv:2310.12952](https://arxiv.org/abs/2310.12952)
- **Rezaei & Dieng (2025).** *Vendi-RAG: Adaptively Trading-Off Diversity And
  Quality Significantly Improves Retrieval Augmented Generation With LLMs.*
  [arXiv:2502.11228](https://arxiv.org/abs/2502.11228)
- **Bilmes, Bhatt & Das (2026).** *How Much Is a Dataset Worth? Scaling Laws, the
  Vendi Score, and Matrix Spectral Functions.*
  [arXiv:2605.29448](https://arxiv.org/abs/2605.29448)
- **Deprez, Vandecappelle, Lauwereins & Goedemé (2026).** *Diversity by Chance:
  Rethinking the Need for Determinantal Point Processes in Active Learning.*
  SciTePress.
- **Hill (1973).** *Diversity and Evenness: A Unifying Notation and Its
  Consequences.* Ecology 54(2).
- **Leinster & Cobbold (2012).** *Measuring diversity: the importance of species
  similarity.* Ecology 93(3).
- **Kulesza & Taskar (2012).** *Determinantal Point Processes for Machine
  Learning.* [arXiv:1207.6083](https://arxiv.org/abs/1207.6083)
- **Naeem, Oh, Uh, Choi & Yoo (2020).** *Reliable Fidelity and Diversity Metrics
  for Generative Models.* ICML. (the *other* "coverage")
- **Wang et al. (2023).** *Self-Consistency Improves Chain of Thought Reasoning
  in Language Models.* ICLR. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- **Snell, Lee, Xu & Kumar (2024).** *Scaling LLM Test-Time Compute Optimally can
  be More Effective than Scaling Model Parameters.*
  [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- **Chen et al. (2021).** *Evaluating Large Language Models Trained on Code.*
  (pass@k) [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- **Cobbe et al. (2021).** *Training Verifiers to Solve Math Word Problems.*
  (GSM8K) [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- **Hendrycks et al. (2021).** *Measuring Mathematical Problem Solving With the
  MATH Dataset.* [arXiv:2103.03874](https://arxiv.org/abs/2103.03874)

Implementation: [vertaix/Vendi-Score](https://github.com/vertaix/Vendi-Score),
pinned to the commit recorded in `ENVIRONMENT.md`.
