# Alignment with the Vendi Score literature

An audit of this project against the papers that define the measures it uses.
Every claim below is enforced by `tests/correctness/test_paper_alignment.py`,
which transcribes the source equations and checks our code against them (36
tests). Run `make gate` to re-verify.

## Sources audited

| Paper | Role here |
|---|---|
| Friedman & Dieng, *The Vendi Score* (arXiv:2210.02410) | defines the q = 1 score as the exponential of the Shannon / von Neumann entropy of the kernel spectrum |
| Pasarkar & Dieng, *Cousins of the Vendi Score*, AISTATS 2024 (arXiv:2310.12952) | defines the order-q family, its axioms and inequalities — the paper Adji asked us to read |
| Rezaei & Dieng, *Vendi-RAG* (arXiv:2502.11228) | uses VS to trade diversity against answer quality in RAG |
| Bilmes, Bhatt & Das, *How Much Is a Dataset Worth?* (arXiv:2605.29448) | submodularity of matrix spectral functions; the greedy guarantee and the secular-equation speedup the blueprint cites |
| Deprez et al., *Diversity by Chance* (SciTePress 143492, 2026) | independent finding that DPP-based diverse selection barely beats random downstream |

## Definitions we are held to

**VS_q — Eq (6) of Cousins.**

    VS_q(x, k) = exp( 1/(1-q) * log sum_{i in supp(lambda)} lambda_i^q )

where `lambda` are the eigenvalues of the **normalized** similarity matrix and
`supp` is its nonzero eigenvalues. Verified against our
`spectra.functionals` for every q in {0, 0.1, 0.5, 1, 2, inf} on random kernels.

Two consequences we depend on and test:

- **It is Rényi, not von Neumann.** Eq (1) defines the Hill number as the
  exponential of the *Rényi* entropy of order q; von Neumann entropy is the
  q = 1 special case, i.e. the original Vendi Score. This is the correction Adji
  gave us, and it is now stated in `metrics.py`, `RESULTS.md`, and every figure
  caption that names the family.
- **Every order q is diversity.** The paper places q = 0 (richness) inside the
  same family. It is never coverage. Our low-q results are labelled diversity
  throughout.

**Theorem 4.1 (Similarity-Eigenvalue-Prevalence).** For a block kernel with
multiplicities `M_i` and `C = sum M_i`, the normalized kernel has exactly `N`
nonzero eigenvalues and `lambda_i = M_i / C`. Verified directly, and used as the
fast path in `functionals_from_counts`.

**Eq (7), monotonicity:** `VS_inf <= ... <= VS_1 <= VS_0`.
**Eq (8), order-2 bound:** `sqrt(VS_2) <= VS_inf <= VS_2`.
Both hold on synthetic kernels *and* on **1,728 real chain spectra** with zero
violations.

**Axiom 1 (effective number):** N equally abundant dissimilar items score N;
identical items score 1. Tested.

## Where our vocabulary must not collide with hers

**"Coverage" is overloaded, and we use the less common sense.** *Cousins* uses
"coverage" nine times, always meaning the **reference-based generative-model
evaluation metric** (the precision/recall/coverage family, which needs training
data). It never mentions determinants: "determinant", "DPP", "volume" and
"logdet" appear **zero** times in that paper.

Our "coverage" is the **pseudo log-determinant** — the sum of the logs of the
nonzero eigenvalues, a DPP-style log-volume — which is the definition Adji gave
directly for this project. Because the same word means something else in her
published work, every artifact here writes it as **"coverage (pseudo
log-determinant)"** on first use and states that it is not the reference-based
coverage metric. This is a naming hazard, not a disagreement.

## Where our findings sit relative to the literature

- **Small or null gains over random are the expected result, not a failure.**
  Deprez et al. find DPP-based diverse selection gives "slightly higher
  diversity but ... insignificant improvements in downstream performance", with
  random sampling competitive at a fraction of the cost. Our winner map says the
  same thing for LLM reasoning-chain selection, which makes our result a
  cross-domain replication rather than a disappointment.
- **Over-optimizing a diversity objective is known to backfire.** Bilmes et al.
  report that "pushing the objective to higher values can make it a poor
  downstream performance proxy". Our effects are small and shrink as model
  strength rises, which is consistent with that caution.
- **Facility location doing well is consistent with prior work.** Bilmes et al.
  find "facility location performs the best" among the objectives they test; it
  wins majority vote on the modal stratum for our weakest model. The blueprint
  already keeps facility location separate from coverage, which we preserve.
- **Diversity pays off when traded against quality, not maximized alone.**
  Vendi-RAG improves RAG by *adaptively* trading diversity against an LLM
  judge's quality score rather than maximizing diversity. That is the same shape
  as our `K_alpha` sweep, which mixes the answer kernel into the embedding
  kernel; reporting alpha* is therefore the part of our design closest to the
  method that is known to work.
- **q = 0 being uninformative is her result, not our artifact.** Cousins states
  plainly that with q = 0 "all species are given equal weight and D_0(p) is
  equal to the size of the support. This is an uninformative measure of
  diversity." Our independent finding that greedy VS_0 on a continuous kernel
  degenerates to index selection (it chose the eight lowest-indexed chains on
  20 of 20 pools) is exactly that statement showing up in a selection setting.

## Where we deviate, and why

| Deviation | Justification |
|---|---|
| Question-centred embedding kernel is primary | On the raw kernel, 40 lexically distinct chains score VS_1 = 1.4. That is **not** the score failing an axiom — distinct strings need not be dissimilar, and the score is correctly reporting that to the encoder every chain in a pool looks like a restatement of the same question. It does mean the raw kernel measures *which question is being answered* rather than *how the chains differ*, so it cannot answer a within-question question. Question-centring is an experimental representation targeting within-question variability, not a correction to VS; it restores VS_1 ~ 9. The raw and corpus-corrected arms are reported alongside as ablations (P-A4), and the choice is a limitation: our embedding numbers are not comparable to work scoring raw sentence-embedding similarity. |
| Batched greedy rather than the secular-equation update | Bilmes et al. report ~35,000x from secular-equation updates; we implement a simpler batched-eigendecomposition greedy (4-5x) that is *exactly* equivalent to naive greedy, cross-checked on 10 cases (T11). The speedup was sufficient for our pool sizes; the secular equation remains the route if pools grow. |
| MATH uses a 1024-token budget, GSM8K 400 | At 400 tokens 67.4% of MATH chains were truncations, leaving a length-biased pool. See `TRIAGE.md`. Datasets are never pooled in a headline claim, so differing budgets are safe. |
