"""Per-question chain pools: answers, equivalence classes, kernels, subsamples.

A ``Pool`` is the unit every analysis stage consumes. It contains only parsed
chains — the blueprint excludes unparsed chains from all selectors and all
aggregations symmetrically — and records how many were dropped.

MATH answer-equivalence classes use one sympy canonicalization per distinct
answer string; correctness of each class against the gold reference uses the
blueprint's ``simplify(a - b) == 0`` semantics with the 5 s timeout, both served
by a persistent :class:`MathCanonicalizer` oracle (string-identity fallback on
timeout, per blueprint B2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .artifacts import embedding_path, subsample_indices
from .datasets import Question
from .hf_bank import read_cached_bank
from .kernels import answer_kernel, embedding_kernel, l2_normalize, mixed_kernel
from .prompts import dataset_key

FloatArray = NDArray[np.float64]


class OracleUnavailable(RuntimeError):
    """The symbolic oracle cannot evaluate, so MATH results would be wrong."""


class _Timeout(Exception):
    pass


class MathCanonicalizer:
    """Symbolic answer oracle: sympy in-process, with a per-query time limit.

    Evaluated in-process on purpose. An earlier design ran sympy in a spawned
    worker; ``spawn`` re-imports ``__main__``, so in any entry point without a
    main guard the worker died and every query fell back to string comparison —
    silently marking nearly every MATH answer wrong. In-process evaluation
    removes that whole failure class, and :meth:`self_check` refuses to start
    when sympy cannot answer a pair whose equivalence is known in advance, so a
    broken oracle fails loudly instead of quietly corrupting a cell.

    Timeouts use ``SIGALRM`` (main thread, Unix). Where unavailable the limit is
    not enforced and ``timeouts_unenforced`` records that for the manifest.
    """

    def __init__(self, timeout_seconds: float = 5.0, *, self_check: bool = True) -> None:
        self.timeout_seconds = timeout_seconds
        self.failures = 0
        self.timeouts = 0
        self.timeouts_unenforced = False
        self._canon_cache: Dict[str, str] = {}
        self._equiv_cache: Dict[Tuple[str, str], bool] = {}
        if self_check:
            self.self_check()

    def self_check(self) -> None:
        """Fail fast if the oracle cannot do symbolic work at all."""
        if not self.equivalent(r"\frac{2}{4}", r"\frac{1}{2}", _checked=True):
            raise OracleUnavailable(
                "sympy could not prove 2/4 == 1/2; MATH equivalence would silently "
                "degrade to string matching. Check the sympy install."
            )

    def _evaluate(self, operation: str, payload: Any) -> Any:
        import signal

        import sympy

        def _raise(signum: Any, frame: Any) -> None:
            raise _Timeout

        armed = False
        if hasattr(signal, "SIGALRM"):
            try:
                signal.signal(signal.SIGALRM, _raise)
                signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
                armed = True
            except ValueError:  # not the main thread
                self.timeouts_unenforced = True
        else:
            self.timeouts_unenforced = True
        try:
            if operation == "canon":
                return _canonical_form(sympy, sympy.simplify(_sympify(sympy, payload)))
            left, right = payload
            first, second = _sympify(sympy, left), _sympify(sympy, right)
            difference = sympy.simplify(first - second)
            if difference == 0:
                return True
            # Exact symbols never simplify to zero against a truncated decimal
            # (2*pi vs 6.283185307179586), yet MATH gold answers are routinely
            # written either way. Fall back to numeric agreement at a tolerance
            # far tighter than any rounding a solution would use, so 1/3 and
            # 0.333 stay distinct.
            if first.is_number and second.is_number:
                try:
                    return bool(abs(sympy.N(difference, 20)) < 1e-9)
                except (TypeError, ValueError):
                    return False
            return False
        except _Timeout:
            self.timeouts += 1
            return None
        except Exception:  # noqa: BLE001 - unparseable answers are expected
            self.failures += 1
            return None
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)

    def canonical(self, value: str) -> str:
        """Canonical sympy form of one answer; the string itself on failure."""
        if value not in self._canon_cache:
            result = self._evaluate("canon", value)
            self._canon_cache[value] = str(result) if result is not None else value
        return self._canon_cache[value]

    def equivalent(self, left: str, right: str, *, _checked: bool = False) -> bool:
        """Blueprint equivalence: simplify(a-b)==0, string match on failure."""
        if left == right:
            return True
        key = (left, right) if left <= right else (right, left)
        if key not in self._equiv_cache:
            result = self._evaluate("equiv", key)
            self._equiv_cache[key] = bool(result) if result is not None else False
        return self._equiv_cache[key]

    def stats(self) -> Dict[str, Any]:
        return {
            "canonical_cached": len(self._canon_cache),
            "equivalence_cached": len(self._equiv_cache),
            "failures": self.failures,
            "timeouts": self.timeouts,
            "timeouts_unenforced": self.timeouts_unenforced,
        }

    def close(self) -> None:  # retained for call-site compatibility
        return None


def _canonical_form(sympy: Any, expression: Any) -> str:
    """A class key that is identical for every equivalent representation.

    ``srepr`` alone is not usable as an equivalence-class key: 1/2 and 0.5
    simplify to ``Rational(1, 2)`` and ``Float('0.5')``, which are equal in
    value but different strings, so the same answer would split across two
    classes and inflate every answer-kernel statistic. Numbers are therefore
    keyed by a fixed-precision decimal and only symbolic results fall back to
    ``srepr``.
    """
    if expression.is_number:
        try:
            return f"num:{sympy.N(expression, 15)}"
        except Exception:  # noqa: BLE001 - non-evaluable numbers keep srepr
            pass
    return f"sym:{sympy.srepr(expression)}"


def _sympify(sympy: Any, value: str) -> Any:
    """Parse one normalized answer, converting the LaTeX forms sympy misses.

    Uses sympy's implicit-multiplication transformations so ordinary algebraic
    answers such as ``2x`` parse the way a reader expects.
    """
    import re as _re

    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    text = value.replace("^", "**")
    text = _re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = _re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = _re.sub(r"\\sqrt(\w)", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\left", "").replace("\\right", "").replace("\\%", "")
    text = text.replace("{", "(").replace("}", ")").replace("\\", "")
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(text, transformations=transformations, evaluate=True)


@dataclass
class Pool:
    """All parsed chains of one question, with everything analysis needs."""

    qid: str
    dataset: str
    model: str
    reference_answer: str
    answers: List[str]
    class_ids: NDArray[np.int64]
    correct: NDArray[np.bool_]
    logprob_sums: FloatArray
    token_counts: NDArray[np.int64]
    embeddings: Optional[FloatArray] = None
    n_unparsed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    corpus_mean: Optional[FloatArray] = None
    corpus_directions: Optional[FloatArray] = None
    _kernel_cache: Dict[str, FloatArray] = field(default_factory=dict, repr=False)
    _embedding_cache: Dict[str, FloatArray] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return len(self.answers)

    @property
    def pass_at_1(self) -> float:
        return float(self.correct.mean()) if self.size else 0.0

    def answer_counts(self) -> Dict[int, int]:
        values, counts = np.unique(self.class_ids, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def question_centered_embeddings(self, components: int = 1) -> FloatArray:
        """Embeddings with this question's *own* leading directions removed.

        Distinct from :meth:`adjusted_embeddings`, which removes directions
        shared across the whole corpus. Chains answering one question are
        genuinely similar, so a large part of a pool's spectral concentration is
        question-specific rather than an encoder artifact; this arm measures
        chains as deviations from their own question's centroid and separates
        the two effects.
        """
        if self.embeddings is None:
            raise ValueError(f"Pool {self.qid} has no embeddings")
        # Cached: this SVD is O(n d^2) on a 1024 x 1024 matrix and the sample-size
        # sweep asks for it thousands of times per pool.
        key = f"qc{components}"
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        values = np.asarray(self.embeddings, dtype=np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        if components > 0 and min(centered.shape) > components:
            _, _, right = np.linalg.svd(centered, full_matrices=False)
            basis = right[:components]
            centered = centered - (centered @ basis.T) @ basis
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            centered = centered + 1e-12
        result = l2_normalize(centered)
        result.flags.writeable = False  # shared cache entry; never mutate in place
        self._embedding_cache[key] = result
        return result

    def adjusted_embeddings(self, components: int = 0) -> FloatArray:
        """Embeddings with the top ``components`` common corpus directions removed.

        Directions are fitted once on the whole (model, dataset) chain corpus, as
        the blueprint specifies, and shared by every pool in the cell. Raw
        sentence-embedding spaces are strongly anisotropic: one shared direction
        typically carries most of the spectral mass, which drives every pool's
        Vendi score toward 1 and makes all selectors agree. Removing it is what
        makes the embedding kernel informative.
        """
        if self.embeddings is None:
            raise ValueError(f"Pool {self.qid} has no embeddings")
        key = f"corpus{components}"
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        if components == 0:
            result = l2_normalize(self.embeddings)
            result.flags.writeable = False
            self._embedding_cache[key] = result
            return result
        if self.corpus_directions is None or self.corpus_mean is None:
            raise ValueError(f"Pool {self.qid} has no fitted corpus directions")
        if components > self.corpus_directions.shape[0]:
            raise ValueError(
                f"Only {self.corpus_directions.shape[0]} corpus directions were fitted"
            )
        basis = self.corpus_directions[:components]
        centered = np.asarray(self.embeddings, dtype=np.float64) - self.corpus_mean
        adjusted = centered - (centered @ basis.T) @ basis
        result = l2_normalize(adjusted)
        result.flags.writeable = False
        self._embedding_cache[key] = result
        return result

    def kernel(self, family: str, alpha: float = 0.5, components: int = 0) -> FloatArray:
        """Return K_emb, K_ans, or K_alpha for the full pool, cached.

        ``components`` removes that many common corpus directions from the
        embeddings first; it has no effect on the answer kernel.
        """
        if family == "answer":
            key = "answer"
        elif family in {"embedding", "embedding_qc"}:
            key = f"{family}:c{components}"
        else:
            key = f"{family}:{alpha:g}:c{components}"
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        if family == "answer":
            kernel = answer_kernel([str(c) for c in self.class_ids])
        elif family == "embedding_qc":
            kernel, _ = embedding_kernel(self.question_centered_embeddings(max(1, components)))
        elif family == "embedding":
            kernel, _ = embedding_kernel(self.adjusted_embeddings(components))
        elif family == "mixed":
            kernel = mixed_kernel(
                self.kernel("answer"), self.kernel("embedding", components=components), alpha
            )
        else:
            raise ValueError(f"Unknown kernel family {family!r}")
        self._kernel_cache[key] = kernel
        return kernel

    def release_caches(self) -> None:
        """Drop cached kernels and adjusted embeddings for this pool.

        Caches make a single pool's sweep fast, but they are per-pool and each
        1024-chain pool holds ~75 MB of kernels once every variant is built.
        Retaining them across a 192-question cell needs ~16 GB and the machine
        starts swapping — one cell took 30x longer than the 96-question run for
        twice the work. Stages that walk pools once release each pool as they
        finish with it.
        """
        self._kernel_cache.clear()
        self._embedding_cache.clear()

    def subsample(self, budget: int, seed: int) -> List[int]:
        """Seeded budget subsample of parsed-chain indices (sorted)."""
        return subsample_indices(self.size, budget, seed)

    def mean_logprobs(self) -> FloatArray:
        """Mean token logprob per chain, the cautionary verifier arm."""
        with np.errstate(invalid="ignore", divide="ignore"):
            values = self.logprob_sums / np.maximum(self.token_counts, 1)
        return np.asarray(np.nan_to_num(values, nan=-1e9), dtype=np.float64)


def _assign_classes(
    dataset: str,
    answers: Sequence[str],
    canonicalizer: Optional[MathCanonicalizer],
) -> Tuple[NDArray[np.int64], Dict[int, str]]:
    """Group parsed answers into equivalence classes; returns ids + representatives."""
    key = dataset_key(dataset)
    forms: List[str] = []
    if key == "math" and canonicalizer is not None:
        forms = [canonicalizer.canonical(answer) for answer in answers]
    else:
        forms = list(answers)
    mapping: Dict[str, int] = {}
    representatives: Dict[int, str] = {}
    ids = np.empty(len(forms), dtype=np.int64)
    for index, (form, answer) in enumerate(zip(forms, answers)):
        if form not in mapping:
            mapping[form] = len(mapping)
            representatives[mapping[form]] = answer
        ids[index] = mapping[form]
    return ids, representatives


def _correct_flags(
    dataset: str,
    class_ids: NDArray[np.int64],
    representatives: Mapping[int, str],
    reference: str,
    canonicalizer: Optional[MathCanonicalizer],
) -> NDArray[np.bool_]:
    key = dataset_key(dataset)
    verdicts: Dict[int, bool] = {}
    for class_id, representative in representatives.items():
        if key == "gsm8k" or canonicalizer is None:
            verdicts[class_id] = representative == reference
        else:
            verdicts[class_id] = canonicalizer.equivalent(representative, reference)
    return np.asarray([verdicts[int(c)] for c in class_ids], dtype=bool)


def build_pool(
    qid: str,
    dataset: str,
    model: str,
    rows: Sequence[Mapping[str, Any]],
    reference: str,
    *,
    embeddings: Optional[FloatArray] = None,
    canonicalizer: Optional[MathCanonicalizer] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Pool:
    """Assemble one pool from cached chain rows (and aligned embeddings)."""
    parsed_indices = [index for index, row in enumerate(rows) if row.get("parsed")]
    answers = [str(rows[index]["answer"]) for index in parsed_indices]
    class_ids, representatives = _assign_classes(dataset, answers, canonicalizer)
    correct = _correct_flags(dataset, class_ids, representatives, reference, canonicalizer)
    logprobs = np.asarray(
        [float(rows[index].get("logprob_sum") or 0.0) for index in parsed_indices],
        dtype=np.float64,
    )
    tokens = np.asarray(
        [int(rows[index].get("token_count") or 0) for index in parsed_indices], dtype=np.int64
    )
    pool_embeddings: Optional[FloatArray] = None
    if embeddings is not None:
        if embeddings.shape[0] != len(rows):
            raise ValueError(f"{qid}: embeddings rows {embeddings.shape[0]} != chains {len(rows)}")
        pool_embeddings = np.asarray(embeddings[parsed_indices], dtype=np.float64)
    return Pool(
        qid=qid,
        dataset=dataset_key(dataset),
        model=model,
        reference_answer=reference,
        answers=answers,
        class_ids=class_ids,
        correct=correct,
        logprob_sums=logprobs,
        token_counts=tokens,
        embeddings=pool_embeddings,
        n_unparsed=len(rows) - len(parsed_indices),
        metadata=dict(metadata or {}),
    )


def load_questions_file(cache: Path, dataset: str) -> Dict[str, Question]:
    """Read the question bank pulled from the Hub (questions/{dataset}.jsonl)."""
    import json

    path = cache / "questions" / f"{dataset_key(dataset)}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Question bank missing: {path}")
    questions: Dict[str, Question] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question = Question.from_dict(json.loads(line))
            questions[question.qid] = question
    return questions


def load_cell(
    cache: Path,
    model_short: str,
    dataset: str,
    *,
    encoder_short: str = "bge-large-en-v1.5",
    with_embeddings: bool = True,
    canonicalizer: Optional[MathCanonicalizer] = None,
    anisotropy_components: int = 4,
    anisotropy_fit_per_question: int = 64,
    anisotropy_seed: int = 0,
) -> List[Pool]:
    """Load every pool of one (model, dataset) cell from the local cache.

    Also fits the common embedding directions once on the whole cell corpus and
    shares them with every pool, so ``Pool.kernel("embedding", components=c)``
    uses a single corpus-level basis rather than a per-question one.
    """
    key = dataset_key(dataset)
    bank = read_cached_bank(cache, model_short, key)
    questions = load_questions_file(cache, key)
    if canonicalizer is None and key == "math":
        canonicalizer = MathCanonicalizer()
    pools: List[Pool] = []
    for qid in sorted(bank):
        if qid not in questions:
            raise KeyError(f"{qid} has chains but no question record")
        question = questions[qid]
        embeddings: Optional[FloatArray] = None
        if with_embeddings:
            path = embedding_path(cache, encoder_short, model_short, key, qid)
            if path.exists():
                embeddings = np.load(path).astype(np.float64)
        pools.append(
            build_pool(
                qid,
                key,
                model_short,
                bank[qid],
                question.reference_answer,
                embeddings=embeddings,
                canonicalizer=canonicalizer,
                metadata=dict(question.metadata),
            )
        )
    if with_embeddings and anisotropy_components > 0:
        fit_corpus_directions(
            pools,
            components=anisotropy_components,
            per_question=anisotropy_fit_per_question,
            seed=anisotropy_seed,
        )
    return pools


def fit_corpus_directions(
    pools: Sequence[Pool],
    *,
    components: int = 4,
    per_question: int = 64,
    seed: int = 0,
) -> Optional[FloatArray]:
    """Fit the leading common directions on a cell's whole chain corpus.

    A bounded per-question sample keeps the SVD cheap while still spanning every
    question; the fitted mean and basis are attached to each pool in place.
    """
    embedded = [pool for pool in pools if pool.embeddings is not None]
    if not embedded:
        return None
    generator = np.random.default_rng(seed)
    rows = []
    for pool in embedded:
        source = np.asarray(pool.embeddings, dtype=np.float64)
        count = min(per_question, source.shape[0])
        rows.append(source[generator.choice(source.shape[0], size=count, replace=False)])
    corpus = np.concatenate(rows)
    mean = corpus.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(corpus - mean, full_matrices=False)
    directions = np.asarray(right[:components], dtype=np.float64)
    for pool in embedded:
        pool.corpus_mean = mean
        pool.corpus_directions = directions
    return directions
