"""
evaluator.py
============

Core NLP evaluation engine for the AI Autonomous Viva-Voce & Technical Examiner System.

Pipeline for one answer
-----------------------
1. **Filler detection** - regex scan for hesitation sounds and verbal fillers;
   occurrences are recorded, penalised, and stripped to produce a *clean* answer.
2. **Semantic match (50 %)** - the clean answer and the model answer are
   embedded with two TF-IDF vectorisers (stemmed word 1-2 grams + character
   3-5 grams) fitted on the whole reference corpus, compared with cosine
   similarity, calibrated to 0-100, and blended with *concept coverage*
   (share of each model-answer sentence's TF-IDF weight present in the answer).
3. **Keyword match (35 %)** - stem-aware phrase matching of the mandatory
   technical terms, with partial credit and a keyword-stuffing guard.
4. **Answer depth (15 %)** - content length versus a target, lexical diversity
   and sentence coherence (function-word ratio).
5. **Composite** - weighted sum minus filler penalty -> final score (0-100),
   letter grade, status badge and a full diagnostics dictionary.

The module depends only on scikit-learn, NumPy and the standard library, so
it can be unit-tested or reused outside Streamlit::

    from evaluator import VivaEvaluator
    from questions import get_question

    result = VivaEvaluator().evaluate("Overfitting is when ...", get_question("AIML-01"))
    print(result.final_score, result.grade, result.status)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from questions import get_question, get_reference_corpus, keyword_display_name, validate_question_bank

__all__ = ["VivaEvaluator", "EvaluationResult", "tokenize", "stem"]

# =============================================================================
# TEXT UTILITIES
# =============================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9(\"'])")
_WHITESPACE_RE = re.compile(r"\s+")

_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - frozenset(config.STOPWORD_EXCEPTIONS)

# Ordered derivational suffix rules: (suffix, replacement). The first rule whose
# remaining stem is at least 3 characters long is applied.
_SUFFIX_RULES: Tuple[Tuple[str, str], ...] = (
    ("ization", "iz"), ("isation", "iz"),
    ("izing", "iz"), ("ising", "iz"), ("ized", "iz"), ("ised", "iz"),
    ("izer", "iz"), ("iser", "iz"), ("ize", "iz"), ("ise", "iz"),
    ("ibility", "ibl"), ("ability", "abl"), ("ity", ""),
    ("fulness", "ful"), ("iveness", "iv"), ("ousness", "ous"),
    ("ational", "at"), ("ation", "at"),
    ("ments", ""), ("ment", ""), ("ness", ""),
    ("ingly", ""), ("ily", "y"), ("ly", ""),
    ("ing", ""), ("ed", ""), ("er", ""),
)
_INFLECTIONAL = {"ing", "ed", "ingly"}


def normalize_text(text: str) -> str:
    """Collapse whitespace and normalise typographic quotes/dashes."""
    if not text:
        return ""
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric tokens (hyphens, slashes and symbols split words)."""
    return _TOKEN_RE.findall(text.lower()) if text else []


@lru_cache(maxsize=65536)
def stem(token: str) -> str:
    """
    Lightweight, dependency-free suffix stemmer.

    It is deliberately conservative and is applied identically to answers,
    model answers and keywords, so only *consistency* matters: e.g.
    ``overfitting/overfits -> overfit``, ``regularisation/regularized -> regulariz``,
    ``labelled/labeled -> label``, ``tuples -> tupl``, ``scaling/scaled/scale -> scal``.
    """
    word = token.lower()
    if len(word) <= 3 or word.isdigit():
        return word

    # 1. Plural / third-person forms
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("xes") or word.endswith("ches") or word.endswith("shes"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is", "as")):
        word = word[:-1]

    # 2. One derivational / inflectional suffix
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + replacement
            # 3. Undouble a final consonant left by -ing / -ed (stopp -> stop, labell -> label)
            if suffix in _INFLECTIONAL and len(word) >= 4 and word[-1] == word[-2]:
                consonant = word[-1]
                if consonant not in "aeiousz" and not (consonant == "l" and len(word) <= 4):
                    word = word[:-1]
            break

    # 4. Silent final 'e' (scale -> scal, biase -> bias, mutable -> mutabl)
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation."""
    text = normalize_text(text)
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _calibrate(raw: float, floor: float, ceiling: float) -> float:
    """Linearly map a raw similarity in [floor, ceiling] onto [0, 100]."""
    return _clip01((raw - floor) / (ceiling - floor)) * 100.0


# =============================================================================
# RESULT OBJECT
# =============================================================================

@dataclass(frozen=True)
class EvaluationResult:
    """Immutable outcome of evaluating one viva answer."""

    question_id: str
    final_score: float
    grade: str
    grade_label: str
    status: str
    status_icon: str
    status_color: str
    passed: bool
    semantic_score: float
    keyword_score: float
    depth_score: float
    filler_penalty: float
    diagnostics: Dict[str, object] = field(default_factory=dict)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable representation (for report downloads / APIs)."""
        return asdict(self)

    def summary_row(self) -> Dict[str, object]:
        """Flat row used for the session history table and CSV export."""
        fillers = self.diagnostics.get("fillers", {})
        keywords = self.diagnostics.get("keywords", {})
        return {
            "Evaluated At (UTC)": self.evaluated_at,
            "Question ID": self.question_id,
            "Final Score": self.final_score,
            "Grade": self.grade,
            "Status": self.status,
            "Semantic %": self.semantic_score,
            "Keyword %": self.keyword_score,
            "Depth %": self.depth_score,
            "Filler Penalty": self.filler_penalty,
            "Fillers Found": fillers.get("total", 0),
            "Keywords Missing": len(keywords.get("missing", [])),
            "Words": self.diagnostics.get("word_count", 0),
        }


# =============================================================================
# EVALUATOR
# =============================================================================

class VivaEvaluator:
    """
    Autonomous viva-voce answer evaluator.

    Parameters
    ----------
    reference_corpus:
        Documents used to fit IDF statistics. Defaults to every model answer in
        the question bank, giving domain-aware term weighting.
    """

    def __init__(self, reference_corpus: Optional[Sequence[str]] = None) -> None:
        config.validate_config()
        validate_question_bank()

        corpus = list(reference_corpus) if reference_corpus is not None else get_reference_corpus()
        self._reference_corpus: List[str] = [normalize_text(doc) for doc in corpus if doc and doc.strip()]
        if not self._reference_corpus:
            raise ValueError("reference_corpus must contain at least one non-empty document.")

        self._weights = dict(config.SCORING_WEIGHTS)
        self._sem = dict(config.SEMANTIC_SETTINGS)
        self._kw = dict(config.KEYWORD_SETTINGS)
        self._depth = dict(config.DEPTH_SETTINGS)
        self._filler_rules = dict(config.FILLER_PENALTY_RULES)
        self._filler_patterns = self._compile_filler_patterns()

    # ------------------------------------------------------------------ public

    def evaluate(self, student_answer: str, question: Dict[str, object]) -> EvaluationResult:
        """
        Evaluate ``student_answer`` against a question entry from ``questions.py``.

        Returns an :class:`EvaluationResult` with the final score (0-100),
        letter grade, status badge and a detailed ``diagnostics`` dictionary.
        """
        if not isinstance(question, dict) or not question.get("ideal_answer"):
            raise ValueError("question must be a question-bank entry with an 'ideal_answer'.")

        raw_answer = normalize_text(student_answer or "")[: config.ANSWER_LIMITS["max_chars"]]
        ideal_answer = normalize_text(str(question["ideal_answer"]))
        keywords: List[str] = list(question.get("keywords", []))
        question_id = str(question.get("id", "CUSTOM"))

        # 1. Fluency: detect fillers on the raw transcript, then strip them.
        filler_report = self.detect_fillers(raw_answer)
        clean_answer = filler_report.pop("clean_text")

        raw_tokens = tokenize(raw_answer)
        clean_tokens = tokenize(clean_answer)
        insufficient = len(clean_tokens) < config.ANSWER_LIMITS["min_words"]

        # 2-4. Component scores on the cleaned answer.
        semantic = self.compute_semantic_similarity(clean_answer, ideal_answer)
        keyword = self.compute_keyword_coverage(clean_answer, keywords)
        depth = self.compute_depth(clean_answer, ideal_answer)

        # A buzzword list is not an explanation: the stuffing guard (already applied
        # to the keyword score) also scales the semantic score.
        semantic["stuffing_adjusted"] = bool(keyword["stuffing_detected"])
        if keyword["stuffing_detected"]:
            semantic["unadjusted_score"] = semantic["score"]
            semantic["score"] = round(semantic["score"] * self._kw["stuffing_multiplier"], 2)

        # 5. Composite score.
        contributions = {
            "semantic": semantic["score"] * self._weights["semantic"],
            "keyword": keyword["score"] * self._weights["keyword"],
            "depth": depth["score"] * self._weights["depth"],
        }
        weighted_score = sum(contributions.values())
        penalty = filler_report["penalty"]
        final_score = 0.0 if insufficient else float(np.clip(weighted_score - penalty, 0.0, 100.0))
        final_score = round(final_score, 1)

        grade, grade_label = self.grade_for(final_score)
        if insufficient:
            status, status_icon, status_color = config.INSUFFICIENT_STATUS
        else:
            status, status_icon, status_color = self.status_for(final_score)

        diagnostics: Dict[str, object] = {
            "question_id": question_id,
            "word_count": len(raw_tokens),
            "clean_word_count": len(clean_tokens),
            "insufficient": insufficient,
            "clean_text": clean_answer,
            "semantic": semantic,
            "keywords": keyword,
            "depth": depth,
            "fillers": filler_report,
            "composite": {
                "weights": dict(self._weights),
                "contributions": {k: round(v, 2) for k, v in contributions.items()},
                "weighted_score": round(weighted_score, 2),
                "filler_penalty": round(penalty, 2),
                "final_score": final_score,
                "pass_mark": config.PASS_MARK,
            },
        }
        diagnostics["feedback"] = self._build_feedback(diagnostics)

        return EvaluationResult(
            question_id=question_id,
            final_score=final_score,
            grade=grade,
            grade_label=grade_label,
            status=status,
            status_icon=status_icon,
            status_color=status_color,
            passed=(not insufficient) and final_score >= config.PASS_MARK,
            semantic_score=round(semantic["score"], 1),
            keyword_score=round(keyword["score"], 1),
            depth_score=round(depth["score"], 1),
            filler_penalty=round(penalty, 1),
            diagnostics=diagnostics,
        )

    def evaluate_by_id(self, student_answer: str, question_id: str) -> EvaluationResult:
        """Convenience wrapper: evaluate against a question looked up by its ID."""
        return self.evaluate(student_answer, get_question(question_id))

    # ------------------------------------------------------- semantic match

    def compute_semantic_similarity(self, answer: str, ideal_answer: str) -> Dict[str, object]:
        """
        TF-IDF + cosine similarity between the answer and the model answer.

        Returns the raw word/char cosines, the calibrated document score, the
        per-sentence concept coverage and the blended semantic ``score`` (0-100).
        """
        answer = normalize_text(answer)
        ideal_answer = normalize_text(ideal_answer)
        concepts_template = [
            {"index": i + 1, "sentence": s, "coverage": 0.0, "covered": False}
            for i, s in enumerate(split_sentences(ideal_answer))
        ]
        empty = {
            "score": 0.0,
            "word_cosine": 0.0,
            "char_cosine": 0.0,
            "document_score": 0.0,
            "concept_score": 0.0,
            "concepts": concepts_template,
            "concepts_covered": 0,
            "concepts_total": len(concepts_template),
        }
        if not tokenize(answer) or not tokenize(ideal_answer):
            return empty

        # Fit on reference corpus + ideal + answer so IDF reflects the domain and
        # the student's own (possibly off-topic) vocabulary lowers similarity.
        corpus = list(self._reference_corpus)
        if ideal_answer not in corpus:
            corpus.append(ideal_answer)
        ideal_idx = corpus.index(ideal_answer)
        corpus.append(answer)
        answer_idx = len(corpus) - 1

        word_vectorizer = TfidfVectorizer(analyzer=self._word_analyzer, **self._word_vectorizer_params())
        word_matrix = word_vectorizer.fit_transform(corpus)
        char_vectorizer = TfidfVectorizer(**config.TFIDF_CHAR_PARAMS)
        char_matrix = char_vectorizer.fit_transform(corpus)

        word_cos = float(cosine_similarity(word_matrix[answer_idx], word_matrix[ideal_idx])[0, 0])
        char_cos = float(cosine_similarity(char_matrix[answer_idx], char_matrix[ideal_idx])[0, 0])

        word_cal = _calibrate(word_cos, self._sem["word_calibration_floor"], self._sem["word_calibration_ceiling"])
        char_cal = _calibrate(char_cos, self._sem["char_calibration_floor"], self._sem["char_calibration_ceiling"])
        document_score = self._sem["word_weight"] * word_cal + self._sem["char_weight"] * char_cal

        # Concept coverage: for each model-answer sentence, the share of its
        # unigram TF-IDF weight carried by terms that also occur in the answer.
        # (Bigrams are excluded here because paraphrases rarely reuse exact pairs.)
        feature_names = word_vectorizer.get_feature_names_out()
        answer_terms = set(word_matrix[answer_idx].indices.tolist())
        threshold = self._sem["concept_match_threshold"]
        concepts: List[Dict[str, object]] = []
        credits: List[float] = []
        for item in concepts_template:
            sent_vec = word_vectorizer.transform([item["sentence"]])
            unigrams = [
                (idx, weight)
                for idx, weight in zip(sent_vec.indices, sent_vec.data)
                if " " not in feature_names[idx]
            ]
            total_weight = float(sum(weight for _, weight in unigrams))
            if total_weight <= 0:
                continue
            shared = float(sum(weight for idx, weight in unigrams if idx in answer_terms))
            coverage = shared / total_weight
            covered = coverage >= threshold
            concepts.append({**item, "coverage": round(coverage, 3), "covered": covered})
            credits.append(min(1.0, coverage / threshold))
        concept_score = float(np.mean(credits) * 100.0) if credits else 0.0

        score = self._sem["document_weight"] * document_score + self._sem["concept_weight"] * concept_score
        return {
            "score": round(float(np.clip(score, 0.0, 100.0)), 2),
            "word_cosine": round(word_cos, 4),
            "char_cosine": round(char_cos, 4),
            "word_score": round(word_cal, 2),
            "char_score": round(char_cal, 2),
            "document_score": round(document_score, 2),
            "concept_score": round(concept_score, 2),
            "concepts": concepts,
            "concepts_covered": sum(1 for c in concepts if c["covered"]),
            "concepts_total": len(concepts),
        }

    # -------------------------------------------------------- keyword match

    def compute_keyword_coverage(self, answer: str, keywords: Sequence[str]) -> Dict[str, object]:
        """
        Stem-aware keyword coverage.

        * **matched** - the keyword (or an alternative) appears as a phrase
        * **partial** - every word of a multi-word keyword appears, but not together
        * **missing** - not found

        Coverage % = (matched + partial_credit x partial) / total x 100, reduced by
        ``stuffing_multiplier`` when the answer reads like a list of buzzwords.
        """
        raw_tokens = tokenize(answer)
        stems = [stem(t) for t in raw_tokens]
        stem_set = set(stems)
        compact = "".join(raw_tokens)
        stopword_ratio = self._stopword_ratio(raw_tokens)
        stuffing = (
            len(raw_tokens) >= self._kw["stuffing_min_tokens"]
            and stopword_ratio < self._kw["stuffing_stopword_ratio"]
        )

        details: List[Dict[str, object]] = []
        credit_total = 0.0
        for spec in keywords:
            status, credit, matched_as = "missing", 0.0, ""
            for alternative in [a.strip() for a in spec.split("|") if a.strip()]:
                alt_status, alt_credit = self._match_keyword(alternative, stems, stem_set, compact)
                if alt_credit > credit:
                    status, credit, matched_as = alt_status, alt_credit, alternative
                if credit >= 1.0:
                    break
            credit_total += credit
            details.append(
                {
                    "keyword": keyword_display_name(spec),
                    "spec": spec,
                    "status": status,
                    "credit": credit,
                    "matched_as": matched_as,
                }
            )

        total = len(keywords)
        coverage_pct = (credit_total / total * 100.0) if total else 100.0
        score = coverage_pct * (self._kw["stuffing_multiplier"] if stuffing else 1.0)
        return {
            "score": round(score, 2),
            "coverage_pct": round(coverage_pct, 2),
            "matched": [d["keyword"] for d in details if d["status"] == "matched"],
            "partial": [d["keyword"] for d in details if d["status"] == "partial"],
            "missing": [d["keyword"] for d in details if d["status"] == "missing"],
            "details": details,
            "total": total,
            "stuffing_detected": stuffing,
            "stopword_ratio": round(stopword_ratio, 3),
        }

    def _match_keyword(
        self, keyword: str, stems: List[str], stem_set: set, compact: str
    ) -> Tuple[str, float]:
        kw_tokens = tokenize(keyword)
        if not kw_tokens:
            return "missing", 0.0
        kw_stems = [stem(t) for t in kw_tokens]

        # Exact phrase on stemmed token sequence.
        n = len(kw_stems)
        for i in range(len(stems) - n + 1):
            if stems[i : i + n] == kw_stems:
                return "matched", 1.0

        # Hyphenation / spacing variants: "back propagation" ~ "backpropagation",
        # "trade-off" ~ "tradeoff". Only for long keywords to avoid false hits.
        kw_compact = "".join(kw_tokens)
        if len(kw_compact) >= 7 and kw_compact in compact:
            return "matched", 1.0

        # Partial: all content words of a multi-word keyword present somewhere.
        if n > 1:
            content = [s for t, s in zip(kw_tokens, kw_stems) if t not in _STOP_WORDS] or kw_stems
            if all(s in stem_set for s in content):
                return "partial", float(self._kw["partial_credit"])
        return "missing", 0.0

    # --------------------------------------------------------- answer depth

    def compute_depth(self, answer: str, ideal_answer: str) -> Dict[str, object]:
        """
        Depth = length vs target (60 %) + lexical diversity (20 %) + coherence (20 %).

        * target content words = max(min_target, target_ratio x model-answer content words)
        * diversity = type-token ratio of stemmed content words vs a reference ratio
        * coherence = share of function words vs a reference (low => buzzword list)
        """
        tokens = tokenize(answer)
        content = [stem(t) for t in tokens if t not in _STOP_WORDS and len(t) > 1]
        ideal_content = [t for t in tokenize(ideal_answer) if t not in _STOP_WORDS and len(t) > 1]

        target = max(
            float(self._depth["min_target_content_words"]),
            self._depth["target_ratio"] * len(ideal_content),
        )
        length_component = _clip01(len(content) / target) if target else 0.0
        ttr = (len(set(content)) / len(content)) if content else 0.0
        diversity_component = _clip01(ttr / self._depth["diversity_reference"]) if content else 0.0
        stopword_ratio = self._stopword_ratio(tokens)
        coherence_component = _clip01(stopword_ratio / self._depth["coherence_reference"]) if tokens else 0.0

        score = 100.0 * (
            self._depth["length_weight"] * length_component
            + self._depth["diversity_weight"] * diversity_component
            + self._depth["coherence_weight"] * coherence_component
        )
        return {
            "score": round(score, 2),
            "content_words": len(content),
            "target_content_words": int(round(target)),
            "length_component": round(length_component * 100, 1),
            "diversity_component": round(diversity_component * 100, 1),
            "coherence_component": round(coherence_component * 100, 1),
            "type_token_ratio": round(ttr, 3),
            "stopword_ratio": round(stopword_ratio, 3),
            "sentences": len(split_sentences(answer)),
        }

    # ------------------------------------------------------ filler detection

    def detect_fillers(self, text: str) -> Dict[str, object]:
        """
        Regex-based filler / hesitation detector.

        Returns every occurrence (with character span), per-filler counts, the
        computed penalty (after free allowance, cap and density surcharge) and
        ``clean_text`` - the answer with fillers removed for content analysis.
        """
        text = normalize_text(text)
        rules = self._filler_rules
        per_word = rules["per_word_points"]

        candidates: List[Tuple[int, int, str]] = []
        for filler, pattern in self._filler_patterns.items():
            for match in pattern.finditer(text):
                start, end = self._core_span(filler, match)
                candidates.append((start, end, filler))

        # Resolve overlaps (keep earliest, then longest).
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
        occurrences: List[Dict[str, object]] = []
        last_end = -1
        for start, end, filler in candidates:
            if start < last_end:
                continue
            occurrences.append({"filler": filler, "text": text[start:end], "start": start, "end": end})
            last_end = end

        free = int(rules["free_allowance"])
        base_penalty = 0.0
        for position, occ in enumerate(occurrences):
            points = 0.0 if position < free else float(per_word.get(occ["filler"], rules["default_points"]))
            occ["points"] = points
            occ["forgiven"] = position < free
            base_penalty += points
        capped = min(base_penalty, float(rules["max_penalty"]))

        word_count = len(tokenize(text))
        density = (len(occurrences) / word_count) if word_count else 0.0
        counted = max(0, len(occurrences) - free)
        surcharge = (
            float(rules["density_surcharge"])
            if counted > 0 and density > rules["density_threshold"]
            else 0.0
        )
        penalty = capped + surcharge
        max_possible = float(rules["max_penalty"]) + float(rules["density_surcharge"])

        breakdown: Dict[str, int] = {}
        for occ in occurrences:
            breakdown[occ["filler"]] = breakdown.get(occ["filler"], 0) + 1
        breakdown = dict(sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])))

        return {
            "total": len(occurrences),
            "counted": counted,
            "forgiven": min(free, len(occurrences)),
            "breakdown": breakdown,
            "occurrences": occurrences,
            "base_penalty": round(base_penalty, 2),
            "capped_penalty": round(capped, 2),
            "density": round(density, 4),
            "density_surcharge": surcharge,
            "penalty": round(penalty, 2),
            "fluency_score": round(100.0 * (1.0 - penalty / max_possible), 1) if max_possible else 100.0,
            "clean_text": self._strip_spans(text, occurrences),
        }

    # ----------------------------------------------------------- grading

    @staticmethod
    def grade_for(score: float) -> Tuple[str, str]:
        """Letter grade and descriptor for a 0-100 score."""
        for threshold, grade, label in config.GRADE_BANDS:
            if score >= threshold:
                return grade, label
        return config.GRADE_BANDS[-1][1], config.GRADE_BANDS[-1][2]

    @staticmethod
    def status_for(score: float) -> Tuple[str, str, str]:
        """Status badge (label, icon, colour) for a 0-100 score."""
        for threshold, label, icon, color in config.STATUS_BADGES:
            if score >= threshold:
                return label, icon, color
        _, label, icon, color = config.STATUS_BADGES[-1]
        return label, icon, color

    # ----------------------------------------------------------- internals

    def _word_analyzer(self, text: str) -> List[str]:
        """Tokenise -> drop stop words -> stem -> 1..n word n-grams."""
        stems = [stem(t) for t in tokenize(text) if t not in _STOP_WORDS and len(t) > 1]
        low, high = config.TFIDF_WORD_PARAMS["ngram_range"]
        features: List[str] = []
        for n in range(int(low), int(high) + 1):
            features.extend(" ".join(stems[i : i + n]) for i in range(len(stems) - n + 1))
        return features

    @staticmethod
    def _word_vectorizer_params() -> Dict[str, object]:
        return {k: v for k, v in config.TFIDF_WORD_PARAMS.items() if k != "ngram_range"}

    @staticmethod
    def _stopword_ratio(tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in ENGLISH_STOP_WORDS) / len(tokens)

    @staticmethod
    def _compile_filler_patterns() -> Dict[str, "re.Pattern[str]"]:
        patterns: Dict[str, re.Pattern] = {}
        for filler in config.FILLER_WORDS:
            override = config.FILLER_REGEX_OVERRIDES.get(filler)
            if override:
                source = override
            else:
                words = [re.escape(w) for w in filler.split()]
                source = r"\b" + r"\s+".join(words) + r"\b"
            patterns[filler] = re.compile(source, re.IGNORECASE | re.MULTILINE)
        return patterns

    @staticmethod
    def _core_span(filler: str, match: "re.Match[str]") -> Tuple[int, int]:
        """Trim context characters (commas, sentence starts) from a filler match."""
        words = [re.escape(w) for w in filler.split()]
        core = re.search(r"\b" + r"\s+".join(words) + r"\b", match.group(0), re.IGNORECASE)
        if core:
            return match.start() + core.start(), match.start() + core.end()
        # Elongated hesitation sounds ("ummm") - the whole match is the filler.
        stripped = match.group(0)
        lead = len(stripped) - len(stripped.lstrip(" ,.;:!?"))
        trail = len(stripped) - len(stripped.rstrip(" ,.;:!?"))
        return match.start() + lead, match.end() - trail

    @staticmethod
    def _strip_spans(text: str, occurrences: Sequence[Dict[str, object]]) -> str:
        if not occurrences:
            return text
        parts: List[str] = []
        cursor = 0
        for occ in occurrences:
            parts.append(text[cursor : int(occ["start"])])
            cursor = int(occ["end"])
        parts.append(text[cursor:])
        cleaned = " ".join(parts)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)   # no space before punctuation
        cleaned = re.sub(r"([,;:])\s*(?=[,;:.!?])", "", cleaned)  # drop orphaned commas
        cleaned = re.sub(r"^[\s,;:]+", "", cleaned)
        return _WHITESPACE_RE.sub(" ", cleaned).strip()

    # ----------------------------------------------------------- feedback

    def _build_feedback(self, diag: Dict[str, object]) -> List[Dict[str, str]]:
        """Actionable examiner feedback as (level, message) items."""
        strong = config.FEEDBACK_THRESHOLDS["strong"]
        moderate = config.FEEDBACK_THRESHOLDS["moderate"]
        sem = diag["semantic"]
        kw = diag["keywords"]
        dep = diag["depth"]
        fil = diag["fillers"]
        feedback: List[Dict[str, str]] = []

        if diag["insufficient"]:
            feedback.append({
                "level": "error",
                "message": (
                    f"The answer is too short to evaluate (minimum "
                    f"{config.ANSWER_LIMITS['min_words']} meaningful words after removing fillers)."
                ),
            })
            return feedback

        # Semantic
        covered, total = sem["concepts_covered"], sem["concepts_total"]
        if sem["score"] >= strong:
            feedback.append({
                "level": "success",
                "message": f"Strong conceptual alignment with the model answer ({covered}/{total} key ideas fully covered).",
            })
        elif sem["score"] >= moderate:
            feedback.append({
                "level": "info",
                "message": (
                    f"The core idea is present, but only {covered}/{total} key ideas of the model answer "
                    "were fully addressed. Add the mechanism, an example and practical implications."
                ),
            })
        else:
            feedback.append({
                "level": "warning",
                "message": (
                    f"Low semantic match ({covered}/{total} key ideas fully covered). Revisit the definition "
                    "and explain how and why it works, not just what it is."
                ),
            })
        uncovered = [c for c in sem["concepts"] if not c["covered"]]
        if uncovered and sem["score"] < strong:
            hint = _shorten(uncovered[0]["sentence"], 18)
            feedback.append({"level": "info", "message": f"Consider also explaining: “{hint}”"})

        # Keywords
        if kw["missing"]:
            shown = ", ".join(kw["missing"][:6]) + ("…" if len(kw["missing"]) > 6 else "")
            level = "warning" if kw["coverage_pct"] < moderate else "info"
            feedback.append({"level": level, "message": f"Missing technical terms: {shown}."})
        else:
            feedback.append({"level": "success", "message": "All expected technical keywords were used."})
        if kw["partial"]:
            feedback.append({
                "level": "info",
                "message": "Use the precise terminology for: " + ", ".join(kw["partial"]) + ".",
            })
        if kw["stuffing_detected"]:
            feedback.append({
                "level": "warning",
                "message": (
                    "The answer reads like a list of buzzwords. Keyword credit was reduced - "
                    "explain the terms in complete sentences."
                ),
            })

        # Depth
        if dep["score"] < moderate:
            feedback.append({
                "level": "warning",
                "message": (
                    f"The answer lacks depth ({dep['content_words']} content words; about "
                    f"{dep['target_content_words']} expected). Elaborate with reasoning and an example."
                ),
            })
        elif dep["score"] < strong:
            feedback.append({
                "level": "info",
                "message": "Reasonable depth - one more example or edge case would strengthen it.",
            })

        # Fluency
        if fil["total"] == 0:
            feedback.append({"level": "success", "message": "Fluent delivery - no filler words detected."})
        else:
            top = ", ".join(f"‘{w}’ ×{c}" for w, c in list(fil["breakdown"].items())[:4])
            level = "warning" if fil["penalty"] >= 5 else "info"
            feedback.append({
                "level": level,
                "message": (
                    f"{fil['total']} filler/hesitation word(s) detected ({top}); "
                    f"-{fil['penalty']:.1f} points. Pause silently instead of filling gaps."
                ),
            })
        return feedback


def _shorten(sentence: str, max_words: int) -> str:
    words = sentence.split()
    return sentence if len(words) <= max_words else " ".join(words[:max_words]) + "…"
