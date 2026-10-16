"""
config.py
=========

Central configuration for the **AI Autonomous Viva-Voce & Technical Examiner System**.

Every tunable value used by the NLP engine (``evaluator.py``) and the dashboard
(``app.py``) lives here, so behaviour can be adjusted without touching logic:

* application / page setup
* student & internship metadata
* scoring weightages (Semantic 50 %, Keyword 35 %, Depth 15 %)
* TF-IDF vectoriser parameters and similarity calibration
* filler / hesitation words and their per-word penalty rules
* grading bands, status badges and feedback thresholds

``validate_config()`` is executed by the evaluator at start-up so a bad edit
(e.g. weights that no longer sum to 1.0) fails fast with a clear message.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# =============================================================================
# 1. APPLICATION & PAGE SETUP
# =============================================================================

APP_TITLE: str = "AI Autonomous Viva-Voce & Technical Examiner System"
APP_SHORT_TITLE: str = "AI Viva Examiner"
APP_VERSION: str = "1.0.0"
APP_TAGLINE: str = (
    "Automated NLP evaluation of viva-voce technical answers using TF-IDF "
    "vectorisation, cosine similarity, keyword coverage and verbal fluency analysis."
)

# Passed verbatim to st.set_page_config(**PAGE_CONFIG)
PAGE_CONFIG: Dict[str, str] = {
    "page_title": f"{APP_SHORT_TITLE} | Viva-Voce Evaluation",
    "page_icon": "🎓",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Brand palette used by the dashboard (works on light and dark Streamlit themes)
THEME: Dict[str, str] = {
    "primary": "#0F62FE",      # deep blue  - headers, primary accents
    "secondary": "#8A3FFC",    # violet     - secondary accents
    "success": "#198038",      # green      - pass / matched
    "warning": "#B28600",      # amber      - borderline / partial
    "danger": "#DA1E28",       # red        - fail / missing / penalties
    "neutral": "#6F6F6F",      # grey       - captions
}

# =============================================================================
# 2. STUDENT & PROGRAMME METADATA
# =============================================================================

STUDENT_PROFILE: Dict[str, str] = {
    "name": "Pankaj Kumawat",
    "certificate_id": "BHIBMAC05438",
    "program": "IBM SkillsBuild AI Automation & Intelligent Solutions Internship",
    "track": "AI Automation & Intelligent Solutions",
    "project_title": APP_TITLE,
}

# =============================================================================
# 3. SCORING WEIGHTAGES  (must sum to 1.0)
# =============================================================================

SCORING_WEIGHTS: Dict[str, float] = {
    "semantic": 0.50,  # TF-IDF + cosine similarity against the model answer
    "keyword": 0.35,   # coverage of mandatory technical terms
    "depth": 0.15,     # length, lexical diversity and sentence coherence
}

COMPONENT_LABELS: Dict[str, str] = {
    "semantic": "Semantic Match",
    "keyword": "Keyword Match",
    "depth": "Answer Depth",
}

# =============================================================================
# 4. SEMANTIC SIMILARITY ENGINE (TF-IDF + COSINE)
# =============================================================================

# Word-level TF-IDF: unigrams + bigrams so phrases like "learning rate" count.
# The evaluator supplies its own analyzer (tokenise -> stop-word removal ->
# light stemming -> n-grams), so only these vectoriser options are passed on.
TFIDF_WORD_PARAMS: Dict[str, object] = {
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "smooth_idf": True,
    "norm": "l2",
}

# Words in scikit-learn's English stop-word list that carry technical meaning
# in viva answers (e.g. generator "next", Pandas "forward fill", "first-class").
STOPWORD_EXCEPTIONS: List[str] = [
    "next", "back", "fill", "first", "last", "call", "system", "bottom", "top",
    "front", "full", "empty", "per", "whole", "amount", "move", "find", "detail",
    "describe", "part", "interest",
]

# Character-level TF-IDF (within word boundaries): tolerant of morphology,
# British/American spelling and minor typos ("regularisation" ~ "regularization").
TFIDF_CHAR_PARAMS: Dict[str, object] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": "unicode",
}

SEMANTIC_SETTINGS: Dict[str, float] = {
    # Blend of the two calibrated cosine signals -> document-level similarity.
    "word_weight": 0.60,
    "char_weight": 0.40,
    # Final semantic score = document similarity blended with concept coverage
    # (share of the model answer's ideas, sentence by sentence, that were addressed).
    "document_weight": 0.60,
    "concept_weight": 0.40,
    # Raw cosine values are small for short spoken answers compared with a long
    # model answer, so each signal is linearly calibrated to 0-100:
    #   raw <= floor   -> 0 %
    #   raw >= ceiling -> 100 %
    # Character n-grams share many fragments even between unrelated English
    # texts, hence the higher floor for the char signal.
    "word_calibration_floor": 0.03,
    "word_calibration_ceiling": 0.35,
    "char_calibration_floor": 0.15,
    "char_calibration_ceiling": 0.60,
    # A model-answer sentence counts as "covered" when this share of its TF-IDF
    # weight (its informative terms) appears in the student's answer.
    "concept_match_threshold": 0.45,
}

# =============================================================================
# 5. KEYWORD COVERAGE
# =============================================================================

KEYWORD_SETTINGS: Dict[str, float] = {
    # Credit for a multi-word keyword whose words all appear, but not as a phrase.
    "partial_credit": 0.5,
    # Keyword-stuffing guard: genuine English prose is ~35-55 % function words.
    # A "list of buzzwords" answer falls far below this ratio.
    "stuffing_stopword_ratio": 0.18,
    "stuffing_min_tokens": 12,
    # Multiplier applied to BOTH the keyword and the semantic score when
    # stuffing is detected (a buzzword list is not an explanation).
    "stuffing_multiplier": 0.40,
}

# =============================================================================
# 6. ANSWER DEPTH
# =============================================================================

DEPTH_SETTINGS: Dict[str, float] = {
    # Target content words = max(min_target, target_ratio * model-answer content words)
    "target_ratio": 0.55,
    "min_target_content_words": 25,
    # Sub-component weights (must sum to 1.0)
    "length_weight": 0.60,
    "diversity_weight": 0.20,
    "coherence_weight": 0.20,
    # Type-token ratio that earns full diversity credit
    "diversity_reference": 0.55,
    # Function-word ratio that earns full coherence credit
    "coherence_reference": 0.30,
}

# Input guard-rails
ANSWER_LIMITS: Dict[str, int] = {
    "min_words": 5,       # below this the answer is marked "Insufficient Answer"
    "max_chars": 8000,    # hard cap on text area / uploaded transcript length
}

# =============================================================================
# 7. FILLER / HESITATION WORD DETECTION
# =============================================================================

# Canonical filler words & phrases detected in transcribed viva answers.
FILLER_WORDS: List[str] = [
    "umm",
    "uhh",
    "err",
    "hmm",
    "ahh",
    "basically",
    "actually",
    "literally",
    "like",
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "so yeah",
    "you see",
]

# Custom regular expressions for fillers that need elongation tolerance or
# context. Fillers not listed here are matched as whole words/phrases with
# flexible whitespace, case-insensitively.
FILLER_REGEX_OVERRIDES: Dict[str, str] = {
    # Elongated hesitation sounds: um, umm, ummm / uh, uhh / er, err / hm, hmm / ah, ahh
    "umm": r"\bu+m+\b",
    "uhh": r"\bu+h+\b",
    "err": r"\be+r+\b",
    "hmm": r"\bh+m+\b",
    "ahh": r"\ba+h+\b",
    # "like" is only a filler in discourse position ("..., like, ..." or at the
    # start of a clause) - NOT in "languages like Python" or "I like pandas".
    "like": r"(?:(?:^|[,.;:!?])\s*like\b)|(?:\blike\s*,)",
    # Hedges are fillers only in hedging position ("it kind of reduces...",
    # "..., sort of, ...") - not as nouns ("a special kind of iterator",
    # "this sort of problem"). The match includes the context word; the
    # evaluator trims the reported span back to "kind of" / "sort of".
    "kind of": (
        r"(?:(?:^|[,.;:!?])\s*|\b(?:it|it's|its|is|was|are|were|i|we|you|they|he|she|"
        r"just|really|and|so|to|be|been|get|gets|got|then|also)\s+)kind\s+of\b"
    ),
    "sort of": (
        r"(?:(?:^|[,.;:!?])\s*|\b(?:it|it's|its|is|was|are|were|i|we|you|they|he|she|"
        r"just|really|and|so|to|be|been|get|gets|got|then|also)\s+)sort\s+of\b"
    ),
}

# Penalty rules - points deducted from the 0-100 composite score.
FILLER_PENALTY_RULES: Dict[str, object] = {
    # The first N filler occurrences (in order of appearance) are forgiven;
    # natural speech is rarely 100 % fluent.
    "free_allowance": 1,
    # Default points per counted occurrence for words not in per_word_points.
    "default_points": 1.5,
    # Points per occurrence for each filler (heavier for pure hesitation sounds).
    "per_word_points": {
        "umm": 2.0,
        "uhh": 2.0,
        "err": 2.0,
        "hmm": 1.5,
        "ahh": 1.5,
        "basically": 1.5,
        "actually": 1.0,
        "literally": 1.5,
        "like": 1.5,
        "you know": 2.0,
        "i mean": 1.5,
        "kind of": 1.0,
        "sort of": 1.0,
        "so yeah": 1.5,
        "you see": 1.0,
    },
    # Maximum total deduction from fillers (excluding the density surcharge).
    "max_penalty": 15.0,
    # If fillers make up more than this share of all words (and at least one
    # filler was not forgiven), add a surcharge.
    "density_threshold": 0.06,
    "density_surcharge": 5.0,
}

# =============================================================================
# 8. GRADING THRESHOLDS, STATUS BADGES & FEEDBACK
# =============================================================================

# (minimum score, letter grade, descriptor) - evaluated top-down.
GRADE_BANDS: List[Tuple[float, str, str]] = [
    (90.0, "A+", "Outstanding"),
    (80.0, "A", "Excellent"),
    (70.0, "B+", "Very Good"),
    (60.0, "B", "Good"),
    (50.0, "C", "Satisfactory"),
    (40.0, "D", "Needs Improvement"),
    (0.0, "F", "Unsatisfactory"),
]

PASS_MARK: float = 40.0

# (minimum score, status label, icon, colour) - evaluated top-down.
STATUS_BADGES: List[Tuple[float, str, str, str]] = [
    (75.0, "DISTINCTION", "🏆", THEME["success"]),
    (50.0, "PASS", "✅", THEME["primary"]),
    (PASS_MARK, "BORDERLINE PASS", "⚠️", THEME["warning"]),
    (0.0, "FAIL", "❌", THEME["danger"]),
]

INSUFFICIENT_STATUS: Tuple[str, str, str] = ("INSUFFICIENT ANSWER", "🚫", THEME["danger"])

# Component-level thresholds used to phrase feedback messages (0-100 scale).
FEEDBACK_THRESHOLDS: Dict[str, float] = {
    "strong": 75.0,
    "moderate": 50.0,
}

# =============================================================================
# 9. UPLOADS
# =============================================================================

UPLOAD_SETTINGS: Dict[str, object] = {
    "allowed_types": ["pdf", "txt"],
    "max_pdf_pages": 20,
}


# =============================================================================
# 10. CONFIGURATION VALIDATION
# =============================================================================

def validate_config() -> None:
    """Raise ``ValueError`` if any configuration invariant is violated."""
    _check_sum("SCORING_WEIGHTS", SCORING_WEIGHTS.values())
    _check_sum(
        "SEMANTIC_SETTINGS word/char weights",
        (SEMANTIC_SETTINGS["word_weight"], SEMANTIC_SETTINGS["char_weight"]),
    )
    _check_sum(
        "SEMANTIC_SETTINGS document/concept weights",
        (SEMANTIC_SETTINGS["document_weight"], SEMANTIC_SETTINGS["concept_weight"]),
    )
    _check_sum(
        "DEPTH_SETTINGS sub-weights",
        (
            DEPTH_SETTINGS["length_weight"],
            DEPTH_SETTINGS["diversity_weight"],
            DEPTH_SETTINGS["coherence_weight"],
        ),
    )

    for signal in ("word", "char"):
        floor = SEMANTIC_SETTINGS[f"{signal}_calibration_floor"]
        ceiling = SEMANTIC_SETTINGS[f"{signal}_calibration_ceiling"]
        if not 0.0 <= floor < ceiling <= 1.0:
            raise ValueError(
                f"SEMANTIC_SETTINGS requires 0 <= {signal}_calibration_floor < "
                f"{signal}_calibration_ceiling <= 1."
            )

    for name, rows in (("GRADE_BANDS", GRADE_BANDS), ("STATUS_BADGES", STATUS_BADGES)):
        thresholds = [row[0] for row in rows]
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError(f"{name} must be ordered from highest to lowest threshold.")
        if thresholds[-1] != 0.0:
            raise ValueError(f"{name} must end with a 0.0 catch-all band.")

    missing_rules = [w for w in FILLER_WORDS if w not in FILLER_PENALTY_RULES["per_word_points"]]
    if missing_rules:
        # Not fatal (default_points applies) but surfaces typos early.
        raise ValueError(f"FILLER_PENALTY_RULES['per_word_points'] is missing: {missing_rules}")

    unknown_overrides = [w for w in FILLER_REGEX_OVERRIDES if w not in FILLER_WORDS]
    if unknown_overrides:
        raise ValueError(f"FILLER_REGEX_OVERRIDES has entries not in FILLER_WORDS: {unknown_overrides}")


def _check_sum(name: str, values, tolerance: float = 1e-6) -> None:
    total = float(sum(values))
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name} must sum to 1.0 (currently {total:.4f}).")
