# 🎓 AI Autonomous Viva-Voce & Technical Examiner System

An automated AI evaluation engine that assesses students' viva-voce technical answers using
**Natural Language Processing**: TF-IDF vectorisation, cosine similarity, technical keyword coverage,
answer-depth analysis and verbal filler-word detection — delivered through an interactive
Streamlit dashboard.

| | |
|---|---|
| **Student** | Pankaj Kumawat |
| **Certificate ID** | BHIBMAC05438 |
| **Programme** | IBM SkillsBuild AI Automation & Intelligent Solutions Internship |
| **Version** | 1.0.0 |

---

## 1. Features

- **14-question viva bank** across two domains — *AI & Machine Learning* and *Python & Data Science* —
  each with a comprehensive model answer and mandatory technical keywords.
- **Semantic evaluation** with two TF-IDF representations (stemmed word 1–2-grams and character
  3–5-grams) compared by cosine similarity, plus sentence-level **concept coverage** of the model answer.
- **Keyword coverage** that tolerates plurals, verb forms, hyphenation and British/American spelling
  (`regularisation` = `regularization`, `back propagation` = `backpropagation`), with partial credit and
  accepted alternatives (`global interpreter lock|gil`).
- **Filler / hesitation detection** (`umm`, `uhh`, `err`, `basically`, `actually`, `like`, `you know`,
  `kind of` …) with context-aware rules — *"languages like Python"* and *"a kind of neural network"* are
  **not** penalised.
- **Anti-gaming guard**: answers that are just lists of buzzwords are detected and their keyword and
  semantic credit is reduced.
- **Composite score (0–100)**, letter grade, status badge (Distinction / Pass / Borderline / Fail) and
  actionable examiner feedback.
- **Dashboard**: KPI cards, side-by-side diagnostics, interactive Plotly charts (score radar, score
  build-up, concept coverage, filler breakdown), highlighted transcript, model-answer reveal,
  PDF/TXT transcript upload, JSON report and CSV session-history export.

---

## 2. Folder Layout

```
ai-viva-examiner/
├── app.py            # Streamlit dashboard (UI layer)
├── evaluator.py      # VivaEvaluator – NLP scoring engine (no Streamlit dependency)
├── questions.py      # Viva question bank + accessor/validation helpers
├── config.py         # All settings: metadata, weights, fillers, penalties, grade bands
├── requirements.txt  # Python dependencies with minimum versions
└── README.md         # This document
```

---

## 3. Installation

**Prerequisites:** Python **3.10 or newer** and `pip`.

```bash
# 1. Enter the project folder
cd ai-viva-examiner

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies: `streamlit>=1.40.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`, `plotly>=5.18.0`,
`numpy>=1.24.0`, `pypdf>=4.0.0`.

---

## 4. Running the Application

```bash
streamlit run app.py
```

Streamlit prints a local URL (normally **http://localhost:8501**) and opens it in your browser.
Stop the server with `Ctrl + C`.

### Using the dashboard

1. In the **sidebar**, choose a **technical domain** and a **viva question**.
   *Examiner mode* reveals the expected keywords and model answer in advance.
2. **Type or paste** the answer (a speech-to-text transcript works well), or **upload** a PDF/TXT
   transcript — its text is placed in the answer box for review.
3. Click **🔍 Evaluate Answer**.
4. Review the **KPI cards** (Final Score, Grade, Semantic Accuracy, Keyword Match), the **status badge**,
   the **diagnostic breakdown** and the **interactive charts**, then read the **examiner feedback**.
5. Expand the **highlighted transcript**, **concept coverage** table or **model answer** for detail.
6. Download the **JSON report** for the current answer or the **CSV session history**.

---

## 5. System Architecture

```
                         ┌──────────────────────────────┐
  Student answer ───────►│  app.py  (Streamlit UI)      │◄──── PDF / TXT upload (pypdf)
  Question selection ───►│  sidebar · KPIs · charts     │
                         └──────────────┬───────────────┘
                                        │ evaluate(answer, question)
                                        ▼
┌──────────────────────────────── evaluator.py : VivaEvaluator ────────────────────────────────┐
│                                                                                               │
│  ① Filler detection (regex) ──► penalty  ──────────────────────────────────────────┐          │
│        │ strip fillers                                                             │          │
│        ▼                                                                           │          │
│     clean answer                                                                   │          │
│        ├──► ② Semantic match  (TF-IDF word + char ► cosine ► calibrate             │          │
│        │                       + concept coverage)                  ×0.50 ──┐      │          │
│        ├──► ③ Keyword match   (stem-aware phrases, partial credit,          │      │          │
│        │                       stuffing guard)                      ×0.35 ──┼─► Σ ─┴─► −penalty │
│        └──► ④ Answer depth    (length · diversity · coherence)      ×0.15 ──┘          │      │
│                                                                                        ▼      │
│                           ⑤ Final score 0–100 ► grade ► status badge ► feedback ► diagnostics │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
           ▲                                   ▲
     questions.py (model answers,        config.py (weights, calibration,
     keywords, reference corpus)         fillers, penalties, grade bands)
```

`evaluator.py` depends only on scikit-learn, NumPy and the standard library, so the engine can be reused
in scripts, APIs or tests without Streamlit.

### 5.1 Pre-processing

- **Normalisation** — typographic quotes/dashes unified, whitespace collapsed.
- **Tokenisation** — lower-case alphanumeric tokens.
- **Light stemming** — a conservative, dependency-free suffix stemmer applied identically to answers,
  model answers and keywords (`overfitting/overfits → overfit`, `labelled/labeled → label`,
  `normalisation/normalized → normaliz`).
- **Stop words** — scikit-learn's English list, minus technically meaningful words such as
  `next`, `fill`, `first`, `back` (see `STOPWORD_EXCEPTIONS` in `config.py`).

### 5.2 Filler detection & fluency penalty

Each filler in `FILLER_WORDS` is compiled into a case-insensitive regex. Hesitation sounds tolerate
elongation (`um`, `umm`, `ummm`), while ambiguous words use context rules (`like` only counts in
discourse position such as *", like,"*; `kind of` / `sort of` only in hedging position such as
*"it kind of reduces"*).

```
penalty = min( Σ points(filler_i) for occurrences after the first N forgiven , max_penalty )
        + density_surcharge   if fillers / words > density_threshold
```

Defaults: first **1** occurrence forgiven, **1.0–2.0** points per occurrence (pure hesitations such as
`umm` cost the most), cap **15**, surcharge **+5** when fillers exceed **6 %** of all words. Fillers are
removed before content analysis so they are not penalised twice.

### 5.3 Semantic match (50 %)

1. Two `TfidfVectorizer`s are fitted on the **reference corpus** (every model answer in the bank) plus
   the student's answer, so IDF weights are domain-aware and off-topic vocabulary lowers similarity:
   - **word** analyser: stop-word removal → stemming → unigrams + bigrams, sub-linear TF;
   - **char_wb** analyser: 3–5-character n-grams (robust to typos and spelling variants).
2. **Cosine similarity** between the answer vector and the model-answer vector is computed for each
   representation:  `cos(θ) = (a · b) / (‖a‖ · ‖b‖)`.
3. Because short spoken answers rarely reach high raw cosine against a long model answer, each value
   is **linearly calibrated** to 0–100 (word: 0.03 → 0 %, 0.35 → 100 %; char: 0.15 → 0 %, 0.60 → 100 %)
   and blended 60 : 40 into a *document score*.
4. **Concept coverage** — the model answer is split into sentences (ideas). For each one, the share of
   its unigram TF-IDF weight that also occurs in the student's answer is measured; ≥ 45 % counts as
   *covered*.
5. `semantic = 0.6 × document score + 0.4 × concept score`.

### 5.4 Keyword match (35 %)

For every keyword (and its `|`-separated alternatives):

| Result | Rule | Credit |
|---|---|---|
| **Matched** | stemmed phrase appears contiguously, or as a hyphen/space variant | 1.0 |
| **Partial** | all words of a multi-word keyword appear, but not together | 0.5 |
| **Missing** | not found | 0.0 |

`keyword % = Σ credit / number of keywords × 100`.
**Stuffing guard:** if an answer of ≥ 12 tokens has < 18 % function words (normal prose has
35–55 %), it is treated as a buzzword list and both keyword and semantic scores are multiplied by **0.4**.

### 5.5 Answer depth (15 %)

```
target   = max(25, 0.55 × content words in the model answer)
depth    = 0.6 × min(1, content words / target)
         + 0.2 × min(1, type-token ratio / 0.55)        # lexical diversity (penalises repetition)
         + 0.2 × min(1, function-word ratio / 0.30)     # sentence coherence
```

### 5.6 Composite score, grade and status

```
weighted = 0.50 × semantic + 0.35 × keyword + 0.15 × depth
final    = clip(weighted − filler penalty, 0, 100)
```

Answers with fewer than **5** meaningful words (after removing fillers) are marked
**INSUFFICIENT ANSWER** with a score of 0.

| Score | Grade | Descriptor | | Score | Status |
|---|---|---|---|---|---|
| ≥ 90 | A+ | Outstanding | | ≥ 75 | 🏆 DISTINCTION |
| ≥ 80 | A | Excellent | | ≥ 50 | ✅ PASS |
| ≥ 70 | B+ | Very Good | | ≥ 40 | ⚠️ BORDERLINE PASS |
| ≥ 60 | B | Good | | < 40 | ❌ FAIL |
| ≥ 50 | C | Satisfactory | | | |
| ≥ 40 | D | Needs Improvement | | | |
| < 40 | F | Unsatisfactory | | | |

The pass mark is **40/100**.

### 5.7 Calibration benchmarks

Measured on question **AIML-01** (*"What is overfitting… how can it be prevented?"*):

| Answer type | Final | Grade | Status |
|---|---|---|---|
| Model answer (self-check, all 14 questions) | 99.9–100 | A+ | Distinction |
| Thorough paraphrased answer | 94.0 | A+ | Distinction |
| Concise, correct answer (British spelling) | 83.0 | A | Distinction |
| Good content, heavy filler words (12 fillers) | 53.6 | C | Pass (−20 pts) |
| Correct but very brief (2 sentences) | 42.6 | D | Borderline pass |
| Bare list of buzzwords | 33.2 | F | Fail (stuffing guard) |
| Vague answer | 19.0 | F | Fail |
| Off-topic answer (about Python lists) | 8.7 | F | Fail |
| "umm I don't know" | 0.0 | F | Insufficient answer |

Each evaluation takes roughly 30 ms on a laptop CPU.

---

## 6. Using the Engine from Python

```python
from evaluator import VivaEvaluator
from questions import get_question

evaluator = VivaEvaluator()
result = evaluator.evaluate(
    "Overfitting is when a model memorises noise in the training data and fails to generalise...",
    get_question("AIML-01"),
)

print(result.final_score, result.grade, result.status)    # e.g. 83.0 A DISTINCTION
print(result.diagnostics["keywords"]["missing"])           # terms the student did not use
print(result.diagnostics["fillers"]["breakdown"])          # {'umm': 2, 'basically': 1, ...}
report = result.to_dict()                                  # JSON-serialisable
```

Custom questions that are not in the bank can be evaluated by passing a dictionary with
`id`, `ideal_answer` and `keywords`.

---

## 7. Customisation

| Goal | Where |
|---|---|
| Add or edit questions / domains | `QUESTION_BANK` in `questions.py` (validated on start-up) |
| Accept alternative keyword wording | Use `|` inside a keyword, e.g. `"loss function|cost function"` |
| Change scoring weightages | `SCORING_WEIGHTS` in `config.py` (must sum to 1.0) |
| Make semantic scoring stricter/looser | `*_calibration_floor` / `*_calibration_ceiling` in `SEMANTIC_SETTINGS` |
| Add filler words or change penalties | `FILLER_WORDS`, `FILLER_REGEX_OVERRIDES`, `FILLER_PENALTY_RULES` |
| Change grade bands, status badges, pass mark | `GRADE_BANDS`, `STATUS_BADGES`, `PASS_MARK` |
| Student details / page title | `STUDENT_PROFILE`, `PAGE_CONFIG` |

`config.validate_config()` and `questions.validate_question_bank()` run when the evaluator starts, so
an invalid edit is reported immediately in the dashboard instead of producing wrong scores.

---

## 8. Limitations

- **Lexical, not deep semantic understanding.** TF-IDF measures shared vocabulary. A factually wrong
  answer that reuses the right terms (e.g. *"a tuple is a list that is sorted automatically"*) can still
  earn some semantic credit (≈ 20–30 / 100 in testing), and true synonyms with no shared words are not
  recognised.
- **Transcription quality matters.** Filler detection works on text; speech-to-text engines that
  silently drop "umm"/"uh" will under-report hesitation.
- **Heuristic filler context.** Rules for `like`, `kind of` and `sort of` reduce false positives but
  cannot cover every phrasing; `actually` and `basically` are always counted as requested in the rubric.
- **Calibration** was tuned on the bundled question bank; very short or very long model answers may
  need different calibration values.

## 9. Future Enhancements

- Sentence-embedding similarity (e.g. transformer encoders) alongside TF-IDF for synonym awareness.
- Live microphone capture with speech-to-text (e.g. IBM Watson Speech to Text) and pause/pace metrics.
- Adaptive follow-up questions based on missing concepts.
- Persistent student records and examiner dashboards backed by a database.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| "No extractable text found" on PDF upload | The PDF is a scanned image — run OCR first or upload a TXT transcript. |
| "Configuration error" banner | An edit in `config.py` / `questions.py` broke a rule; the message says which. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

---

*Developed by **Pankaj Kumawat** (Certificate ID **BHIBMAC05438**) as part of the
IBM SkillsBuild AI Automation & Intelligent Solutions Internship.*
