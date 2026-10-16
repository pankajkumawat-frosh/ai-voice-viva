"""
app.py
======

Streamlit dashboard for the **AI Autonomous Viva-Voce & Technical Examiner System**.

Run with::

    streamlit run app.py

Layout
------
* **Sidebar** - student profile, domain & question selectors, examiner mode,
  scoring rubric and session controls.
* **Main panel** - question card, answer input (typed or uploaded PDF/TXT
  transcript), KPI score cards, status badge, side-by-side diagnostics and
  interactive Plotly Express charts, examiner feedback, transcript
  highlighting, concept coverage, model answer, report downloads and the
  session history.
"""

from __future__ import annotations

import html
import io
import json
import re
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from evaluator import EvaluationResult, VivaEvaluator, normalize_text, tokenize
from questions import get_domains, get_questions, keyword_display_name

# st.set_page_config must be the first Streamlit command executed.
st.set_page_config(**config.PAGE_CONFIG)

HISTORY_LIMIT = 100
LEVEL_RENDERERS = {
    "success": st.success,
    "info": st.info,
    "warning": st.warning,
    "error": st.error,
}

# =============================================================================
# STYLING
# =============================================================================

CUSTOM_CSS = """
<style>
.viva-hero {
    background: linear-gradient(120deg, #0F62FE 0%, #6929C4 100%);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: #FFFFFF;
    margin-bottom: 1.1rem;
}
.viva-hero h1 {
    color: #FFFFFF;
    font-size: 1.75rem;
    line-height: 1.25;
    margin: 0 0 0.35rem 0;
    padding: 0;
}
.viva-hero p { color: rgba(255,255,255,0.88); margin: 0 0 0.8rem 0; font-size: 0.95rem; }
.viva-chip {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 999px;
    padding: 0.18rem 0.75rem;
    margin: 0.15rem 0.35rem 0.15rem 0;
    font-size: 0.8rem;
    color: #FFFFFF;
}
.q-card {
    border: 1px solid rgba(128,128,128,0.28);
    border-left: 5px solid #0F62FE;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    background: rgba(15,98,254,0.05);
    margin-bottom: 0.9rem;
}
.q-meta { font-size: 0.8rem; opacity: 0.75; margin-bottom: 0.35rem; letter-spacing: 0.02em; }
.q-text { font-size: 1.12rem; font-weight: 600; line-height: 1.45; }
.kpi-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-top: 4px solid var(--kpi-accent, #0F62FE);
    border-radius: 12px;
    padding: 0.85rem 1rem 0.9rem 1rem;
    background: rgba(128,128,128,0.06);
    height: 100%;
}
.kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.75; }
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: 0.15rem; }
.kpi-sub { font-size: 0.82rem; opacity: 0.78; margin-top: 0.15rem; }
.status-banner {
    border-radius: 12px;
    padding: 0.8rem 1.1rem;
    margin: 0.9rem 0 0.4rem 0;
    font-weight: 600;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.2rem;
    align-items: center;
}
.status-banner .status-label { font-size: 1.05rem; letter-spacing: 0.04em; }
.status-banner .status-detail { font-weight: 400; opacity: 0.9; }
.kw-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.16rem 0.7rem;
    margin: 0.16rem 0.3rem 0.16rem 0;
    font-size: 0.82rem;
    border: 1px solid transparent;
}
.kw-matched { background: rgba(25,128,56,0.14); border-color: rgba(25,128,56,0.55); }
.kw-partial { background: rgba(178,134,0,0.14); border-color: rgba(178,134,0,0.55); }
.kw-missing { background: rgba(218,30,40,0.10); border-color: rgba(218,30,40,0.50); text-decoration: line-through; }
.transcript {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    line-height: 1.7;
    background: rgba(128,128,128,0.04);
}
.transcript mark {
    background: rgba(218,30,40,0.20);
    color: inherit;
    border-radius: 4px;
    padding: 0 0.2rem;
}
.sidebar-profile { font-size: 0.88rem; line-height: 1.55; }
.sidebar-profile b { font-weight: 600; }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# STATE & RESOURCES
# =============================================================================

@st.cache_resource(show_spinner="Loading NLP evaluation engine...")
def get_evaluator() -> VivaEvaluator:
    """One evaluator per server process (config + question bank validated once)."""
    return VivaEvaluator()


def init_state() -> None:
    defaults = {
        "answer_text": "",
        "result": None,
        "result_question_id": None,
        "result_answer": "",
        "history": [],
        "upload_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_answer() -> None:
    """Clear the answer and current result (used when the question changes)."""
    st.session_state["answer_text"] = ""
    st.session_state["result"] = None
    st.session_state["result_question_id"] = None
    st.session_state["result_answer"] = ""
    st.session_state["upload_notice"] = None


def reset_session() -> None:
    reset_answer()
    st.session_state["history"] = []


# =============================================================================
# FILE UPLOAD (PDF / TXT TRANSCRIPTS)
# =============================================================================

def extract_text_from_upload(file_name: str, data: bytes) -> str:
    """Extract plain text from an uploaded PDF or TXT answer transcript."""
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if extension == "txt":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")

    if extension == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("PDF support requires 'pypdf' (pip install -r requirements.txt).") from exc

        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    if not reader.decrypt(""):
                        raise ValueError
                except Exception as exc:
                    raise ValueError("The PDF is password-protected. Upload an unlocked copy.") from exc
            max_pages = int(config.UPLOAD_SETTINGS["max_pdf_pages"])
            pages = list(reader.pages)[:max_pages]
            text = "\n".join((page.extract_text() or "").strip() for page in pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read the PDF: {exc}") from exc

        if not text.strip():
            raise ValueError("No extractable text found - the PDF may be a scanned image.")
        return text

    raise ValueError(f"Unsupported file type '.{extension}'. Upload a PDF or TXT file.")


def on_upload() -> None:
    """Callback: load an uploaded transcript into the answer text area."""
    uploaded = st.session_state.get("answer_upload")
    if uploaded is None:
        return
    try:
        text = extract_text_from_upload(uploaded.name, uploaded.getvalue())
    except (ValueError, RuntimeError) as exc:
        st.session_state["upload_notice"] = ("error", str(exc))
        return

    text = " ".join(text.split())
    limit = int(config.ANSWER_LIMITS["max_chars"])
    truncated = len(text) > limit
    st.session_state["answer_text"] = text[:limit]
    st.session_state["result"] = None
    message = f"Loaded {len(tokenize(text[:limit]))} words from '{uploaded.name}'."
    if truncated:
        message += f" The transcript was truncated to {limit:,} characters."
    st.session_state["upload_notice"] = ("warning" if truncated else "success", message)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> Dict[str, object]:
    profile = config.STUDENT_PROFILE
    with st.sidebar:
        st.markdown(f"## {config.PAGE_CONFIG['page_icon']} {config.APP_SHORT_TITLE}")
        st.markdown(
            f"""
            <div class="sidebar-profile">
            <b>Student:</b> {html.escape(profile['name'])}<br>
            <b>Certificate ID:</b> {html.escape(profile['certificate_id'])}<br>
            <b>Programme:</b> {html.escape(profile['program'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("### Viva Setup")
        domain = st.selectbox(
            "Technical domain",
            options=get_domains(),
            key="domain_select",
            on_change=reset_answer,
            help="Subject area of the viva question bank.",
        )
        questions = get_questions(domain)
        by_id = {str(q["id"]): q for q in questions}
        question_id = st.selectbox(
            "Viva question",
            options=list(by_id.keys()),
            format_func=lambda qid: f"{qid} · {by_id[qid].get('topic', '')}",
            key=f"question_select_{domain}",
            on_change=reset_answer,
            help="The answer box is cleared when you switch questions.",
        )
        examiner_mode = st.toggle(
            "Examiner mode",
            value=False,
            help="Reveal the expected keywords and model answer before evaluation.",
        )

        st.divider()
        with st.expander("Scoring rubric", expanded=False):
            weights = config.SCORING_WEIGHTS
            st.markdown(
                "\n".join(
                    f"- **{config.COMPONENT_LABELS[name]}** — {weight:.0%}"
                    for name, weight in weights.items()
                )
            )
            rules = config.FILLER_PENALTY_RULES
            st.markdown(
                f"- **Filler penalty** — per word (first {rules['free_allowance']} forgiven), "
                f"capped at {rules['max_penalty']:.0f} pts, +{rules['density_surcharge']:.0f} pts if "
                f"fillers exceed {rules['density_threshold']:.0%} of words\n"
                f"- **Pass mark** — {config.PASS_MARK:.0f}/100"
            )
            bands = " · ".join(f"{grade} ≥ {threshold:.0f}" for threshold, grade, _ in config.GRADE_BANDS)
            st.caption(f"Grades: {bands}")

        history = st.session_state["history"]
        st.caption(f"Evaluations this session: **{len(history)}**")
        st.button("Reset session", on_click=reset_session, help="Clear the answer, result and history.")

    return {"domain": domain, "question": by_id[question_id], "examiner_mode": examiner_mode}


# =============================================================================
# MAIN PANEL SECTIONS
# =============================================================================

def render_header() -> None:
    profile = config.STUDENT_PROFILE
    chips = "".join(
        f'<span class="viva-chip">{html.escape(text)}</span>'
        for text in (
            f"\U0001F464 {profile['name']}",
            f"\U0001F194 {profile['certificate_id']}",
            f"\U0001F393 {profile['program']}",
        )
    )
    st.markdown(
        f"""
        <div class="viva-hero">
            <h1>{html.escape(config.APP_TITLE)}</h1>
            <p>{html.escape(config.APP_TAGLINE)}</p>
            {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_question_card(question: Dict[str, object], domain: str, examiner_mode: bool) -> None:
    meta = " · ".join(
        str(part)
        for part in (question["id"], domain, question.get("topic", ""), question.get("difficulty", ""))
        if part
    )
    st.markdown(
        f"""
        <div class="q-card">
            <div class="q-meta">{html.escape(meta)}</div>
            <div class="q-text">{html.escape(str(question['question']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if examiner_mode:
        with st.expander("Examiner view: expected keywords & model answer", expanded=False):
            st.markdown(keyword_chips_html([keyword_display_name(k) for k in question["keywords"]], "matched"),
                        unsafe_allow_html=True)
            st.markdown(text_block_html(str(question["ideal_answer"])), unsafe_allow_html=True)


def render_answer_input(question: Dict[str, object]) -> bool:
    """Render the answer area; return True when 'Evaluate' was clicked."""
    st.markdown("#### Student Answer")
    st.text_area(
        "Type or paste the transcribed viva answer",
        key="answer_text",
        height=230,
        max_chars=int(config.ANSWER_LIMITS["max_chars"]),
        placeholder=(
            "Answer as you would in the viva. Speech-to-text transcripts are welcome - filler "
            "words such as 'umm', 'basically' or 'you know' are detected and penalised."
        ),
        label_visibility="collapsed",
    )

    upload_col, info_col = st.columns([3, 2], gap="medium")
    with upload_col:
        st.file_uploader(
            "Or upload an answer transcript (PDF / TXT)",
            type=list(config.UPLOAD_SETTINGS["allowed_types"]),
            key="answer_upload",
            on_change=on_upload,
            help="Text is extracted and placed in the answer box for review before evaluation.",
        )
    with info_col:
        words = len(tokenize(st.session_state["answer_text"]))
        st.metric("Words in answer", words, help="Updates when the text box loses focus.")
        notice_slot = st.empty()
        notice = st.session_state.get("upload_notice")
        if notice:
            level, message = notice
            render = getattr(notice_slot, level if level in LEVEL_RENDERERS else "info")
            render(md_escape(message))

    button_col, clear_col, _ = st.columns([2, 1, 5])
    with button_col:
        clicked = st.button("\U0001F50D Evaluate Answer", type="primary")
    with clear_col:
        st.button("Clear", on_click=reset_answer)
    if clicked:
        notice_slot.empty()  # an earlier upload message is no longer relevant
    return clicked


def run_evaluation(question: Dict[str, object]) -> None:
    answer = st.session_state["answer_text"]
    if not tokenize(answer):
        st.warning("Please type or upload an answer before evaluating.")
        return
    try:
        with st.spinner("Analysing answer with TF-IDF, cosine similarity and fluency checks..."):
            result = get_evaluator().evaluate(answer, question)
    except Exception as exc:  # surface any unexpected engine error without crashing the app
        st.error(f"Evaluation failed: {exc}")
        return

    st.session_state["result"] = result
    st.session_state["result_question_id"] = question["id"]
    st.session_state["result_answer"] = answer
    st.session_state["upload_notice"] = None
    history: List[Dict[str, object]] = st.session_state["history"]
    history.append(result.summary_row())
    st.session_state["history"] = history[-HISTORY_LIMIT:]


# =============================================================================
# RESULTS
# =============================================================================

def kpi_card(label: str, value: str, sub: str, accent: str) -> str:
    return (
        f'<div class="kpi-card" style="--kpi-accent:{accent}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div>'
        f"</div>"
    )


def keyword_chips_html(keywords: List[str], status: str) -> str:
    return "".join(f'<span class="kw-chip kw-{status}">{html.escape(k)}</span>' for k in keywords)


_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~])")


def md_escape(text: str) -> str:
    """Escape Markdown so code-like text (*args, __call__, a * b) renders literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def text_block_html(text: str) -> str:
    """Literal, HTML-escaped text block (no Markdown interpretation)."""
    return f'<div class="transcript">{html.escape(" ".join(str(text).split()))}</div>'


def score_accent(score: float) -> str:
    if score >= config.FEEDBACK_THRESHOLDS["strong"]:
        return config.THEME["success"]
    if score >= config.FEEDBACK_THRESHOLDS["moderate"]:
        return config.THEME["primary"]
    if score >= config.PASS_MARK:
        return config.THEME["warning"]
    return config.THEME["danger"]


def render_kpis(result: EvaluationResult) -> None:
    diag = result.diagnostics
    sem = diag["semantic"]
    kw = diag["keywords"]
    cards = [
        kpi_card(
            "Final Score",
            f"{result.final_score:.1f}",
            f"out of 100 · pass mark {config.PASS_MARK:.0f}",
            score_accent(result.final_score),
        ),
        kpi_card("Grade", result.grade, result.grade_label, result.status_color),
        kpi_card(
            "Semantic Accuracy",
            f"{result.semantic_score:.1f}%",
            f"cosine {sem['word_cosine']:.3f} · {sem['concepts_covered']}/{sem['concepts_total']} ideas",
            score_accent(result.semantic_score),
        ),
        kpi_card(
            "Keyword Match",
            f"{result.keyword_score:.1f}%",
            f"{len(kw['matched'])} matched · {len(kw['partial'])} partial · {len(kw['missing'])} missing",
            score_accent(result.keyword_score),
        ),
    ]
    for column, card in zip(st.columns(4, gap="small"), cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)

    verdict = "Passed" if result.passed else "Not passed"
    st.markdown(
        f"""
        <div class="status-banner" style="background:{result.status_color}1F;border:1px solid {result.status_color};">
            <span class="status-label">{result.status_icon} {html.escape(result.status)}</span>
            <span class="status-detail">{verdict} · Grade {html.escape(result.grade)}
            ({html.escape(result.grade_label)}) · Answer depth {result.depth_score:.1f}% ·
            Filler penalty −{result.filler_penalty:.1f} pts</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def composition_frame(result: EvaluationResult) -> pd.DataFrame:
    comp = result.diagnostics["composite"]
    raw = {"semantic": result.semantic_score, "keyword": result.keyword_score, "depth": result.depth_score}
    rows = [
        {
            "Component": config.COMPONENT_LABELS[name],
            "Score": raw[name],
            "Weight": f"{comp['weights'][name]:.0%}",
            "Points": comp["contributions"][name],
        }
        for name in ("semantic", "keyword", "depth")
    ]
    rows.append({"Component": "Filler penalty", "Score": None, "Weight": "—", "Points": -comp["filler_penalty"]})
    rows.append({"Component": "Final score", "Score": None, "Weight": "—", "Points": result.final_score})
    return pd.DataFrame(rows)


def render_diagnostics(result: EvaluationResult) -> None:
    diag = result.diagnostics
    sem, kw, depth, fil = diag["semantic"], diag["keywords"], diag["depth"], diag["fillers"]

    st.markdown("##### Score composition")
    st.dataframe(
        composition_frame(result),
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score (0-100)", min_value=0, max_value=100, format="%.1f"),
            "Points": st.column_config.NumberColumn("Weighted points", format="%.1f"),
        },
    )

    st.markdown("##### Keyword coverage")
    chips = (
        keyword_chips_html(kw["matched"], "matched")
        + keyword_chips_html(kw["partial"], "partial")
        + keyword_chips_html(kw["missing"], "missing")
    )
    st.markdown(chips or "_No keywords defined for this question._", unsafe_allow_html=True)
    st.caption(
        f"Coverage {kw['coverage_pct']:.1f}% · green = matched, amber = partial (half credit), "
        f"struck-through = missing"
        + (" · ⚠️ keyword-stuffing penalty applied" if kw["stuffing_detected"] else "")
    )

    st.markdown("##### NLP signals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Word TF-IDF cosine", f"{sem['word_cosine']:.3f}", help="Stemmed word 1-2 grams vs model answer.")
    c2.metric("Char TF-IDF cosine", f"{sem['char_cosine']:.3f}", help="Character 3-5 grams (spelling tolerant).")
    c3.metric("Concept coverage", f"{sem['concept_score']:.0f}%", help="Share of model-answer ideas addressed.")
    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Content words",
        f"{depth['content_words']} / {depth['target_content_words']}",
        help="Meaningful words vs the target length.",
    )
    c5.metric("Lexical diversity", f"{depth['type_token_ratio']:.2f}", help="Type-token ratio of content words.")
    c6.metric(
        "Fillers detected",
        fil["total"],
        help=f"Density {fil['density']:.1%} of words; first {config.FILLER_PENALTY_RULES['free_allowance']} forgiven.",
    )

    if fil["breakdown"]:
        filler_df = pd.DataFrame(
            [
                {
                    "Filler": word,
                    "Count": count,
                    "Points each": float(
                        config.FILLER_PENALTY_RULES["per_word_points"].get(
                            word, config.FILLER_PENALTY_RULES["default_points"]
                        )
                    ),
                }
                for word, count in fil["breakdown"].items()
            ]
        )
        st.dataframe(filler_df, hide_index=True)
        if fil["density_surcharge"]:
            st.caption(f"Density surcharge applied: +{fil['density_surcharge']:.0f} pts (fillers {fil['density']:.1%} of words).")


def radar_figure(result: EvaluationResult):
    diag = result.diagnostics
    profile = pd.DataFrame(
        {
            "Dimension": ["Semantic Match", "Keyword Match", "Answer Depth", "Fluency", "Concept Coverage"],
            "Score": [
                result.semantic_score,
                result.keyword_score,
                result.depth_score,
                diag["fillers"]["fluency_score"],
                diag["semantic"]["concept_score"],
            ],
        }
    )
    fig = px.line_polar(profile, r="Score", theta="Dimension", line_close=True, range_r=[0, 100], markers=True)
    fig.update_traces(fill="toself", line_color=config.THEME["primary"], hovertemplate="%{theta}: %{r:.1f}<extra></extra>")
    fig.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=30), showlegend=False)
    return fig


def contribution_figure(result: EvaluationResult):
    comp = result.diagnostics["composite"]
    rows = [
        {"Component": config.COMPONENT_LABELS[name], "Points": comp["contributions"][name], "Type": "Earned"}
        for name in ("semantic", "keyword", "depth")
    ]
    rows.append({"Component": "Filler penalty", "Points": -comp["filler_penalty"], "Type": "Penalty"})
    rows.append({"Component": "Final score", "Points": result.final_score, "Type": "Final"})
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="Points",
        y="Component",
        orientation="h",
        color="Type",
        text="Points",
        color_discrete_map={
            "Earned": config.THEME["primary"],
            "Penalty": config.THEME["danger"],
            "Final": config.THEME["secondary"],
        },
        category_orders={"Component": frame["Component"].tolist()},
    )
    fig.update_traces(texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False)
    fig.add_vline(x=config.PASS_MARK, line_dash="dot", line_color=config.THEME["neutral"],
                  annotation_text="Pass mark", annotation_position="top")
    max_points = max(100.0, float(frame["Points"].max()) + 8)
    min_points = min(0.0, float(frame["Points"].min()) - 6)
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=30, t=40, b=30),
        xaxis=dict(range=[min_points, max_points], title="Points (out of 100)"),
        yaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    return fig


def concept_figure(result: EvaluationResult):
    concepts = result.diagnostics["semantic"]["concepts"]
    frame = pd.DataFrame(
        [
            {
                "Idea": f"Idea {c['index']}",
                "Coverage %": round(float(c["coverage"]) * 100, 1),
                "Status": "Covered" if c["covered"] else "Not covered",
                "Model answer sentence": c["sentence"],
            }
            for c in concepts
        ]
    )
    fig = px.bar(
        frame,
        x="Coverage %",
        y="Idea",
        orientation="h",
        color="Status",
        hover_data={"Model answer sentence": True, "Idea": False},
        color_discrete_map={"Covered": config.THEME["success"], "Not covered": config.THEME["warning"]},
        category_orders={"Idea": frame["Idea"].tolist()},
        range_x=[0, 100],
    )
    threshold = config.SEMANTIC_SETTINGS["concept_match_threshold"] * 100
    fig.add_vline(x=threshold, line_dash="dot", line_color=config.THEME["neutral"],
                  annotation_text="Covered threshold", annotation_position="top")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=40, b=30),
        yaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    return fig


def filler_figure(result: EvaluationResult):
    breakdown = result.diagnostics["fillers"]["breakdown"]
    frame = pd.DataFrame({"Filler": list(breakdown.keys()), "Occurrences": list(breakdown.values())})
    fig = px.bar(frame, x="Filler", y="Occurrences", text="Occurrences",
                 color_discrete_sequence=[config.THEME["danger"]])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=30), yaxis=dict(dtick=1))
    return fig


def render_charts(result: EvaluationResult) -> None:
    profile_tab, build_tab, concept_tab, filler_tab = st.tabs(
        ["Score profile", "Score build-up", "Concept coverage", "Filler words"]
    )
    with profile_tab:
        st.plotly_chart(radar_figure(result), key="chart_radar")
    with build_tab:
        st.plotly_chart(contribution_figure(result), key="chart_contribution")
    with concept_tab:
        if result.diagnostics["semantic"]["concepts"]:
            st.plotly_chart(concept_figure(result), key="chart_concepts")
        else:
            st.info("No concepts could be compared for this answer.")
    with filler_tab:
        if result.diagnostics["fillers"]["breakdown"]:
            st.plotly_chart(filler_figure(result), key="chart_fillers")
        else:
            st.success("No filler or hesitation words detected - fluent delivery.")


def highlighted_transcript(answer: str, occurrences: List[Dict[str, object]]) -> str:
    """HTML transcript with filler occurrences wrapped in <mark>."""
    text = normalize_text(answer)[: int(config.ANSWER_LIMITS["max_chars"])]
    parts: List[str] = []
    cursor = 0
    for occ in occurrences:
        start, end = int(occ["start"]), int(occ["end"])
        parts.append(html.escape(text[cursor:start]))
        parts.append(f"<mark title=\"-{occ['points']:.1f} pts\">{html.escape(text[start:end])}</mark>")
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return f'<div class="transcript">{"".join(parts)}</div>'


def render_feedback_and_details(result: EvaluationResult, question: Dict[str, object]) -> None:
    diag = result.diagnostics
    st.markdown("#### Examiner Feedback")
    for item in diag["feedback"]:
        LEVEL_RENDERERS.get(item["level"], st.info)(md_escape(item["message"]))

    with st.expander("Transcript with filler words highlighted", expanded=False):
        st.markdown(
            highlighted_transcript(st.session_state["result_answer"], diag["fillers"]["occurrences"]),
            unsafe_allow_html=True,
        )
        st.caption("Cleaned text used for semantic, keyword and depth analysis:")
        st.markdown(text_block_html(diag["clean_text"] or "(empty)"), unsafe_allow_html=True)

    with st.expander("Concept-by-concept coverage", expanded=False):
        concepts = diag["semantic"]["concepts"]
        if concepts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Idea": c["index"],
                            "Coverage": round(float(c["coverage"]) * 100, 1),
                            "Covered": "✅" if c["covered"] else "❌",
                            "Model answer sentence": c["sentence"],
                        }
                        for c in concepts
                    ]
                ),
                hide_index=True,
                column_config={
                    "Coverage": st.column_config.ProgressColumn("Coverage %", min_value=0, max_value=100, format="%.0f"),
                    "Model answer sentence": st.column_config.TextColumn(width="large"),
                },
            )
        else:
            st.info("No concepts available.")

    with st.expander("Model answer & expected keywords", expanded=False):
        st.markdown(text_block_html(str(question["ideal_answer"])), unsafe_allow_html=True)
        st.markdown(
            keyword_chips_html([keyword_display_name(k) for k in question["keywords"]], "matched"),
            unsafe_allow_html=True,
        )


def build_report(result: EvaluationResult, question: Dict[str, object]) -> str:
    report = {
        "report": config.APP_TITLE,
        "version": config.APP_VERSION,
        "student": config.STUDENT_PROFILE,
        "question": {k: question.get(k) for k in ("id", "topic", "difficulty", "question")},
        "answer": st.session_state["result_answer"],
        "evaluation": result.to_dict(),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def render_results(result: EvaluationResult, question: Dict[str, object]) -> None:
    st.divider()
    st.markdown("### Evaluation Results")
    if st.session_state["answer_text"] != st.session_state["result_answer"]:
        st.info("The answer has been edited since this evaluation. Click **Evaluate Answer** to re-score it.")
    render_kpis(result)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Diagnostic Breakdown")
        render_diagnostics(result)
    with right:
        st.markdown("#### Visual Evaluation")
        render_charts(result)

    render_feedback_and_details(result, question)

    st.download_button(
        "⬇️ Download evaluation report (JSON)",
        data=build_report(result, question),
        file_name=f"viva_report_{result.question_id}_{result.evaluated_at[:19].replace(':', '-')}.json",
        mime="application/json",
    )


def render_history() -> None:
    history = st.session_state["history"]
    if not history:
        return
    st.divider()
    st.markdown("### Session History")
    frame = pd.DataFrame(history)
    m1, m2, m3 = st.columns(3)
    m1.metric("Answers evaluated", len(frame))
    m2.metric("Average score", f"{frame['Final Score'].mean():.1f}")
    m3.metric("Pass rate", f"{(frame['Final Score'] >= config.PASS_MARK).mean():.0%}")
    st.dataframe(
        frame.iloc[::-1],
        hide_index=True,
        column_config={
            "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%.1f"),
        },
    )
    st.download_button(
        "⬇️ Download session history (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="viva_session_history.csv",
        mime="text/csv",
    )


def render_footer() -> None:
    profile = config.STUDENT_PROFILE
    st.divider()
    st.caption(
        f"{config.APP_TITLE} v{config.APP_VERSION} · Developed by {profile['name']} "
        f"(Certificate ID {profile['certificate_id']}) · {profile['program']}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    inject_css()
    init_state()
    try:
        get_evaluator()
    except ValueError as exc:  # invalid config.py / questions.py edit
        st.error(f"Configuration error: {exc}")
        st.stop()

    selection = render_sidebar()
    question = selection["question"]

    render_header()
    render_question_card(question, str(selection["domain"]), bool(selection["examiner_mode"]))

    if render_answer_input(question):
        run_evaluation(question)

    result: Optional[EvaluationResult] = st.session_state.get("result")
    if result is not None and st.session_state.get("result_question_id") == question["id"]:
        render_results(result, question)

    render_history()
    render_footer()


if __name__ == "__main__":
    main()
