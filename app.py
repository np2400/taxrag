"""Bare Streamlit UI: a text box, a submit button, and the synthesized
answer with citations. No polish — Phase 7 owns that.
"""

import os

import streamlit as st

st.set_page_config(page_title="TaxRAG", page_icon="📄")

# Local dev reads GROQ_API_KEY from .env (see config/settings.py). Streamlit
# Community Cloud has no .env -- secrets are set in its dashboard instead and
# surface via st.secrets, so bridge that into the environment variable
# config.settings expects, before anything below imports it.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.pipeline import Pipeline, PipelineConfig


@st.cache_resource
def get_pipeline() -> Pipeline:
    """Cached across reruns — Streamlit re-executes this whole script on
    every interaction, and Pipeline() is expensive to construct (loads
    the embedding model, opens the Chroma connection, builds the BM25
    index, and on a fresh deploy with no prebuilt data/chroma/, embeds
    the whole corpus once). Without this, every keystroke would reload
    all of that from scratch.

    use_sparse=True: explicit here, though it now matches
    PipelineConfig's default -- hybrid (BM25 + dense, RRF-fused) is what
    the README and eval results describe."""
    return Pipeline(PipelineConfig(use_sparse=True))


st.title("TaxRAG — Small-Business Tax Research Assistant")
st.caption("Research assistant surfacing federal tax authority. Not tax advice.")
st.caption(
    "Covers the home office deduction, vehicle and mileage deduction, and "
    "self-employment tax — federal only."
)
st.caption(
    "Examples:  \n"
    "What does IRC §280A(c)(1) require for the home-office exception?  \n"
    "What records are required to substantiate business vehicle use?  \n"
    "When can a self-employed taxpayer deduct vehicle mileage?"
)

with st.form("ask_form"):
    query = st.text_input("Ask a question about home office, mileage, or self-employment tax:")
    submitted = st.form_submit_button("Ask")

if submitted and query:
    with st.spinner("Building index (first run only)..."):
        pipeline = get_pipeline()
    with st.spinner("Retrieving and generating..."):
        answer = pipeline.answer(query)

    if answer.refused:
        # answer.text is the raw model output, which for a refusal is the
        # literal REFUSAL_MARKER string ("INSUFFICIENT_INFORMATION") --
        # generate.py's contract with the model, not something meant for
        # an end user to see. Shown here as a real explanation instead.
        st.warning(
            "I couldn't find sufficient authority in my corpus to answer "
            "this. This system covers the home office deduction, vehicle "
            "and mileage deduction, and self-employment tax — federal "
            "only. Questions outside that scope, or requiring judgment "
            "about a specific taxpayer's situation, are declined by "
            "design."
        )
    else:
        # st.write renders strings as markdown, and markdown treats $...$
        # as LaTeX math -- so a dollar amount like "$5 per square foot"
        # gets rendered as a math expression instead of plain text.
        # Escaped only here, at display time: the raw answer stored in
        # eval results (generate.py) must stay unescaped.
        st.write(answer.text.replace("$", "\\$"))
        if answer.citations:
            st.markdown("**Sources cited:**")
            for c in answer.citations:
                st.markdown(f"- {c}")
