"""Bare Streamlit UI: a text box, a submit button, and the synthesized
answer with citations. No polish — Phase 7 owns that.
"""

import os

import streamlit as st

# Local dev reads GROQ_API_KEY from .env (see config/settings.py). Streamlit
# Community Cloud has no .env -- secrets are set in its dashboard instead and
# surface via st.secrets, so bridge that into the environment variable
# config.settings expects, before anything below imports it.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.pipeline import Pipeline


@st.cache_resource
def get_pipeline() -> Pipeline:
    """Cached across reruns — Streamlit re-executes this whole script on
    every interaction, and Pipeline() is expensive to construct (loads
    the embedding model, opens the Chroma connection, and on a fresh
    deploy with no prebuilt data/chroma/, embeds the whole corpus once).
    Without this, every keystroke would reload all of that from scratch."""
    return Pipeline()


st.title("TaxRAG — Small-Business Tax Research Assistant")
st.caption("Research assistant surfacing federal tax authority. Not tax advice.")

query = st.text_input("Ask a question about home office, mileage, or self-employment tax:")

if st.button("Ask") and query:
    with st.spinner("Building index (first run only)..."):
        pipeline = get_pipeline()
    with st.spinner("Retrieving and generating..."):
        answer = pipeline.answer(query)

    if answer.refused:
        st.warning(answer.text)
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
