"""Bare Streamlit UI: a text box, a submit button, and the synthesized
answer with citations. No polish — Phase 7 owns that.
"""

import streamlit as st

from src.pipeline import Pipeline


@st.cache_resource
def get_pipeline() -> Pipeline:
    """Cached across reruns — Streamlit re-executes this whole script on
    every interaction, and Pipeline() is expensive to construct (loads
    the embedding model, opens the Chroma connection). Without this,
    every keystroke would reload both from scratch."""
    return Pipeline()


st.title("TaxRAG — Small-Business Tax Research Assistant")
st.caption("Research assistant surfacing federal tax authority. Not tax advice.")

query = st.text_input("Ask a question about home office, mileage, or self-employment tax:")

if st.button("Ask") and query:
    pipeline = get_pipeline()
    with st.spinner("Retrieving and generating..."):
        answer = pipeline.answer(query)

    if answer.refused:
        st.warning(answer.text)
    else:
        st.write(answer.text)
        if answer.citations:
            st.markdown("**Sources cited:**")
            for c in answer.citations:
                st.markdown(f"- {c}")
