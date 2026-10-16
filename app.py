''' A. Marsili 13/10/26
streamlit app to talk to my PhD
using retrieval.py and generation.py'''

import streamlit as st
from src.retrieval import Retriever
from src.generation import generate_answer
from pathlib import Path
import base64

PDF_DIR = Path("PhDdoc")
PDF_FILE = PDF_DIR / "VD2_MARSILI_AURELIEN_21112012.pdf"

st.set_page_config(page_title="Talk to My PhD", page_icon="🎓",
    # layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎓 Talk to My PhD")
st.markdown("A little RAG demosntration: ask questions about my PhD thesis, grounded in the actual text.")

@st.cache_resource
def load_retriever():
    """getting the Retriever instance is expensive, so using the cache_resource trick"""
    return Retriever()

@st.cache_data
def load_pdf_base64(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

retriever = load_retriever()

# default value for pages. Remove?
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1
# default value for show_pdf, showing only if the button is clicked
if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False

query = st.text_input("Ask a question about my PhD thesis:",
                      placeholder="E.g. 'What is the default loss profile?' or 'How many detectors in the BLM system?'")

if query:
    with st.spinner("Searching thesis and generating answer..."):
        results = retriever.search(query, top_k=5)
        answer = generate_answer(query, results)

    st.subheader("Answer")
    st.write(answer)

    # default to first result's page if nothing clicked yet this query
    if results:
        st.session_state.selected_page = results[0].get("page", 1)

    with st.expander("Sources used", expanded=True):
        for i, r in enumerate(results):
            col1, col2 = st.columns([4, 1]) # it's   always same file name, always same width: this way the button text is not truncated
            with col1:
                st.markdown(f"**{r['source']}**, page {r.get('page')} (distance: {r['distance']:.3f})")
                st.caption(r["text"][:300] + "...")
            with col2:
                if st.button("Jump to page", key=f"view_{i}"):
                    st.session_state.selected_page = r.get("page", 1)
                    st.session_state.show_pdf = True
                    

    if st.session_state.show_pdf:
        st.divider()
        with st.expander(f"📖 PDF — page {st.session_state.selected_page}", expanded=True):
            base64_pdf = load_pdf_base64(PDF_FILE)
            pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{base64_pdf}#page={st.session_state.selected_page}"
                width="100%"
                height="1100"
                type="application/pdf">
            </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
