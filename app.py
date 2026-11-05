''' A. Marsili 13/10/26
streamlit app to talk to my PhD
using retrieval.py and generation.py'''

import streamlit as st
from src.retrieval import Retriever
from src.generation import generate_answer
from pathlib import Path
import os
import pymupdf as fitz
# import streamlit.components.v1 as components #deprecated, use st.iframe instead

PDF_DIR = Path("static") # so the file can be served by streamlit cloud
PDF_FILE = PDF_DIR / "VD2_MARSILI_AURELIEN_21112012.pdf"
# bridge for secrets on streamlit cloud
# try:
#     if "MISTRAL_API_KEY" in st.secrets:
#         os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
# except FileNotFoundError:  # no secrets.toml locally
#     print("no secrets.toml locally")


# st.set_page_config(page_title="Talk to My PhD", page_icon="🎓",
#     # layout="wide",
#     initial_sidebar_state="expanded"
# )
# st.title("🎓 Talk to My PhD")
# st.markdown("A little RAG demosntration: ask questions about my PhD thesis, grounded in the actual text.")

# @st.cache_resource
# def load_retriever():
#     """getting the Retriever instance is expensive, so using the cache_resource trick"""
#     return Retriever()

# # @st.cache_data
# # def load_pdf_base64(pdf_path: Path) -> str:
# #     with open(pdf_path, "rb") as f:
# #         return base64.b64encode(f.read()).decode("utf-8")

@st.cache_data
def render_pdf_page(page_number: int) -> bytes:
    """Render a 1-based page of the thesis to PNG bytes."""
    with fitz.open(PDF_FILE) as doc:
        return doc[page_number - 1].get_pixmap(dpi=130).tobytes("png")

@st.cache_data
def get_page_count() -> int:
    with fitz.open(PDF_FILE) as doc:
        return len(doc)

# retriever = load_retriever()

# # default value for pages. Remove?
# if "selected_page" not in st.session_state:
#     st.session_state.selected_page = 1
# # default value for show_pdf, showing only if the button is clicked
if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False

# query = st.text_input("Ask a question about my PhD thesis:",
#                       placeholder="E.g. 'What is the default loss profile?' or 'How many detectors in the BLM system?'")

# if query:
#     with st.spinner("Searching thesis and generating answer..."):
#         results = retriever.search(query, top_k=5)
#         answer = generate_answer(query, results)

#     st.subheader("Answer")
#     st.write(answer)

#     # default to first result's page only when the query changes, so navigation survives reruns
#     if results and st.session_state.get("last_query") != query:
#         st.session_state.last_query = query
#         st.session_state.selected_page = results[0].get("page", 1)

#     with st.expander("Sources used", expanded=True):
#         for i, r in enumerate(results):
#             col1, col2 = st.columns([4, 1]) # it's   always same file name, always same width: this way the button text is not truncated
#             with col1:
#                 st.markdown(f"**{r['source']}**, page {r.get('page')} (distance: {r['distance']:.3f})")
#                 st.caption(r["text"][:300] + "...")
#             with col2:
#                 if st.button("Jump to page", key=f"view_{i}"):
#                     st.session_state.selected_page = r.get("page", 1)
#                     st.session_state.show_pdf = True

selected_page = st.number_input("Page", 1, get_page_count(), st.session_state.selected_page)                

if True:
    if st.session_state.show_pdf:
        st.divider()
        with st.expander(f"📖 PDF — page {st.session_state.selected_page}", expanded=True):
            # iframes get stripped/blocked online, so render the page as an image
            n_pages = get_page_count()
            page = max(1, min(int(st.session_state.selected_page), n_pages))
            st.session_state.selected_page = page

            def go(delta):
                st.session_state.selected_page = max(1, min(st.session_state.selected_page + delta, n_pages))

            c1, c2, c3 = st.columns([1, 1, 2])
            c1.button("◀ Previous", on_click=go, args=(-1,), disabled=page <= 1)
            c2.button("Next ▶", on_click=go, args=(1,), disabled=page >= n_pages)
            c3.number_input("Page", 1, n_pages, key="selected_page")
            st.image(render_pdf_page(st.session_state.selected_page),
                     caption=f"Page {st.session_state.selected_page} / {n_pages}", use_container_width=True)
            with open(PDF_FILE, "rb") as f:
                st.download_button("Download full PDF", f, file_name=PDF_FILE.name, mime="application/pdf")

