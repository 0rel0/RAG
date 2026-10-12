import streamlit as st
from dotenv import load_dotenv
from src.retrieval import Retriever
from src.generation import generate_answer

load_dotenv()

st.set_page_config(page_title="Talk to My PhD", page_icon="🎓")
st.title("🎓 Talk to My PhD")
st.markdown("Ask questions about my PhD thesis, grounded in the actual text.")

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

query = st.text_input("Ask a question abouth the thesis:")

if query:
    with st.spinner("Searching thesis and generating answer..."):
        results = retriever.search(query, top_k=5)
        answer = generate_answer(query, results)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources used"):
        for r in results:
            st.markdown(f"**{r['source']}** (distance: {r['distance']:.3f})")
            st.caption(r["text"][:300] + "...")