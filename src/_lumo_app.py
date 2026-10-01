"""
Speak to My PhD — Interactive RAG Demo
======================================
A question-answering system over doctoral research using Retrieval-Augmented Generation.

Stack: LangChain | FAISS | sentence-transformers | Streamlit | Groq/Llama
Author: [A. Marsili]
Date: 01 Oct 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS & DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
import os
import streamlit as st
from typing import List, Tuple

# Document processing
import PyMuPDF as fitz  # For PDF → text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings & Vector Store
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader

# LLM Generation (Groq API for speed, fallback to local)
from groq import Groq  # pip install groq
import openai  # Alternative API

# ─────────────────────────────────────────────────────────────────────────────
# 2. PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎓 Speak to My PhD",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #6d4aff; }
    .subheader { font-size: 1.2rem; color: #666; }
    .citation-box { background-color: #f0f0ff; border-left: 4px solid #6d4aff; padding: 10px; margin: 10px 0; }
    .metric-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize persistent session variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'embeddings_model' not in st.session_state:
        st.session_state.embeddings_model = None
    if 'corpus_loaded' not in st.session_state:
        st.session_state.corpus_loaded = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# 4. DOCUMENT LOADING STAGE (STAGE 1: INGESTION)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    LOAD PHASE: Extract raw text from all PDF files in the corpus folder.
    
    Input: Path to folder containing thesis PDFs
    Output: List of LangChain Document objects with metadata
    
    This maps to: Data Preparation & Parsing in RAG pipeline.
    """
    documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # Extract text using PyMuPDF
                loader = PyMuPDFLoader(filepath)
                loaded_docs = loader.load()
                
                # Add source metadata for citations
                for doc in loaded_docs:
                    doc.metadata['source_file'] = filename
                    doc.metadata['chunk_source'] = filepath
                
                documents.extend(loaded_docs)
                st.info(f"✓ Loaded {filename}")
                
            except Exception as e:
                st.error(f"⚠ Failed to load {filename}: {str(e)}")
    
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 500, 
                    chunk_overlap: int = 50) -> List[Document]:
    """
    CHUNKING PHASE: Split long documents into manageable chunks for retrieval.
    
    Input: Raw extracted text documents
    Output: Smaller overlapping chunks (better for semantic search)
    
    This maps to: Index preparation & segmentation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return text_splitter.split_documents(documents)

# ─────────────────────────────────────────────────────────────────────────────
# 5. EMBEDDING GENERATION (STAGE 2: INDEXING)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    EMBEDDING PHASE: Load the sentence-transformer model for vector conversion.
    
    Returns: Initialized HuggingFaceEmbeddings model
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Set to 'cuda' if GPU available
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )
    return embeddings

@st.cache_resource(show_spinner=False)
def create_vector_store(documents: List[Document], embeddings) -> FAISS:
    """
    INDEXING PHASE: Create FAISS vector database from embedded chunks.
    
    Input: Chunked documents + embedding model
    Output: FAISS vector store ready for similarity search
    
    This maps to: Building the retrieval index (offline, one-time operation).
    """
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# ─────────────────────────────────────────────────────────────────────────────
# 6. RETRIEVAL STAGE (STAGE 3: QUERY)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def retrieve_relevant_passages(query: str, vector_store: FAISS, 
                               k: int = 3) -> List[Tuple[str, float]]:
    """
    RETRIEVAL PHASE: Find top-k most relevant passages for the query.
    
    Input: User question text
    Output: List of (passage_text, relevance_score) tuples
    
    This maps to: Semantic search over indexed corpus.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return [(doc.page_content, score) for doc, score in results]

# ─────────────────────────────────────────────────────────────────────────────
# 7. PROMPT AUGMENTATION (STAGE 4: CONTEXT BUILDING)
# ─────────────────────────────────────────────────────────────────────────────
def build_augmented_prompt(query: str, passages: List[Tuple[str, float]]) -> str:
    """
    AUGMENTATION PHASE: Combine query + retrieved context into LLM prompt.
    
    Input: Original question + top-k passages
    Output: Complete prompt with instructions
    
    This maps to: Preparing context-grounded input for generation.
    """
    context_text = "\n\n---\n\n".join([f"[Source]\n{p[0]}" for p in passages])
    
    prompt = f"""You are a research assistant with access to doctoral thesis content. 
Answer the user's question using ONLY the provided context below. 

Be concise and cite specific sections when possible.

═══════════════════════════════════════════
RETRIEVED CONTEXT FROM THESIS:
═══════════════════════════════════════════
{context_text}
═══════════════════════════════════════════

QUESTION: {query}

ANSWER (base your response on the context above):
"""
    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# 8. GENERATION STAGE (STAGE 5: ANSWER SYNTHESIS)
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(prompt: str, use_groq: bool = True) -> str:
    """
    GENERATION PHASE: Call LLM to produce final answer.
    
    Two modes:
    1. Groq API (fast, free tier available)
    2. Local model via Ollama/transformers
    
    Returns: Generated answer string
    """
    if use_groq:
        # Mode 1: Groq Cloud API (recommended for demo)
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",  # Free tier model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower = more deterministic
                max_tokens=500,
                top_p=1.0
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.warning(f"⚠ Groq API error: {str(e)}. Falling back to mock response.")
            return f"[API Error] Please check GROQ_API_KEY environment variable.\n\nFor demo purposes, this would return: The answer to '{prompt[:50]}...' would be generated by the LLM using the retrieved context."
    
    else:
        # Mode 2: Local inference (placeholder for future expansion)
        st.info("ℹ Local mode not configured. Set up ollama or transformers for offline use.")
        return "[Local mode disabled for demo]"

# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN STREAMLIT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<div class="main-header">🎓 Speak to My PhD</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Interactive Q&A over my doctoral research using RAG</div>', unsafe_allow_html=True)
    st.divider()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Corpus path
        corpus_path = st.text_input(
            "Thesis Folder Path",
            value="./phd_corpus/",
            help="Folder containing your thesis PDF files"
        )
        
        # Model settings
        k_values = st.slider("Results per Query (k)", min_value=1, max_value=5, value=3,
                            help="Number of passages to retrieve")
        
        api_choice = st.radio(
            "LLM Backend",
            options=["Groq API (recommended)", "Local Model"],
            index=0
        )
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Thesis", type="primary"):
                with st.spinner("Loading and indexing documents..."):
                    try:
                        docs = load_documents_from_folder(corpus_path)
                        
                        if not docs:
                            st.warning("⚠ No PDF files found in folder.")
                            return
                        
                        chunks = chunk_documents(docs)
                        st.success(f"✓ Processed {len(chunks)} chunks from {len(docs)} documents")
                        
                        embeddings = load_embedding_model()
                        vector_store = create_vector_store(chunks, embeddings)
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.corpus_loaded = True
                        
                        # Show stats card
                        st.rerun()  # Refresh UI state
                        
                    except Exception as e:
                        st.error(f"❌ Loading failed: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.button("🗑️ Clear Cache"):
                st.session_state.vector_store = None
                st.session_state.corpus_loaded = False
                st.info("Cache cleared. Reload the thesis to continue.")
                st.rerun()
    
    # Main content area
    if not st.session_state.corpus_loaded:
        st.info("""
        ## 👋 Welcome!
        
        Get started by loading your thesis:
        
        1. Place your PhD PDF files in a folder (default: `./phd_corpus/`)
        2. Enter the folder path in the sidebar above
        3. Click **Load Thesis**
        
        Once loaded, ask natural language questions about your research!
        """)
        
        # Sample questions
        st.divider()
        st.subheader("💡 Try these sample questions:")
        
        sample_questions = [
            "What was the primary contribution of this research?",
            "How did you validate your methodology?",
            "What were the main limitations acknowledged?",
            "Summarize the literature review approach.",
            "What datasets were used in the experiments?"
        ]
        
        for i, q in enumerate(sample_questions, 1):
            with st.expander(f"{i}. {q}", expanded=(i==1)):
                if st.button("Ask this question", key=f"sample_{i}"):
                    st.session_state['user_question'] = q
    
    else:
        # Display loaded status
        with st.container():
            st.success("✅ Thesis loaded and indexed!")
            st.caption("Vector store initialized with FAISS + sentence-transformers")
        
        st.divider()
        
        # Question input
        user_question = st.text_input(
            "Ask a question about your PhD:",
            placeholder="e.g., What was the novel method proposed in Chapter 3?",
            key="user_question",
            label_visibility="collapsed"
        )
        
        if user_question:
            with st.spinner("🔍 Searching thesis & generating answer..."):
                try:
                    # RETRIEVAL STAGE
                    passages = retrieve_relevant_passages(
                        user_question, 
                        st.session_state.vector_store, 
                        k=k_values
                    )
                    
                    # Display retrieved context (for transparency)
                    with st.expander("📚 Retrieved Context (click to expand)"):
                        for i, (text, score) in enumerate(passages, 1):
                            with st.container():
                                st.markdown(f"**Passage {i}** (relevance: {score:.3f})")
                                st.text_area("Content", text, height=100, disabled=True, key=f"text_{i}")
                            st.divider()
                    
                    # AUGMENTATION STAGE
                    augmented_prompt = build_augmented_prompt(user_question, passages)
                    
                    # GENERATION STAGE
                    answer = generate_answer(augmented_prompt, use_groq=(api_choice == "Groq API (recommended)"))
                    
                    # Display final answer
                    st.divider()
                    st.subheader("🤖 Answer:")
                    st.markdown(answer)
                    
                    # Citations box
                    st.markdown("<div class='citation-box'>📖 Sources: {} document(s) retrieved from your thesis corpus.</div>".format(len(passages)), unsafe_allow_html=True)
                    
                    # Update history
                    st.session_state.conversation_history.append({
                        "question": user_question,
                        "answer": answer,
                        "sources": len(passages)
                    })
                    
                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
                    st.exception(e)
        
        # Conversation history
        if st.session_state.conversation_history:
            st.divider()
            st.subheader("🕐 Previous Questions")
            for i, turn in enumerate(st.session_state.conversation_history[-5:], 1):
                with st.expander(f"{i}. {turn['question'][:50]}..."):
                    st.write(turn['answer'])
    
    # Footer
    st.divider()
    st.caption(
        "Built with RAG (Retrieval-Augmented Generation) | " +
        "[View Code on GitHub](https://github.com/yourusername/phd-rag-demo) | " +
        "[Documentation](https://docs.proton.me/support/lumo)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10. RUN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()