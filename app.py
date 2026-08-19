import os
import gc
import time
import streamlit as st
from dotenv import load_dotenv

from core.loader     import load_multiple_pdfs, save_uploaded_file
from core.chunker    import create_parent_child_chunks, update_doc_info_chunk_count
from core.vectorstore import (
    get_embedding_model,
    build_vectorstore,
    close_vectorstore,
    get_store_stats
)
from core.retriever  import build_retriever
from core.chain      import DocuLegalChain
from utils.helpers   import format_answer_for_display
import warnings
import os
import logging

# Silence transformer warnings from unrelated model configs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

st.set_page_config(
    page_title = "DocuLegal",
    page_icon  = "⚖️",
    layout     = "wide"
)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

def initialize_session_state():
    """Initialize all Streamlit session state variables."""
    defaults = {
        "chain"           : None,
        "doc_infos"       : [],
        "all_documents"   : [],
        "child_chunks"    : [],
        "chat_history"    : [],
        "vectorstore"     : None,
        "docstore"        : None,
        "embeddings"      : None,
        "processing_done" : False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────
# Document Processing Pipeline
# ──────────────────────────────────────────────

def process_uploaded_files(uploaded_files) -> bool:
    """
    Run uploaded PDFs through the full pipeline.
    Returns True on success, False on failure.
    """
    try:

        # Step 0 — close any existing vector store to release Windows file locks
        if st.session_state.vectorstore is not None:
            close_vectorstore(st.session_state.vectorstore)
            st.session_state.vectorstore = None
            st.session_state.chain       = None
            gc.collect()
            time.sleep(0.5)

        # Step 1 — save uploaded files to disk
        with st.spinner("💾 Saving uploaded files..."):
            file_paths = []
            for f in uploaded_files:
                path = save_uploaded_file(f)
                file_paths.append(path)

        # Step 2 — load documents
        with st.spinner("📖 Reading documents..."):
            documents, doc_infos          = load_multiple_pdfs(file_paths)
            st.session_state.all_documents = documents

        # Step 3 — chunk documents
        with st.spinner("✂️ Creating smart chunks..."):
            parent_chunks, child_chunks = create_parent_child_chunks(
                documents      = documents,
                parent_size    = 1500,
                child_size     = 300,
                parent_overlap = 100,
                child_overlap  = 30
            )
            doc_infos = update_doc_info_chunk_count(doc_infos, child_chunks)
            st.session_state.doc_infos    = doc_infos
            st.session_state.child_chunks = child_chunks

        # Step 4 — load embedding model (only once per session)
        if st.session_state.embeddings is None:
            with st.spinner("🧠 Loading embedding model (first time only)..."):
                st.session_state.embeddings = get_embedding_model()

        # Step 5 — build vector store
        with st.spinner("🗄️ Building vector store..."):
            vectorstore, docstore = build_vectorstore(
                child_chunks  = child_chunks,
                parent_chunks = parent_chunks,
                embeddings    = st.session_state.embeddings,
                reset         = True
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.docstore    = docstore

        # Step 6 — build retriever and chain
        with st.spinner("🔗 Setting up retriever and chain..."):
            retriever = build_retriever(
                vectorstore   = vectorstore,
                docstore      = docstore,
                all_documents = child_chunks,
                fetch_k       = 12,
                final_k       = 4,
                use_reranking = True
            )
            st.session_state.chain = DocuLegalChain(
                retriever = retriever,
                provider  = "groq"
            )
            st.session_state.processing_done = True

        return True

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


# ──────────────────────────────────────────────
# UI — Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with upload controls and document info."""

    with st.sidebar:

        st.title("⚖️ DocuLegal")
        st.caption("Understand any legal document in plain English.")
        st.divider()

        # Upload section
        st.subheader("📎 Upload Documents")
        uploaded_files = st.file_uploader(
            label                 = "Upload legal PDFs",
            type                  = ["pdf"],
            accept_multiple_files = True,
            help                  = "Upload one or more legal PDFs to analyze"
        )

        if uploaded_files:
            if st.button("⚡ Process Documents", type="primary", use_container_width=True):
                success = process_uploaded_files(uploaded_files)
                if success:
                    st.success(f"✅ {len(uploaded_files)} document(s) ready!")
                    st.session_state.chat_history = []
                    if st.session_state.chain:
                        st.session_state.chain.clear_history()
                    st.rerun()

        # Loaded documents info
        if st.session_state.doc_infos:
            st.divider()
            st.subheader("📋 Loaded Documents")
            for doc_info in st.session_state.doc_infos:
                with st.expander(f"📄 {doc_info.filename}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pages", doc_info.page_count)
                    with col2:
                        st.metric("Chunks", doc_info.chunk_count)

        # Vector store stats
        if st.session_state.vectorstore and st.session_state.docstore:
            st.divider()
            st.subheader("🗄️ Store Stats")
            try:
                stats = get_store_stats(
                    st.session_state.vectorstore,
                    st.session_state.docstore
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Child Chunks",  stats["child_chunks"])
                with col2:
                    st.metric("Parent Chunks", stats["parent_chunks"])
            except Exception:
                pass

        # Clear conversation
        if st.session_state.processing_done:
            st.divider()
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.chain:
                    st.session_state.chain.clear_history()
                st.rerun()

        # Disclaimer
        st.divider()
        st.caption(
            "⚠️ DocuLegal is for informational purposes only and is "
            "not a substitute for professional legal advice."
        )


# ──────────────────────────────────────────────
# UI — Answer Card
# ──────────────────────────────────────────────

def render_answer(display_data: dict):
    """Render a structured legal answer card."""

    # Confidence + answer header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 📋 Answer")
    with col2:
        st.markdown(f"**{display_data['confidence']}**")

    # Main answer text
    st.write(display_data["answer"])

    # Warnings
    if display_data["warnings"]:
        for warning in display_data["warnings"]:
            st.warning(warning)

    # Source clauses
    if display_data["sources"]:
        with st.expander("📌 Source Clauses", expanded=True):
            for i, src in enumerate(display_data["sources"]):
                st.markdown(f"**{src['label']}**")
                if src["excerpt"]:
                    st.markdown(
                        f"<blockquote style='color:#555;font-size:0.9em;"
                        f"border-left:3px solid #ccc;padding-left:10px'>"
                        f"{src['excerpt']}</blockquote>",
                        unsafe_allow_html=True
                    )
                if i < len(display_data["sources"]) - 1:
                    st.divider()


# ──────────────────────────────────────────────
# UI — Welcome Screen
# ──────────────────────────────────────────────

def render_welcome():
    """Show the welcome/onboarding screen before any doc is uploaded."""

    st.markdown("## 👋 Welcome to DocuLegal")
    st.markdown(
        "Upload a legal PDF from the sidebar and start asking questions "
        "in plain English. DocuLegal will find the exact clause and explain "
        "it clearly — with source citations and confidence scores."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Employment Contracts")
        st.markdown("""
- *What is my notice period?*
- *Can they change my salary without consent?*
- *What happens to my stock if I'm terminated?*
- *Do I own work I create outside office hours?*
        """)

    with col2:
        st.markdown("### 🏠 Rental Agreements")
        st.markdown("""
- *Who pays for repairs and maintenance?*
- *Can the landlord enter without notice?*
- *What is the penalty for breaking the lease?*
- *How much notice must I give before leaving?*
        """)

    with col3:
        st.markdown("### 📑 Service Agreements")
        st.markdown("""
- *Who owns the intellectual property I create?*
- *What are the payment terms if a deadline is missed?*
- *How can either party terminate this agreement?*
- *What are the confidentiality obligations?*
        """)

    st.divider()
    st.info("👈 Upload one or more legal PDFs from the sidebar to get started.")


# ──────────────────────────────────────────────
# UI — Main Chat Interface
# ──────────────────────────────────────────────

def render_main():
    """Render the main chat interface."""

    st.title("⚖️ DocuLegal")
    st.subheader("Ask anything about your legal documents — in plain English.")

    # Show welcome screen if no documents loaded yet
    if not st.session_state.processing_done:
        render_welcome()
        return

    # Comparison mode toggle (only when multiple docs loaded)
    comparison_mode = False
    if len(st.session_state.doc_infos) > 1:
        comparison_mode = st.toggle(
            "🔀 Comparison Mode",
            help="Compare clauses across multiple uploaded documents"
        )
        if comparison_mode:
            st.info(
                "Comparison mode active — ask questions like "
                "*'How does the termination clause differ between the two contracts?'*"
            )

    st.divider()

    # Render existing chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                render_answer(message["content"])

    # Chat input
    question = st.chat_input(
        placeholder="Ask a question about your legal document..."
    )

    if question:

        # Show user message immediately
        st.session_state.chat_history.append({
            "role"    : "user",
            "content" : question
        })
        with st.chat_message("user"):
            st.write(question)

        # Generate and show answer
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("🔍 Searching your document..."):
                try:
                    legal_answer = st.session_state.chain.ask(
                        question        = question,
                        comparison_mode = comparison_mode
                    )
                    display_data = format_answer_for_display(legal_answer)

                except Exception as e:
                    display_data = {
                        "answer"      : f"An error occurred: {str(e)}",
                        "confidence"  : "⚫ Error",
                        "sources"     : [],
                        "warnings"    : ["Something went wrong. Please try again."],
                        "is_comparison": False
                    }
                    import traceback
                    print(traceback.format_exc())

            render_answer(display_data)

        # Save to history
        st.session_state.chat_history.append({
            "role"    : "assistant",
            "content" : display_data
        })


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

def main():
    initialize_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()