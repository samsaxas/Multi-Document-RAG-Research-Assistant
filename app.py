"""
Streamlit chat UI for the RAG-Based Multi-Document Research Analyst.
Run with: streamlit run app.py
"""
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from rag_pipeline import build_index, ask, DOCS_DIR

st.set_page_config(page_title="Multi-Document Research Analyst", page_icon="📄", layout="wide")

# Initialize session state for multiple chat sessions
if "sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.sessions = {
        initial_id: {
            "title": "New Chat",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": []
        }
    }
    st.session_state.current_session_id = initial_id

if "current_session_id" not in st.session_state or st.session_state.current_session_id not in st.session_state.sessions:
    if st.session_state.sessions:
        st.session_state.current_session_id = next(iter(st.session_state.sessions.keys()))
    else:
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {
            "title": "New Chat",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": []
        }
        st.session_state.current_session_id = new_id

# --- Sidebar UI ---
with st.sidebar:
    st.header("💬 Chats")
    
    # + New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {
            "title": "New Chat",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": []
        }
        st.session_state.current_session_id = new_id
        st.rerun()

    st.markdown("---")
    st.subheader("📜 Chat History")
    
    session_ids = list(st.session_state.sessions.keys())
    for sid in reversed(session_ids):
        sess = st.session_state.sessions[sid]
        title = sess.get("title", "New Chat")
        is_active = (sid == st.session_state.current_session_id)
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            button_label = f"💬 {title}"
            if st.button(
                button_label, 
                key=f"sess_{sid}", 
                use_container_width=True,
                type="secondary" if not is_active else "primary"
            ):
                st.session_state.current_session_id = sid
                st.rerun()
        with col2:
            if len(st.session_state.sessions) > 1:
                if st.button("🗑️", key=f"del_{sid}", help="Delete chat"):
                    del st.session_state.sessions[sid]
                    if st.session_state.current_session_id == sid:
                        st.session_state.current_session_id = next(iter(st.session_state.sessions.keys()))
                    st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Settings & Info")
    
    backend_kind = st.radio(
        "Embedding backend",
        ["tfidf", "huggingface"],
        help="tfidf = local fallback (works without internet). "
             "huggingface = production backend (sentence-transformers), needs internet access to download the model.",
    )
    
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    if not has_key:
        st.warning("GEMINI_API_KEY not set — answers will show retrieved context only, not a generated response. "
                   "Set the env var and restart to enable real generation.")
    
    st.markdown("**Indexed documents:**")
    if os.path.exists(DOCS_DIR):
        for f in sorted(os.listdir(DOCS_DIR)):
            st.markdown(f"- `{f}`")


@st.cache_resource
def get_store(backend_kind):
    return build_index(DOCS_DIR, backend_kind=backend_kind)


store = get_store(backend_kind)

# --- Main Chat Area ---
current_session = st.session_state.sessions[st.session_state.current_session_id]

st.title("📄 Multi-Document Research Analyst")
st.caption("RAG pipeline: PDF parsing → chunking → FAISS retrieval → Gemini generation with a grounding guardrail")

# Render active session messages
for turn in current_session["messages"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("🔍 Retrieved context & citations"):
            for r in turn.get("retrieved", []):
                st.markdown(f"**[{r['source']}, p.{r['page']}]** *(score: {r['score']:.3f})*")
                st.text(r["text"][:300])

# Handle new user input
question = st.chat_input("Ask a question about the indexed documents...")
if question:
    with st.chat_message("user"):
        st.write(question)
    
    with st.spinner("Searching documents and generating grounded answer..."):
        result = ask(question, store, use_llm=True)
    
    # Update title if it's the first message in this chat
    if current_session["title"] == "New Chat":
        clean_title = question.strip()
        if len(clean_title) > 28:
            clean_title = clean_title[:28] + "..."
        current_session["title"] = clean_title
        
    current_session["messages"].append(result)
    st.rerun()
