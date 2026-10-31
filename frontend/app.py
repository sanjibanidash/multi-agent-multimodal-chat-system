"""
frontend/app.py
----------------
Streamlit frontend for the Multi-Agent Chat System.

Features:
  - Sidebar: session management, agent selector, file upload
  - Chat window: message history with agent badges
  - SQL view: shows generated SQL + result table
  - RAG view: shows source chunks used for the answer
  - Cache status indicator
"""
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Chat",
    page_icon="🤖",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # local display history


# ── Helper: call the /chat endpoint ──────────────────────────────────────────
def send_chat(message: str, agent: str, file=None) -> dict:
    data = {"message": message, "agent": agent if agent != "Auto" else ""}
    if st.session_state.session_id:
        data["session_id"] = st.session_state.session_id

    files_payload = None
    if file is not None:
        files_payload = {"file": (file.name, file.getvalue(), file.type)}

    resp = requests.post(f"{API_BASE}/chat", data=data, files=files_payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def upload_document(file) -> dict:
    files_payload = {"file": (file.name, file.getvalue(), file.type)}
    data = {}
    if st.session_state.session_id:
        data["session_id"] = st.session_state.session_id
    resp = requests.post(f"{API_BASE}/upload/document", files=files_payload, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Agent selector
    agent_choice = st.selectbox(
        "Agent",
        ["Auto", "multimodal", "sql", "rag"],
        help="'Auto' lets the system route your request to the best agent.",
    )

    st.divider()

    # File upload
    st.subheader("📎 File Upload")
    uploaded_file = st.file_uploader(
        "Upload image, PDF, TXT, or DOCX",
        type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "docx"],
        help=(
            "Images → Multimodal Agent\n"
            "PDF/TXT/DOCX → choose 'rag' agent or 'Auto'\n"
            "For RAG: click 'Ingest Document' first."
        ),
    )

    if uploaded_file and uploaded_file.type not in (
        "image/png", "image/jpeg", "image/gif", "image/webp"
    ):
        if st.button("📥 Ingest Document into RAG"):
            with st.spinner("Ingesting document..."):
                try:
                    info = upload_document(uploaded_file)
                    # Save session_id returned by server
                    st.session_state.session_id = info.get(
                        "collection", st.session_state.session_id
                    )
                    st.success(
                        f"✅ Ingested **{info['filename']}** "
                        f"({info.get('chunks_stored', '?')} chunks)"
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # Session management
    st.subheader("🔑 Session")
    st.caption(f"Session ID: `{st.session_state.session_id or 'Not started'}`")

    if st.button("🗑️ Clear Session"):
        if st.session_state.session_id:
            try:
                requests.delete(f"{API_BASE}/session/{st.session_state.session_id}", timeout=10)
            except Exception:
                pass
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🤖 Multi-Agent Multimodal Chat")
st.caption(
    "Powered by **Multimodal Agent** · **SQL Agent** · **RAG Agent**  |  "
    "Ask anything, upload files, or query your business data."
)

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        # Agent badge for assistant messages
        if role == "assistant":
            agent_used = msg.get("agent_used", "")
            cache_hit = msg.get("cache_hit", False)
            badge = {"multimodal": "🧠 Multimodal", "sql": "🗄️ SQL", "rag": "📚 RAG"}.get(
                agent_used, agent_used
            )
            cols = st.columns([3, 1])
            cols[0].markdown(f"**Agent:** `{badge}`")
            if cache_hit:
                cols[1].success("⚡ Cache Hit")

        st.markdown(msg["content"])

        # SQL-specific extras
        if msg.get("sql_query"):
            with st.expander("🔍 Generated SQL Query"):
                st.code(msg["sql_query"], language="sql")
        if msg.get("raw_results"):
            with st.expander("📊 Raw Database Results"):
                st.dataframe(msg["raw_results"])

        # RAG-specific extras
        if msg.get("sources"):
            with st.expander("📄 Source Chunks Used"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source_file']}** (chunk {src['chunk_index']}, "
                        f"relevance: {src['relevance_score']})\n\n> {src['text_preview']}"
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question…")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                file_to_send = uploaded_file if uploaded_file else None
                # Only send image files directly through chat; documents should be ingested first
                if file_to_send and file_to_send.type not in (
                    "image/png", "image/jpeg", "image/gif", "image/webp"
                ):
                    file_to_send = None  # Don't re-send already-ingested docs

                resp = send_chat(user_input, agent_choice, file_to_send)

                # Update session_id from server
                st.session_state.session_id = resp.get("session_id", st.session_state.session_id)

                answer = resp.get("answer", "No response.")
                agent_used = resp.get("agent_used", "")
                cache_hit = resp.get("cache_hit", False)

                badge = {"multimodal": "🧠 Multimodal", "sql": "🗄️ SQL", "rag": "📚 RAG"}.get(
                    agent_used, agent_used
                )

                cols = st.columns([3, 1])
                cols[0].markdown(f"**Agent:** `{badge}`")
                if cache_hit:
                    cols[1].success("⚡ Cache Hit")

                st.markdown(answer)

                if resp.get("sql_query"):
                    with st.expander("🔍 Generated SQL Query"):
                        st.code(resp["sql_query"], language="sql")
                if resp.get("raw_results"):
                    with st.expander("📊 Raw Database Results"):
                        st.dataframe(resp["raw_results"])
                if resp.get("sources"):
                    with st.expander("📄 Source Chunks Used"):
                        for src in resp["sources"]:
                            st.markdown(
                                f"**{src['source_file']}** (chunk {src['chunk_index']}, "
                                f"relevance: {src['relevance_score']})\n\n> {src['text_preview']}"
                            )

                # Save to local history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "agent_used": agent_used,
                    "cache_hit": cache_hit,
                    "sql_query": resp.get("sql_query"),
                    "raw_results": resp.get("raw_results"),
                    "sources": resp.get("sources"),
                })

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to the backend. "
                    "Make sure FastAPI is running: `uvicorn backend.main:app --reload`"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
