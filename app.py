import streamlit as st
import requests
import time
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Verifiable RAG Enterprise", page_icon="⚡", layout="wide")

# Custom CSS for SaaS aesthetic
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #111827;
            letter-spacing: -0.02em;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 9999px;
            background-color: #DEF7EC;
            color: #03543F;
        }
        .abstain-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 9999px;
            background-color: #FDE8E8;
            color: #9B1C1C;
        }
        [data-testid="stSidebar"] {
            background-color: #F9FAFB;
            border-right: 1px solid #E5E7EB;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚡ Enterprise Workspace")
    st.caption(f"Session Identifier: `{st.session_state.session_id}`")
    st.markdown("---")

    st.markdown("#### Document Ingestion")
    st.caption("Upload technical reports or SEC filings to build a tenant-isolated hybrid index.")

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("🚀 Ingest & Build Index", use_container_width=True, type="primary"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"session_id": st.session_state.session_id}
                try:
                    res = requests.post(f"{API_URL}/api/v1/upload", files=files, data=data)
                    if res.status_code == 200:
                        job_id = res.json()["job_id"]
                        
                        with st.status(f"Processing '{uploaded_file.name}'...", expanded=True) as status_box:
                            progress_bar = st.progress(0)
                            while True:
                                job_res = requests.get(f"{API_URL}/api/v1/jobs/{job_id}")
                                if job_res.status_code == 200:
                                    job_info = job_res.json()
                                    state = job_info["status"]
                                    prog = job_info.get("progress", 0)
                                    msg = job_info.get("message", "")
                                    progress_bar.progress(prog)
                                    st.write(f"🔄 {msg}")

                                    if state == "COMPLETED":
                                        status_box.update(label=f"Indexed: {uploaded_file.name}", state="complete", expanded=False)
                                        if uploaded_file.name not in st.session_state.indexed_files:
                                            st.session_state.indexed_files.append(uploaded_file.name)
                                        break
                                    elif state == "FAILED":
                                        status_box.update(label="Ingestion Failed", state="error", expanded=True)
                                        st.error(msg)
                                        break
                                time.sleep(1.5)
                    else:
                        st.error(f"Upload failed: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server at http://localhost:8000. Is api.py running?")
        else:
            st.warning("Please select at least one PDF file.")

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("#### Indexed Documents")
        for doc in st.session_state.indexed_files:
            st.markdown(f"📄 `{doc}`")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT PANEL ---
st.markdown('<div class="main-title">⚡ Verifiable RAG Enterprise</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Zero-hallucination document intelligence powered by Gemini, Hybrid Search, and NLI Semantic Verification.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages and not st.session_state.indexed_files:
    st.info("👋 **Welcome!** Upload one or more technical PDFs in the sidebar and click **'Ingest & Build Index'** to begin.")

# Render Conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("abstained") and message.get("abstention_reason"):
            st.warning(f"🛡️ **Guardrail Abstention:** {message['abstention_reason']}")
        if message.get("claims"):
            with st.expander("🔎 View Verified Receipts & NLI Provenance", expanded=False):
                for idx, claim in enumerate(message["claims"], 1):
                    ent_pct = claim.get("entailment_score", 0.0) * 100
                    st.markdown(f"**{idx}. Claim:** {claim['statement']}")
                    st.markdown(f"> *\"{claim['citation']['verbatim_quote']}\"*")
                    st.caption(
                        f"📍 **Source:** `{claim['citation']['source_file']}` (Page {claim['citation']['page_number']}) | "
                        f"🎯 **NLI Entailment Confidence:** `{ent_pct:.1f}%`"
                    )
                    if idx < len(message["claims"]):
                        st.divider()

# Input Bar
if prompt := st.chat_input("Ask a grounded question about your indexed documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Executing Hybrid Retrieval & NLI Semantic Verification..."):
            try:
                payload = {"query": prompt, "session_id": st.session_state.session_id, "top_k": 5}
                res = requests.post(f"{API_URL}/api/v1/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(data["final_answer"])

                    if data.get("abstained") and data.get("abstention_reason"):
                        st.warning(f"🛡️ **Guardrail Abstention:** {data['abstention_reason']}")

                    claims = data.get("claims", [])
                    if claims:
                        with st.expander("🔎 View Verified Receipts & NLI Provenance", expanded=True):
                            for idx, claim in enumerate(claims, 1):
                                ent_pct = claim.get("entailment_score", 0.0) * 100
                                st.markdown(f"**{idx}. Claim:** {claim['statement']}")
                                st.markdown(f"> *\"{claim['citation']['verbatim_quote']}\"*")
                                st.caption(
                                    f"📍 **Source:** `{claim['citation']['source_file']}` (Page {claim['citation']['page_number']}) | "
                                    f"🎯 **NLI Entailment Confidence:** `{ent_pct:.1f}%`"
                                )
                                if idx < len(claims):
                                    st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["final_answer"],
                        "abstained": data.get("abstained", False),
                        "abstention_reason": data.get("abstention_reason"),
                        "claims": claims,
                    })
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to API server. Ensure `python api.py` is running on port 8000.")
