# app.py
import streamlit as st
from functools import lru_cache

from config import COLLECTION_MAP, DEFAULT_TOP_K

from rag_backend import connect_milvus, answer_question, ensure_collection  
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

import os
from dotenv import load_dotenv
load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")  # keep, but don't force into init_milvus unless supported
# print("########################")
# print(MILVUS_HOST)
# print(MILVUS_PORT)
# print(MILVUS_API_KEY)


# Collection names
PUBLIC_COLLECTION = "offerings_public"
MANAGERS_COLLECTION = "offerings_managers_only"

# Embedding model (384 dimensions)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed for this model

# ---------- Embedding model (example) ----------

# @lru_cache(maxsize=1)
# def get_embedder():
#     from sentence_transformers import SentenceTransformer
#     return SentenceTransformer("all-MiniLM-L6-v2")

# def embed_fn(EMBEDDING_MODEL_NAME):
#     # model = get_embedder()
#     # return model.encode(texts).tolist()
#     model = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     return model

# ---------- Embedding model ----------

@st.cache_resource
def get_embedder(model_name: str):
    return SentenceTransformer(model_name)

def embed_fn(texts):
    """
    texts: List[str]
    returns: List[List[float]]  (Milvus-ready)
    """
    model = get_embedder(EMBEDDING_MODEL_NAME)
    return model.encode(texts, convert_to_numpy=True).tolist()




# ---------- Session state ----------
if "milvus_connected" not in st.session_state:
    st.session_state.milvus_connected = False
    st.session_state.client = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "last_backend_error" not in st.session_state:
    st.session_state.last_backend_error = None

# if "collection_created_checked" not in st.session_state:
#     st.session_state.

#st.session_state.client

# ---------- Actions (safe callbacks) ----------

def connect_milvus_bt():
    try:
        # IMPORTANT: keep the signature you already have
        # init_milvus(MILVUS_HOST, MILVUS_PORT, MILVUS_API_KEY)
        res = connect_milvus(MILVUS_HOST, MILVUS_PORT, MILVUS_API_KEY)
        st.session_state.milvus_connected = True
        st.session_state.client = res
        st.session_state.last_backend_error = None
    except Exception as e:
        st.session_state.milvus_connected = False
        st.session_state.last_backend_error = f"Connect failed: {e}"


def load_data_action():
    try:
        from rag_backend import load_data  # <-- only required when button clicked
    except Exception as e:
        st.session_state.last_backend_error = (
            f"`load_data` not available in rag_backend.py yet (or import failed): {e}"
        )
        # return
    try:
        stats = load_data(embed_fn=embed_fn, collection_map=COLLECTION_MAP)
        st.session_state.data_loaded = True
        st.session_state.last_backend_error = None
        st.sidebar.success(f"Loaded OK: {stats}")
    except Exception as e:
        st.session_state.data_loaded = False
        st.session_state.last_backend_error = f"Load failed: {e}"

def create_check_collections(client, PUBLIC_COLLECTION, MANAGERS_COLLECTION):
    try:
        ensure_collection(client, PUBLIC_COLLECTION, EMBEDDING_DIM)
        ensure_collection(client, MANAGERS_COLLECTION, EMBEDDING_DIM)

        st.session_state.is_collection = True
        #st.session_state.last_backend_error = None
        st.sidebar.success(f"Collections: {st.session_state.client.list_collections()}")
    except Exception as e:
        st.session_state.data_loaded = False
        st.session_state.last_backend_error = f"Load failed: {e}"
    try:
        from rag_backend import load_data  # <-- only required when button clicked
    except Exception as e:
        st.session_state.last_backend_error = (
            f"Something is wrong with creating / checking collections: {e}"
        )

    



# ---------- UI ----------
st.set_page_config(page_title="Tender QA (Milvus + Granite)", layout="wide")
st.title("🔍 Tender Q&A assistant")
st.caption("Milvus + LLM RAG • managers vs employees collections")

with st.sidebar:
    st.header("Settings")
    role = st.radio("Your role", ["Employee", "Manager"])
    top_k = st.slider("Number of passages", 1, 10, DEFAULT_TOP_K)
    show_debug = st.checkbox("Show retrieved passages", value=True)

    st.markdown("---")
    st.subheader("Milvus controls")

    st.write("Status:  \n",
             "🟢 connected" if st.session_state.milvus_connected else "🔴 not connected",
             "  \n",
             "🟢 data loaded" if st.session_state.data_loaded else "🔴 data not loaded")

    st.button("🔌 Connect to Milvus", on_click=connect_milvus_bt)
    st.button("📥 Load data", on_click=load_data_action)
    st.button("🗄️ Create/Check collections", 
              on_click=create_check_collections, 
              args=(      
                st.session_state.client,
                PUBLIC_COLLECTION,
                MANAGERS_COLLECTION,
                  
              ))

    if st.session_state.last_backend_error:
        st.error(st.session_state.last_backend_error)

st.markdown(
    f"Current role: **{role}** – you will only search in "
    f"`{COLLECTION_MAP[role]}` collection."
)

question = st.text_area(
    "Ask a question about tenders:",
    placeholder="e.g. What are the evaluation criteria for IT infrastructure tender?",
    height=100,
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Type your question first.")
    elif not st.session_state.milvus_connected:
        st.warning("Connect to Milvus first (sidebar).")
    else:
        with st.spinner("Searching Milvus and generating answer..."):
            result = answer_question(
                question=question,
                role=role,
                embed_fn=embed_fn,
                collection_map=COLLECTION_MAP,
                top_k=top_k,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        if show_debug:
            st.subheader("Sources")
            for i, p in enumerate(result["passages"], start=1):
                with st.expander(f"Source {i} (score={p['score']:.4f})"):
                    st.write(p["text"])
                    if p.get("source"):
                        st.caption(f"Source: {p['source']}")
