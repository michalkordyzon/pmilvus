# app.py
import streamlit as st
from functools import lru_cache

#from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_MAP, DEFAULT_TOP_K
from config import COLLECTION_MAP, DEFAULT_TOP_K
from rag_backend import init_milvus, answer_question

import os
from dotenv import load_dotenv
load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST")     
MILVUS_PORT = os.getenv("MILVUS_PORT") 
MILVUS_API_KEY= os.getenv("MILVUS_API_KEY") 


# ---------- Embedding model (example) ----------

@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def embed_fn(texts):
    model = get_embedder()
    return model.encode(texts).tolist()  # Milvus expects plain Python lists

# ---------- Init Milvus once ----------

@st.cache_resource
def _init_milvus():
    init_milvus(MILVUS_HOST, MILVUS_PORT)
    return True

# _init_milvus()

# ---------- Streamlit layout ----------

st.set_page_config(page_title="Tender QA (Milvus + Granite)", layout="wide")

st.title("🔍 Tender Q&A assistant")
st.caption("Milvus + LLM RAG • managers vs employees collections")

with st.sidebar:
    st.header("Settings")
    role = st.radio("Your role", ["Employee", "Manager"])
    top_k = st.slider("Number of passages", 1, 10, DEFAULT_TOP_K)
    show_debug = st.checkbox("Show retrieved passages", value=True)

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
