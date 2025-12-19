# app.py
import streamlit as st
#from functools import lru_cache
from config import COLLECTION_MAP, DEFAULT_TOP_K

from rag_backend import connect_milvus, answer_question, ensure_collection, \
    prep_embedding, ingest_pdf_to_collection  
from milvus_utils import drop_milvus_collections, pause_milvus_service
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
from pprint import pprint

import os
from dotenv import load_dotenv
load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
API_KEY = os.getenv("API_KEY")  # keep, but don't force into init_milvus unless supported
SERVICE_ID = os.getenv("SERVICE_ID")
AUTH_INSTANCE_ID = None  # or from st.secrets / env

# print("########################")
# print(MILVUS_HOST)
# print(MILVUS_PORT)
# print(MILVUS_API_KEY)

PUBLIC_PDF_PATH = "./data/offerings_public.pdf"
MANAGERS_PDF_PATH = "./data/offerings_managers_only.pdf"

PUBLIC_COLLECTION = "offerings_public"
MANAGERS_COLLECTION = "offerings_managers_only"

# Embedding model (384 dimensions)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed for this model

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

if "is_embedding" not in st.session_state:
    st.session_state.is_embedding = False

if "last_backend_error" not in st.session_state:
    st.session_state.last_backend_error = None

if "is_collection" not in st.session_state:
    st.session_state.is_collection = False

if "data_is_loaded" not in st.session_state:
    st.session_state.data_is_loaded = False

if "is_sample" not in st.session_state:
    st.session_state.is_sample = False




# ---------- Actions (safe callbacks) ----------

def connect_milvus_bt():
    try:
        # IMPORTANT: keep the signature you already have
        # init_milvus(MILVUS_HOST, MILVUS_PORT, MILVUS_API_KEY)
        res = connect_milvus(MILVUS_HOST, MILVUS_PORT, API_KEY)
        st.session_state.milvus_connected = True
        st.session_state.client = res
        st.session_state.last_backend_error = None
        st.success(f"Connected to Milvus")
    except Exception as e:
        st.session_state.milvus_connected = False
        st.session_state.last_backend_error = f"Connect failed: {e}"


def prepare_embedding():
    try:
        stats = prep_embedding(embed_fn=embed_fn, collection_map=COLLECTION_MAP)
        st.session_state.is_embedding = True
        st.session_state.last_backend_error = None
        #st.sidebar.success(f"Embedding prepared. Stats: {stats}")
        st.success(f"Embedding prepared. Stats: {stats}")
        #bottom_status.success(f"Embedding prepared. Stats: {stats}")
    except Exception as e:
        st.session_state.is_embedding = False
        st.session_state.last_backend_error = f"Load failed: {e}"

def prepare_collections(client, PUBLIC_COLLECTION, MANAGERS_COLLECTION):
    try:
        ensure_collection(client, PUBLIC_COLLECTION, EMBEDDING_DIM)
        ensure_collection(client, MANAGERS_COLLECTION, EMBEDDING_DIM)
        
        st.session_state.is_collection = True
        #st.session_state.last_backend_error = None
        st.success(f"Collections created.")
    except Exception as e:
        st.session_state.is_collection = False
        st.session_state.last_backend_error = f"Something wrong creating collections: {e}"
   
    try:
        st.session_state.client.load_collection(PUBLIC_COLLECTION)
        st.session_state.client.load_collection(MANAGERS_COLLECTION)
        st.success(f"Collections loaded: {st.session_state.client.list_collections()}")
    except Exception as e:
        st.session_state.is_collection = False
        st.session_state.last_backend_error = f"Something wrong with loading collections: {e}"

# def load_data():
#     try:
#         client = st.session_state.client
#         if client is None:
#             st.warning("Connect to Milvus first.")
#             return

#         # Public
#         ingest_pdf_to_collection(
#             client=client,
#             collection_name=PUBLIC_COLLECTION,
#             pdf_path=PUBLIC_PDF_PATH,
#             offering_id="offering_xyz",
#             model=get_embedder(EMBEDDING_MODEL_NAME),
#         )

#         # Managers
#         ingest_pdf_to_collection(
#             client=client,
#             collection_name=MANAGERS_COLLECTION,
#             pdf_path=MANAGERS_PDF_PATH,
#             offering_id="offering_xyz",
#             model=get_embedder(EMBEDDING_MODEL_NAME),
#         )

#         st.session_state.data_is_loaded = True
#         st.success("Data loaded into BOTH collections.")
#     except Exception as e:
#         st.session_state.data_is_loaded = False
#         st.session_state.last_backend_error = f"Load data failed: {type(e).__name__}: {e}"
#         st.error(st.session_state.last_backend_error)



def load_data():
    try:
        client = st.session_state.client
        if client is None:
            st.warning("Connect to Milvus first.")
            return

        # Public
        ingest_pdf_to_collection(
            client=client,
            collection_name=PUBLIC_COLLECTION,
            pdf_path=PUBLIC_PDF_PATH,
            offering_id="offering_xyz",
            model=get_embedder(EMBEDDING_MODEL_NAME),
        )

        # Managers
        ingest_pdf_to_collection(
            client=client,
            collection_name=MANAGERS_COLLECTION,
            pdf_path=MANAGERS_PDF_PATH,
            offering_id="offering_xyz",
            model=get_embedder(EMBEDDING_MODEL_NAME),
        )

        st.session_state.data_is_loaded = True
        st.success("Data loaded into BOTH collections.")
    except Exception as e:
        st.session_state.data_is_loaded = False
        st.session_state.last_backend_error = f"Load data failed: {type(e).__name__}: {e}"
        st.error(st.session_state.last_backend_error)


def sample_col():
    try:
        rows_public = st.session_state.client.query(
                collection_name=PUBLIC_COLLECTION,
                filter="offering_id == 'offering_xyz'",
                output_fields=["id", "offering_id", "text"],
                limit=5,
            )
    
        for r in rows_public:
            #print("----")
            #pprint(r)
            st.write("---")
            st.write(r)
        st.session_state.is_sample= True
        st.success(f"Data sampling works.")
    except Exception as e:
        st.session_state.is_sample = False
        st.session_state.last_backend_error = f"Something is wrong with sampling vector db: {e}"


# def drop_milvus_coll():
#     drop_milvus_collections(
#     client=st.session_state.client,
#     collections=[
#         "offerings_public",
#         "offerings_managers_only",
#     ],)
#     st.session_state.is_collection = False
#     st.session_state.data_is_loaded = False

def drop_milvus_coll():
    try:
        if st.session_state.client is None:
            st.warning("Connect first.")
            return

        drop_milvus_collections(
            client=st.session_state.client,
            collections=[PUBLIC_COLLECTION, MANAGERS_COLLECTION],
        )
        st.session_state.is_collection = False
        st.session_state.data_is_loaded = False
        st.session_state.is_sample = False
        st.success("Collections dropped.")
    except Exception as e:
        st.session_state.last_backend_error = f"Drop failed: {type(e).__name__}: {e}"
        st.error(st.session_state.last_backend_error)





# ------------------------
# ---------- UI ----------
# ------------------------

st.set_page_config(page_title="RAG Demo (Milvus + Granite)", layout="wide")
st.title("🔍 Multitenant Milvus RAG DEMO")
st.caption("Milvus + LLM RAG • managers vs employees collections")

with st.sidebar:
    st.subheader("Quick Setup:")

    # st.write("Status:  \n",
    #          "🟢 connected" if st.session_state.milvus_connected else "🔴 not connected",
    #          "  \n",
    #          "🟢 embedding ready" if st.session_state.is_embedding else "🔴 no embedding")
    st.write("")
    st.button("1 - __Connect to Milvus 🔌", on_click=connect_milvus_bt)
    st.write("🟢 Connected" if st.session_state.milvus_connected else "🔴 Not connected")

    st.write("")
    st.button("2 - Prepare embedding 🧩", on_click=prepare_embedding)
    st.write("🟢 Embedding ready" if st.session_state.is_embedding else "🔴 No embedding")

    st.write("")
    st.button("3 - _Prepare collections 🗄️",
              on_click=prepare_collections,
              args=(      
                st.session_state.client,
                PUBLIC_COLLECTION,
                MANAGERS_COLLECTION,
              ))
    st.write("🟢 Collections ready" if st.session_state.is_collection else "🔴 No collections")

    st.write("")
    st.button("4 - ________Load data 📥", on_click=load_data)
    st.write("🟢 Data load made" if st.session_state.data_is_loaded else "🔴 No collections")

    st.write("")
    st.button("5 - __Sample collection 🧪", on_click=sample_col)
    st.write("🟢 Sample made" if st.session_state.is_sample else "🔴 No sample")
    
    st.markdown("---")

    st.header("Settings")
    role = st.radio("Your role", ["Employee", "Manager"])
    top_k = st.slider("Number of passages", 1, 10, DEFAULT_TOP_K)
    show_debug = st.checkbox("Show retrieved passages", value=True)

    st.markdown("---")
    st.header("Milvus controls")
    
    st.button("DROP Milvus collections ", on_click=drop_milvus_coll)
    #st.write("🟢 Sample made" if st.session_state.is_sample else "🔴 No sample")
    
    # better way -> runs only the if script, not entire app.py
    if st.button("⏸️ Pause Milvus", disabled=not SERVICE_ID):
        try:
            resp = pause_milvus_service(
                service_id=SERVICE_ID,
                auth_instance_id=AUTH_INSTANCE_ID,
            )
            st.success(f"Pause request sent (HTTP {resp.get_status_code()})")
            st.json(resp.get_result())
        except Exception as e:
            st.error("Failed to pause Milvus")
            st.exception(e)
# #dodaj te dwie rzeczy:
# AUTH_INSTANCE_ID = None  # or from st.secrets / env
# service_id = st.text_input("Milvus service_id")
# https address

# # to juz jest
# MILVUS_HOST = os.getenv("MILVUS_HOST")
# MILVUS_PORT = os.getenv("MILVUS_PORT")
# MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")



st.markdown(
    f"Current role: **{role}** – you will only search in "
    f"`{COLLECTION_MAP[role]}` collection."
)

question = st.text_area(
    "Ask a question about tenders:",
    placeholder="TravelFlex",
    height=100,
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Type your question first.")
    elif not st.session_state.milvus_connected:
        st.warning("You are not connected to Milvus - go through a setup using buttons 1-5 on the sidebar.")
    else:
        with st.spinner("Searching Milvus and generating answer..."):
            # result = answer_question(
            #     question=question,
            #     role=role,
            #     embed_fn=embed_fn,
            #     collection_map=COLLECTION_MAP,
            #     top_k=top_k,
            # )
                result = answer_question(
                client=st.session_state.client,
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


#bottom_status = st.empty()

# SANITY CHECKS
#st.write("Collections visible:", st.session_state.client.list_collections())
#st.write("Schema:", st.session_state.client.describe_collection("offerings_public"))
# st.write("Public count:", st.session_state.client.get_collection_stats(PUBLIC_COLLECTION))
# st.write("Managers count:", st.session_state.client.get_collection_stats(MANAGERS_COLLECTION))
