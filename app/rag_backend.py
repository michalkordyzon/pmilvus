# rag_backend.py
from pymilvus import connections, Collection, MilvusClient, DataType
from typing import List, Dict
from pypdf import PdfReader
import time

# def connect_milvus(MILVUS_HOST, MILVUS_PORT,MILVUS_API_KEY) -> MilvusClient:
#     if not (MILVUS_HOST and MILVUS_PORT and MILVUS_API_KEY):
#         raise RuntimeError("Set MILVUS_HOST, MILVUS_PORT and MILVUS_API_KEY first.")
	

#     milvus_uri = f"https://ibmlhapikey_michal.kordyzon@pl.ibm.com:{MILVUS_API_KEY}@{MILVUS_HOST}:{MILVUS_PORT}"
    
#     client = MilvusClient(
#         uri=milvus_uri,
#         secure=True,           # SaaS Milvus is always TLS
#     )
#     return client


#def init_milvus(host: str, port: str):
# def init_milvus(MILVUS_HOST: str, MILVUS_PORT: str,MILVUS_API_KEY: str):
#     #connections.connect(alias="default", host=host, port=port)

#     client = connect_milvus(x, y, z)
#     print("Connected to IBM Milvus.")



# def connect_milvus(MILVUS_HOST, MILVUS_PORT, MILVUS_API_KEY) -> MilvusClient:
#     if not (MILVUS_HOST and MILVUS_PORT and MILVUS_API_KEY):
#         raise RuntimeError("Set MILVUS_HOST, MILVUS_PORT and MILVUS_API_KEY first.")
    
#     milvus_uri = f"https://ibmlhapikey_michal.kordyzon@pl.ibm.com:{MILVUS_API_KEY}@{MILVUS_HOST}:{MILVUS_PORT}"
    
#     client = MilvusClient(
#         uri=milvus_uri,
#         secure=True,           # SaaS Milvus is always TLS
#     )
#     return client





# ---------- connect to Milvus ----------

def connect_milvus(MILVUS_HOST, MILVUS_PORT, MILVUS_API_KEY) -> MilvusClient:
    if not (MILVUS_HOST and MILVUS_PORT and MILVUS_API_KEY):
        raise RuntimeError("Set MILVUS_HOST, MILVUS_PORT and MILVUS_API_KEY first.")

    milvus_uri = f"https://ibmlhapikey_michal.kordyzon@pl.ibm.com:{MILVUS_API_KEY}@{MILVUS_HOST}:{MILVUS_PORT}"

    # # IMPORTANT: make default connection for Collection()
    # connections.connect(alias="default", uri=milvus_uri, secure=True)

    # optional client handle (useful later if you switch to client.search)
    client = MilvusClient(uri=milvus_uri, secure=True)
    return client



def get_collection(collection_name: str) -> Collection:
    return Collection(collection_name)

def semantic_search(
    collection_name: str,
    query: str,
    embed_fn,
    top_k: int = 5
) -> List[Dict]:
    col = get_collection(collection_name)
    q_emb = embed_fn([query])  # shape: (1, dim)
    results = col.search(
        data=q_emb,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["id", "text", "source"]
    )
    hits = results[0]
    return [
        {
            "text": h.entity.get("text"),
            "source": h.entity.get("source"),
            "score": h.distance,
        }
        for h in hits
    ]

# ---------- LLM call ----------

def build_prompt(question: str, passages: List[Dict], role: str) -> str:
    context = "\n\n".join(
        [f"[{i+1}] {p['text']}" for i, p in enumerate(passages)]
    )
    instructions = (
        "You are answering a question about public tender documents.\n"
        f"The user is in the role: {role}.\n"
        "Answer only based on the context below. If not present, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return instructions



def call_llm(prompt: str) -> str:
    # Plug in watsonx / OpenAI / local Granite, etc.
    # pseudo:
    # response = client.generate(model="granite-... ", prompt=prompt, ...)
    # return response.text
    return "LLM answer placeholder"



def answer_question(
    question: str,
    role: str,
    embed_fn,
    collection_map: Dict[str, str],
    top_k: int = 5
) -> Dict:
    collection_name = collection_map[role]
    passages = semantic_search(collection_name, question, embed_fn, top_k=top_k)
    prompt = build_prompt(question, passages, role)
    answer = call_llm(prompt)
    return {"answer": answer, "passages": passages}

def prep_embedding(embed_fn, collection_map):
    # Warm-up: force model load + sanity check dim
    v = embed_fn(["ping"])[0]
    return {
        "ok": True,
        "embedding_dim": len(v),
        "collections": list(collection_map.values()),
    }


def ensure_collection(client: MilvusClient, collection_name: str, dim: int):
    """
    Create collection if it does not exist, then create a simple FLAT index
    on the embedding field (works well on IBM Milvus).
    """
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return

    # 1. Define schema
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )

    schema.add_field(
        field_name="offering_id",
        datatype=DataType.VARCHAR,
        max_length=256,
    )

    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=2048,
    )

    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=dim,
    )

    # 2. Create collection (no index yet)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    print(f"Created collection '{collection_name}'.")

    # 3. Create index on embedding field
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",      # safe + supported everywhere
        metric_type="COSINE",   # good for sentence-transformers
        # no extra params needed for FLAT
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )
    print(f"Created FLAT index on '{collection_name}.embedding'.")




def ingest_pdf_to_collection(
    client: MilvusClient,
    collection_name: str,
    pdf_path: str,
    offering_id: str,
    model,
):
    """Load a PDF, chunk it, embed, and insert into the given collection (with debug prints)."""
    print(f"\nIngesting '{pdf_path}' into collection '{collection_name}' for offering_id='{offering_id}'")
    t0 = time.time()

    # 1. Read PDF
    print("  [1] Reading PDF...")
    full_text = load_pdf_text(pdf_path)
    print(f"      PDF length (chars): {len(full_text)}")

    # 2. Chunk
    print("  [2] Chunking text...")
    chunks = chunk_text(full_text, max_tokens=256, overlap=20)
    print(f"      Number of chunks: {len(chunks)}")
    if len(chunks) == 0:
        print("      WARNING: no chunks produced, skipping insert.")
        return

    # 3. Embeddings
    print("  [3] Embedding chunks...")
    t_embed_start = time.time()
    embeddings = embed_chunks(model, chunks)
    print(f"      Generated {len(embeddings)} embeddings in {time.time() - t_embed_start:.2f} s")

    # 4. Build rows
    print("  [4] Building insert payload...")
    rows = build_insert_payload(offering_id, chunks, embeddings)
    print(f"      Rows to insert: {len(rows)}")

    # 5. Insert into Milvus
    print("  [5] Inserting into Milvus...")
    t_ins_start = time.time()
    res = client.insert(
        collection_name=collection_name,
        data=rows,
        # you can add timeout if needed (if supported in your pymilvus version):
        # timeout=60,
    )
    print(f"      Insert done in {time.time() - t_ins_start:.2f} s")
    print(f"  [✓] Ingestion completed in {time.time() - t0:.2f} s")

    # Optional: print a tiny summary of res
    try:
        print("      Insert response keys:", list(res.keys()))
    except Exception:
        pass


def load_pdf_text(pdf_path: str) -> str:
    """Read PDF and return the full concatenated text."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)

def chunk_text(text: str, max_tokens: int = 256, overlap: int = 20) -> list[str]:
    """
    Simple, safe chunking:
    - If text is short, returns a single chunk.
    - For longer text, creates overlapping windows without infinite loops.
    """
    tokens = text.split()
    if not tokens:
        return []

    # If the text is shorter than one window, just return it as a single chunk
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]

    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + max_tokens, n)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end == n:  # we've reached the end; break to avoid infinite loop
            break

        # Move window forward with overlap
        start = max(0, end - overlap)

    return chunks

def embed_chunks(model, chunks: List[str]) -> List[List[float]]:
    """Embed a list of text chunks using a SentenceTransformer model."""
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def build_insert_payload(
    offering_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> List[Dict]:
    """
    Build Milvus insert payload: one dict per row, with fields matching schema.
    """
    assert len(chunks) == len(embeddings)
    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append(
            {
                # 'id' is auto_id, so we omit it
                "offering_id": offering_id,
                "text": chunk,
                "embedding": emb,
            }
        )
    return rows
