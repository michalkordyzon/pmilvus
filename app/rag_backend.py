# rag_backend.py
from pymilvus import connections, Collection
from typing import List, Dict

# ---------- Milvus setup ----------

def init_milvus(host: str, port: str):
    connections.connect(alias="default", host=host, port=port)

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

# rag_backend.py
def load_data(embed_fn, collection_map):
    """
    Minimal placeholder: implement ingestion later.
    Return stats so Streamlit can show success.
    """
    return {"ok": True, "note": "load_data() not implemented yet"}
