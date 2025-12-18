from pymilvus import MilvusClient
from typing import List

def drop_milvus_collections(
    client: MilvusClient,
    collections: List[str],
) -> None:
    """
    Drop given Milvus collections if they exist.
    Safe to call multiple times.
    """
    for name in collections:
        try:
            if client.has_collection(name):
                client.drop_collection(name)
                print(f"[Milvus] Dropped collection '{name}'")
            else:
                print(f"[Milvus] Collection '{name}' does not exist")
        except Exception as e:
            print(f"[Milvus] Failed to drop '{name}': {e}")


def drop_all_milvus_collections(client: MilvusClient) -> None:
    """
    Drop all collections in the current Milvus database.
    DEV ONLY.
    """
    try:
        cols = client.list_collections()
        for name in cols:
            client.drop_collection(name)
            print(f"[Milvus] Dropped collection '{name}'")
    except Exception as e:
        print(f"[Milvus] Failed to drop collections: {e}")
