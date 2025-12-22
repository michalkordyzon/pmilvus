from milvus_utils import get_wxd_client
client = get_wxd_client()
#print([m for m in dir(client) if "pause" in m.lower()])
print([m for m in dir(client)])
#print([m for m in dir(client) if "resume" in m.lower()])
