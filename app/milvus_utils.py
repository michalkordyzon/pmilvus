import os
from pymilvus import MilvusClient
from typing import List
from typing import Optional
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
#from ibm_watsonxdata.watsonx_data_v2 import ibm_watsonxdata
#from ibm_watsonxdata.watsonx_data_v2 import WatsonxDataV2

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
WXDATA_URL = os.getenv("WXDATA_URL")

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


def get_wxd_client(api_key: Optional[str] = None, url: Optional[str] = None):
    api_key = api_key or API_KEY
    url = url or WXDATA_URL

    if not api_key:
        raise ValueError("Missing API_KEY")
    if not url:
        raise ValueError("Missing WXDATA_URL")

    authenticator = IAMAuthenticator(api_key)

    # Your environment supports WatsonxDataV2 (as seen in your traceback)
    from ibm_watsonxdata.watsonx_data_v2 import WatsonxDataV2
    client = WatsonxDataV2(authenticator=authenticator)

    # Must be base host only, e.g. https://api.eu-gb.watsonxdata.cloud.ibm.com
    client.set_service_url(url)
    return client

# def get_wxd_client(
#     api_key: Optional[str] = None,
#     url: Optional[str] = None,
#     ) -> ibm_watsonxdata:
#     api_key = API_KEY
#     url = WXDATA_URL

#     if not api_key:
#         raise ValueError("Missing WATSONXDATA_APIKEY")
#     if not url:
#         raise ValueError("Missing WATSONXDATA_URL")

#     authenticator = IAMAuthenticator(api_key)


#     client = ibm_watsonxdata(authenticator=authenticator)
#     client.set_service_url(url)
#     return client

def pause_milvus_service(service_id: str, *, auth_instance_id: Optional[str] = None):
    client = get_wxd_client()
    return client.create_engine_pause(
        engine_id=service_id,
        auth_instance_id=auth_instance_id,
    )


# def pause_milvus_service(
#     service_id: str,
#     *,
#     auth_instance_id: Optional[str] = None,
# ):
#     client = get_wxd_client()
#     return client.create_milvus_service_pause(
#         service_id=service_id,
#         auth_instance_id=auth_instance_id,
#     )
