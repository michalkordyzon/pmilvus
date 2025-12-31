import os
from pymilvus import MilvusClient
from typing import List
from typing import Optional
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
#from ibm_watsonxdata.watsonx_data_v2 import ibm_watsonxdata
#from ibm_watsonxdata import create_milvus_service_pause
import ibm_watsonxdata as wx_data
import json
#from ibm_watsonxdata.watsonx_data_v2 import WatsonxDataV2

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
WXDATA_URL = os.getenv("WXDATA_URL")
SERVICE_ID = os.getenv("SERVICE_ID")
INSTANCE_ID = os.getenv("INSTANCE_ID")

# def drop_milvus_collections(
#     client: MilvusClient,
#     collections: List[str],
# ) -> None:
#     """
#     Drop given Milvus collections if they exist.
#     Safe to call multiple times.
#     """
#     for name in collections:
#         try:
#             if client.has_collection(name):
#                 client.drop_collection(name)
#                 print(f"[Milvus] Dropped collection '{name}'")
#             else:
#                 print(f"[Milvus] Collection '{name}' does not exist")
#         except Exception as e:
#             print(f"[Milvus] Failed to drop '{name}': {e}")




def pause_milvus_service(service_id: str, *, auth_instance_id: Optional[str] = None):

    # wx_data.create_milvus_service_pause(
    #     engine_id=SERVICE_ID,
    #     auth_instance_id=INSTANCE_ID,
    # )
    response = wx_data.create_milvus_service_pause(
        service_id=SERVICE_ID,
    )
    success_response = response.get_result()
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(json.dumps(success_response, indent=2))

# response = watsonx_data_service.create_milvus_service_resume(
#   service_id='testString',
# )
# success_response = response.get_result()

# print(json.dumps(success_response, indent=2))


# create_milvus_service_pause(
#         self,
#         service_id: str,
#         *,
#         auth_instance_id: Optional[str] = None,
#         **kwargs,
#     ) -> DetailedResponse