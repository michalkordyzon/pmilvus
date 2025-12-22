from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonxdata.watsonx_data_v2 import WatsonxDataV2


# Replace with your actual values
API_KEY = "C3Bsce8TEU4mcNmGw31sLOyRrVQ2sKK2L-0UxS-OLcXs"
WXDATA_URL = "https://eu-gb.lakehouse.cloud.ibm.com"
MILVUS_ENGINE_ID = "milvus238"        # e.g., "milvus238"
WATSONX_INSTANCE_ID = "crn:v1:bluemix:public:lakehouse:eu-gb:a/2311e98e393e41b2867a3d538ab3837e:f64ecc48-3824-42c0-8cb8-cd6d9f363756::"  # CRN or GUID of your watsonx.data instance

# 1. Authenticate
authenticator = IAMAuthenticator(API_KEY)
client = WatsonxDataV2(authenticator=authenticator)

# 2. Set the proper watsonx.data URL
client.set_service_url(WXDATA_URL)

# 3. Call the pause operation
try:
    response = client.pause_milvus_service(
        engine_id=MILVUS_ENGINE_ID,
        auth_instance_id=WATSONX_INSTANCE_ID
    )
    print(f"Status Code: {response.get_status_code()}")
    print("Result:")
    print(response.get_result())

except AttributeError as e:
    print("❌ SDK may be outdated. Upgrade using: pip install --upgrade ibm-watsonxdata")
    print("Details:", e)

except Exception as e:
    print("❌ Request failed.")
    print("Details:", e)

