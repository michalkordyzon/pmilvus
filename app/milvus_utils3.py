import os
from dotenv import load_dotenv
load_dotenv()
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonxdata.watsonx_data_v2 import WatsonxDataV2

api_key = os.getenv("API_KEY") or os.getenv("API_KEY")
url = os.getenv("WXDATA_URL")


authenticator = IAMAuthenticator(api_key)
client = WatsonxDataV2(authenticator=authenticator)
client.set_service_url(url)

response = client.some_api_method(params…)
print(response)



