"""
- PyMilvus (MilvusClient) uses gRPC address (host:port).
- watsonx.data pause/resume uses REST API host (base URL).
"""

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

import requests
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

try:
    # PyMilvus 2.4+ usually exposes MilvusClient here
    from pymilvus import MilvusClient  # type: ignore
except Exception:  # pragma: no cover
    MilvusClient = object  # type: ignore

import os
from dotenv import load_dotenv
load_dotenv()

import os
from pymilvus import MilvusClient
from typing import List
from typing import Optional
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonxdata.watsonx_data_v2 import WatsonxDataV2

# -----------------------------
# Env / configuration
# -----------------------------
API_KEY = os.getenv("API_KEY") or os.getenv("WXDATA_API_KEY")
WXDATA_URL = os.getenv("WXDATA_URL")  # MUST be the REST API host base URL (e.g. https://api.eu-gb.watsonxdata.cloud.ibm.com)

# For Milvus vector DB connection (PyMilvus):
# MILVUS_GRPC_HOST = os.getenv("MILVUS_GRPC_HOST")
# MILVUS_GRPC_PORT = os.getenv("MILVUS_GRPC_PORT")
# MILVUS_GRPC_URI = os.getenv("MILVUS_GRPC_URI")  # optional convenience like "https://host:port" or "host:port"
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")  # if required by your Milvus deployment
MILVUS_TLS = os.getenv("MILVUS_TLS", "true").lower() in ("1", "true", "yes", "y")
MILVUS_SERVER_PEM_PATH = os.getenv("MILVUS_SERVER_PEM_PATH")  # if you need to provide server cert


# -----------------------------
# Milvus (PyMilvus) helpers
# -----------------------------
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


def get_milvus_client(
    *,
    grpc_uri: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[str] = None,
    token: Optional[str] = None,
    tls: Optional[bool] = None,
    server_pem_path: Optional[str] = None,
) -> MilvusClient:
    """
    Create a PyMilvus MilvusClient using your gRPC endpoint.

    You should pass the watsonx.data "gRPC address" endpoint here (NOT REST API host).
    """
    grpc_uri = grpc_uri or MILVUS_GRPC_URI
    host = host or MILVUS_GRPC_HOST
    port = port or MILVUS_GRPC_PORT
    token = token or MILVUS_TOKEN
    tls = MILVUS_TLS if tls is None else tls
    server_pem_path = server_pem_path or MILVUS_SERVER_PEM_PATH

    # Normalize uri
    uri = None
    if grpc_uri:
        uri = grpc_uri.strip()
        # Allow "host:port"
        if "://" not in uri:
            scheme = "https" if tls else "http"
            uri = f"{scheme}://{uri}"
    else:
        if not host or not port:
            raise ValueError("Missing Milvus gRPC connection info. Provide grpc_uri or (host, port).")
        scheme = "https" if tls else "http"
        uri = f"{scheme}://{host.strip()}:{str(port).strip()}"

    kwargs: Dict[str, Any] = {"uri": uri}
    if token:
        kwargs["token"] = token

    # PyMilvus TLS parameters vary a bit by version; keep it simple:
    # - If your environment requires a custom server cert, pass it if supported by your version.
    # Some versions accept "tls" / "secure" flags; others infer from https:// uri.
    # We'll pass "secure" defensively.
    kwargs["secure"] = bool(tls)

    # Some builds accept "server_pem_path". If yours doesn't, it will raise TypeError;
    # in that case, remove it and rely on system trust store or set REQUESTS_CA_BUNDLE, etc.
    if server_pem_path:
        kwargs["server_pem_path"] = server_pem_path

    try:
        return MilvusClient(**kwargs)  # type: ignore
    except TypeError:
        # Fallback if your PyMilvus doesn't accept server_pem_path / secure
        kwargs.pop("server_pem_path", None)
        kwargs.pop("secure", None)
        return MilvusClient(**kwargs)  # type: ignore


# -----------------------------
# watsonx.data (management API) helpers
# -----------------------------
def _require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise ValueError(f"Missing {name}. Set it as an environment variable or pass explicitly.")
    return value


def get_iam_bearer_token(api_key: Optional[str] = None) -> str:
    """
    Get IAM Bearer token using IBM IAMAuthenticator.
    """
    api_key = api_key or API_KEY
    api_key = _require_env(api_key, "API_KEY/WXDATA_API_KEY")

    auth = IAMAuthenticator(api_key)
    return auth.token_manager.get_token()

def create_milvus_service_pause(
        self,
        service_id: str,
        *,
        auth_instance_id: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse

# def pause_milvus_service(
#     service_id: str,
#     *,
#     auth_instance_id: Optional[str] = None,
#     api_key: Optional[str] = None,
#     base_url: Optional[str] = None,
#     timeout_s: int = 60,
# ) -> Dict[str, Any]:

    """
    Strategy:
    - Try /engines/{id}/pause first
    - If 404, fallback to /presto_engines/{id}/pause (common for Milvus service control)
    """
    base_url = base_url or WXDATA_URL
    base_url = _require_env(base_url, "WXDATA_URL (watsonx.data REST API host base URL)")

    token = get_iam_bearer_token(api_key)
    headers = {"Authorization": f"Bearer {token}"}
    params = {"auth_instance_id": auth_instance_id} if auth_instance_id else None

    # 1) Engine pause
    url1 = f"{base_url.rstrip('/')}/engines/{service_id}/pause"
    r1 = requests.post(url1, headers=headers, params=params, timeout=timeout_s)
    if r1.status_code != 404:
        r1.raise_for_status()
        return {
            "action": "pause",
            "route": "engines",
            "url": url1,
            "status_code": r1.status_code,
            "body": r1.text,
        }

    # 2) Fallback: presto_engines pause
    url2 = f"{base_url.rstrip('/')}/presto_engines/{service_id}/pause"
    r2 = requests.post(url2, headers=headers, params=params, timeout=timeout_s)
    r2.raise_for_status()
    return {
        "action": "pause",
        "route": "presto_engines",
        "url": url2,
        "status_code": r2.status_code,
        "body": r2.text,
    }


def resume_milvus_service(
    service_id: str,
    *,
    auth_instance_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """
    Resume Milvus service in watsonx.data.

    Strategy mirrors pause:
    - Try /engines/{id}/resume first
    - If 404, fallback to /presto_engines/{id}/resume
    """
    base_url = base_url or WXDATA_URL
    base_url = _require_env(base_url, "WXDATA_URL (watsonx.data REST API host base URL)")

    token = get_iam_bearer_token(api_key)
    headers = {"Authorization": f"Bearer {token}"}
    params = {"auth_instance_id": auth_instance_id} if auth_instance_id else None

    url1 = f"{base_url.rstrip('/')}/engines/{service_id}/resume"
    r1 = requests.post(url1, headers=headers, params=params, timeout=timeout_s)
    if r1.status_code != 404:
        r1.raise_for_status()
        return {
            "action": "resume",
            "route": "engines",
            "url": url1,
            "status_code": r1.status_code,
            "body": r1.text,
        }

    url2 = f"{base_url.rstrip('/')}/presto_engines/{service_id}/resume"
    r2 = requests.post(url2, headers=headers, params=params, timeout=timeout_s)
    r2.raise_for_status()
    return {
        "action": "resume",
        "route": "presto_engines",
        "url": url2,
        "status_code": r2.status_code,
        "body": r2.text,
    }


# -----------------------------
# Optional: keep SDK client getter (if you still want it)
# -----------------------------
def get_wxd_client(api_key: Optional[str] = None, url: Optional[str] = None):
    """
    Return an ibm_watsonxdata SDK client configured with REST API host base URL.

    NOTE:
    This client is good for APIs actually implemented in the SDK.
    For Milvus pause/resume, direct REST calls above are more reliable.
    """
    # api_key = api_key
    # url = url

    # api_key = _require_env(api_key, "API_KEY/WXDATA_API_KEY")
    # url = _require_env(url, "WXDATA_URL (watsonx.data REST API host base URL)")

    authenticator = IAMAuthenticator(api_key)

    client = WatsonxDataV2(authenticator=authenticator)
    client.set_service_url(url)
    return client
