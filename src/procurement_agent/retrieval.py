import re
import time
from typing import Any

import requests
from databricks.sdk import WorkspaceClient

from procurement_agent.config import (
    PROJECT_TABLE,
    TIMELINE_TABLE,
    RISK_TABLE,
    VECTOR_INDEX_NAME,
    PROCUREMENT_GENIE_SPACE_ID,
)

DOCUMENT_COLUMNS = [
    "chunk_id", "document_id", "project_id", "document_title",
    "document_type", "page_number", "chunk_text", "source_url",
]

COUNTRIES = [
    "Angola", "Benin", "India", "Nepal", "Bangladesh", "Pakistan",
    "Kenya", "Tanzania", "Ethiopia", "Ghana", "Nigeria", "Uganda",
    "Rwanda", "Mozambique", "Zambia", "Malawi", "Cameroon", "Senegal",
    "Liberia", "Sierra Leone",
]

GENIE_POLL_INTERVAL_SECONDS = 1.5
GENIE_TIMEOUT_SECONDS = 90
GENIE_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}


def _genie_request(method: str, path: str, *, json_body: dict | None = None) -> dict:
    """Call the Databricks Genie Conversation API using the app identity."""
    if not PROCUREMENT_GENIE_SPACE_ID:
        raise RuntimeError(
            "Procurement Genie is not configured. Set PROCUREMENT_GENIE_SPACE_ID "
            "to the ID of the World Bank Procurement SQL Agent."
        )

    workspace = WorkspaceClient()
    host = (workspace.config.host or "").rstrip("/")
    if not host:
        raise RuntimeError("Databricks workspace host is not configured.")

    auth_headers = workspace.config.authenticate()
    if not isinstance(auth_headers, dict) or not auth_headers.get("Authorization"):
        raise RuntimeError("Unable to obtain Databricks app authentication headers.")

    response = requests.request(
        method=method,
        url=f"{host}{path}",
        headers={**auth_headers, "Content-Type": "application/json"},
        json=json_body,
        timeout=30,
    )
    if not response.ok:
        # Preserve the API's diagnostic message instead of exposing only
        # requests' generic "400 Client Error" to the Streamlit user.
        detail = response.text[:2000]
        raise RuntimeError(
            f"Databricks Genie API {method} {path} failed "
            f"with HTTP {response.status_code}: {detail}"
        )
    return response.json() if response.content else {}


def _as_dict(value: Any) -> dict:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    return value if isinstance(value, dict) else {}


def query_procurement_genie(question: str) -> dict[str, Any]:
    """Ask the Procurement Genie Agent and return its natural-language answer."""
    start_path = (
        f"/api/2.0/genie/spaces/{PROCUREMENT_GENIE_SPACE_ID}/start-conversation"
    )
    started = _genie_request("POST", start_path, json_body={"content": question})
    message = _as_dict(started.get("message"))
    conversation = _as_dict(started.get("conversation"))
    conversation_id = message.get("conversation_id") or conversation.get("id") or started.get("conversation_id")
    message_id = message.get("message_id") or message.get("id") or started.get("message_id")
    if not conversation_id or not message_id:
        raise RuntimeError("Genie did not return a conversation ID and message ID.")

    deadline = time.monotonic() + GENIE_TIMEOUT_SECONDS
    status = str(message.get("status") or "SUBMITTED").upper()
    while status not in GENIE_TERMINAL_STATUSES:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Procurement Genie did not finish within {GENIE_TIMEOUT_SECONDS} seconds."
            )
        time.sleep(GENIE_POLL_INTERVAL_SECONDS)
        message_path = (
            f"/api/2.0/genie/spaces/{PROCUREMENT_GENIE_SPACE_ID}"
            f"/conversations/{conversation_id}/messages/{message_id}"
        )
        message = _genie_request("GET", message_path)
        status = str(message.get("status") or "").upper()

    if status != "COMPLETED":
        error = message.get("error")
        raise RuntimeError(f"Procurement Genie returned status {status}: {error or 'No details provided'}")

    answer_parts = []
    for attachment in message.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        text_attachment = attachment.get("text")
        if isinstance(text_attachment, dict) and text_attachment.get("content"):
            answer_parts.append(str(text_attachment["content"]))

    # Structured analytics come from Genie’s natural-language response.
    # Do not call SQL query-result endpoints or parse SQL result attachments.
    return {
        "answer": "\\n\\n".join(answer_parts).strip(),
        "status": status,
    }


def search_projects(search_term: str, limit: int = 10):
    term = search_term.strip()
    if not term:
        raise ValueError("search_term cannot be empty.")
    result = query_procurement_genie(
        f"Find up to {int(limit)} World Bank procurement projects matching '{term}'. "
        f"Use {PROJECT_TABLE}. Return project IDs, names, country, status, contract "
        "counts, total contract amounts, and the latest snapshot date where available. "
        "Do not combine multiple snapshots."
    )
    return result


def get_risk_projects(country: str | None = None, limit: int = 20):
    country_filter = f" in {country}" if country else ""
    result = query_procurement_genie(
        f"List up to {int(limit)} World Bank procurement projects{country_filter} "
        "that require review or have procurement risk/data-quality indicators. "
        f"Use {RISK_TABLE} and, if needed, {PROJECT_TABLE}. Return project ID, "
        "project name, country, review flags, issue counts, and review reasons. "
        "Do not describe a review indicator as confirmed misconduct."
    )
    return result


def get_project_timeline(project_id: str, limit: int = 100):
    pid = project_id.strip().upper()
    if not pid:
        raise ValueError("project_id cannot be empty.")
    result = query_procurement_genie(
        f"Show up to {int(limit)} procurement timeline events for project {pid} "
        f"using {TIMELINE_TABLE}. Return project ID, project name, contract number, "
        "event date, event type, description, source reference, and date quality "
        "if available. Sort chronologically. Do not invent missing events."
    )
    return result


def extract_project_id(question: str):
    match = re.search(r"\bP\d{6}\b", question.upper())
    return match.group(0) if match else None


def extract_country(question: str):
    q = question.lower()
    for country in COUNTRIES:
        if re.search(r"\b" + re.escape(country.lower()) + r"\b", q):
            return country
    return None


def detect_route(question: str) -> str:
    q = question.lower()
    structured_terms = [
        "contract value", "supplier", "number of contracts", "project count",
        "procurement notice", "timeline", "event history", "risk", "flag",
        "anomaly", "data quality", "review", "project status",
    ]
    document_terms = [
        "lesson", "lessons learned", "why did", "implementation challenge",
        "procurement challenge", "recommendation", "what happened",
        "best practice", "how can future", "what can future",
    ]
    has_structured = any(term in q for term in structured_terms)
    has_documents = any(term in q for term in document_terms)
    if has_structured and has_documents:
        return "hybrid"
    if has_documents:
        return "documents"
    if has_structured:
        return "structured"
    return "documents"


def retrieve_structured(question: str, limit: int = 10):
    q = question.lower()
    project_id = extract_project_id(question)
    country = extract_country(question)

    if any(word in q for word in ["timeline", "event history", "sequence of events"]):
        if not project_id:
            return {"message": "Provide a project ID such as P000034 to retrieve its timeline."}
        result = get_project_timeline(project_id, limit)
        return {"timeline_answer": result.get("answer", ""), "genie_status": result.get("status", "")}

    if any(word in q for word in ["risk", "flag", "anomaly", "data quality", "review"]):
        result = get_risk_projects(country, limit)
        return {
            "risk_answer": result.get("answer", ""),
            "genie_status": result.get("status", ""),
        }

    if country:
        result = search_projects(country, limit)
        return {"project_answer": result.get("answer", ""), "genie_status": result.get("status", "")}
    if project_id:
        result = search_projects(project_id, limit)
        return {"project_answer": result.get("answer", ""), "genie_status": result.get("status", "")}

    result = query_procurement_genie(
        f"Answer this procurement analytics question using the configured Procurement "
        f"Genie tables. Prefer {PROJECT_TABLE}, {TIMELINE_TABLE}, and {RISK_TABLE} "
        f"as appropriate. Return a concise, evidence-based natural-language answer "
        f"with relevant values, identifiers, and dates. Question: {question}"
    )
    return {"genie_answer": result.get("answer", ""), "genie_status": result.get("status", "")}


def retrieve_documents(question: str, top_k: int = 5):
    workspace = WorkspaceClient()
    response = workspace.vector_search_indexes.query_index(
        index_name=VECTOR_INDEX_NAME,
        query_text=question,
        columns=DOCUMENT_COLUMNS,
        num_results=top_k,
    )
    if hasattr(response, "as_dict"):
        response = response.as_dict()
    manifest = response.get("manifest", {})
    columns = [item["name"] for item in manifest.get("columns", [])]
    data = response.get("result", {}).get("data_array", [])
    if not columns:
        return []
    return [dict(zip(columns, row)) for row in data]
