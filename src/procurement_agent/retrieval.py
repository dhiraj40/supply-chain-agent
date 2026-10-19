import re
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sql import connect

from procurement_agent.config import (
    PROJECT_TABLE,
    TIMELINE_TABLE,
    RISK_TABLE,
    VECTOR_INDEX_NAME,
    SQL_WAREHOUSE_HTTP_PATH,
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


def _sql_connection():
    if not SQL_WAREHOUSE_HTTP_PATH:
        raise RuntimeError(
            "Procurement structured queries need a Databricks SQL Warehouse. "
            "Create one and set DATABRICKS_SQL_WAREHOUSE_HTTP_PATH in app.yaml."
        )
    workspace = WorkspaceClient()
    host = (workspace.config.host or "").replace("https://", "").rstrip("/")
    # Uses the Databricks Apps service-principal identity via the SDK.
    token = workspace.config.authenticate()
    if not isinstance(token, dict):
        raise RuntimeError("Unable to obtain Databricks app authentication headers.")
    access_token = token.get("Authorization", "").removeprefix("Bearer ")
    if not access_token:
        # Some SDK auth providers expose headers via auth_type rather than
        # authenticate(); let the connector use its configured SDK credentials.
        return connect(
            server_hostname=host,
            http_path=SQL_WAREHOUSE_HTTP_PATH,
            credentials_provider=lambda: workspace.config.authenticate,
        )
    return connect(
        server_hostname=host,
        http_path=SQL_WAREHOUSE_HTTP_PATH,
        access_token=access_token,
    )


def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    connection = _sql_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            names = [item[0] for item in cursor.description or []]
            return [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def search_projects(search_term: str, limit: int = 10):
    term = search_term.strip()
    if not term:
        raise ValueError("search_term cannot be empty.")
    pattern = f"%{term}%"
    return _query(
        f"""
        SELECT project_id, project_name, countryshortname, regionname,
               status, contract_count, total_contract_value_usd,
               notice_count, project_match_status
        FROM {PROJECT_TABLE}
        WHERE lower(coalesce(project_id, '')) LIKE lower(?)
           OR lower(coalesce(project_name, '')) LIKE lower(?)
           OR lower(coalesce(countryshortname, '')) LIKE lower(?)
        LIMIT ?
        """,
        (pattern, pattern, pattern, int(limit)),
    )


def get_risk_projects(country: str | None = None, limit: int = 20):
    sql = f"""
        SELECT project_id, project_name, countryshortname, status,
               contract_count, notice_count, deadline_anomaly_count,
               linkage_review_flag, deadline_review_flag, no_contracts_flag,
               no_notices_flag, risk_review_reason
        FROM {RISK_TABLE}
        WHERE coalesce(linkage_review_flag, false)
           OR coalesce(deadline_review_flag, false)
           OR coalesce(no_contracts_flag, false)
           OR coalesce(no_notices_flag, false)
    """
    params: list[Any] = []
    if country:
        sql += " AND lower(coalesce(countryshortname, '')) LIKE lower(?)"
        params.append(f"%{country.strip()}%")
    sql += """
        ORDER BY deadline_anomaly_count DESC NULLS LAST,
                 linkage_review_flag DESC
        LIMIT ?
    """
    params.append(int(limit))
    return _query(sql, tuple(params))


def get_project_timeline(project_id: str, limit: int = 100):
    pid = project_id.strip().upper()
    if not pid:
        raise ValueError("project_id cannot be empty.")
    return _query(
        f"""
        SELECT project_id, project_name, event_type, event_date,
               event_description, source_record_id, event_date_quality
        FROM {TIMELINE_TABLE}
        WHERE upper(coalesce(project_id, '')) = ?
           OR upper(coalesce(project_id_key, '')) = ?
        ORDER BY event_date ASC NULLS LAST
        LIMIT ?
        """,
        (pid, pid, int(limit)),
    )


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
    results = {}

    if any(word in q for word in ["timeline", "event history", "sequence of events"]):
        if project_id:
            results["timeline"] = get_project_timeline(project_id, limit)
        else:
            results["message"] = "Provide a project ID such as P000034 to retrieve its timeline."
    elif any(word in q for word in ["risk", "flag", "anomaly", "data quality", "review"]):
        results["risk_projects"] = get_risk_projects(country, limit)
    elif country:
        results["projects"] = search_projects(country, limit)
    elif project_id:
        results["projects"] = search_projects(project_id, limit)
    else:
        results["message"] = (
            "Please include a country, project ID, or a topic such as risk indicators or timeline."
        )
    return results


def retrieve_documents(question: str, top_k: int = 5):
    workspace = WorkspaceClient()
    response = workspace.vector_search_indexes.query_index(
        index_name=VECTOR_INDEX_NAME,
        query_text=question,
        columns=DOCUMENT_COLUMNS,
        num_results=top_k,
    )
    # SDK response models support as_dict(); keep compatibility with dict results.
    if hasattr(response, "as_dict"):
        response = response.as_dict()
    manifest = response.get("manifest", {})
    columns = [item["name"] for item in manifest.get("columns", [])]
    data = response.get("result", {}).get("data_array", [])
    if not columns:
        return []
    return [dict(zip(columns, row)) for row in data]
