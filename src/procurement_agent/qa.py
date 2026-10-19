import json
import re

from databricks_langchain import ChatDatabricks

from procurement_agent.config import MODEL_NAME
from procurement_agent.retrieval import (
    detect_route,
    retrieve_documents,
    retrieve_structured,
)

SYSTEM_PROMPT = """
You are a procurement intelligence assistant for World Bank transport
and infrastructure projects. Use only supplied evidence. Never invent
project details, dates, statistics, quotations, or sources. Be transparent
when evidence is insufficient and distinguish source statements from
interpretation. Cite document claims with the exact [D#] IDs and structured
claims with the exact [R#] IDs provided. Do not invent citation IDs.
Review flags identify records needing validation; they are not evidence of
fraud, misconduct, or project failure.

For structured answers, preserve IDs, names, statuses, counts, and flag
values. Do not treat missing records as proof an activity never occurred.

For hybrid answers, answer only the document-based part and cite [D#].
Do not list project IDs or produce or interpret a structured review table.
The application appends authoritative review findings separately.
Do not output the retrieved-source registry; the application adds it.
Keep the answer focused and concise.
"""


def _flag_labels(record: dict) -> list[str]:
    flags = []
    if record.get("linkage_review_flag"):
        flags.append("Linkage review required")
    if record.get("deadline_review_flag"):
        flags.append("Deadline/publication-date review required")
    if record.get("no_contracts_flag"):
        flags.append("No contract records found")
    if record.get("no_notices_flag"):
        flags.append("No procurement notice records found")
    return flags


def _format_review_summary(structured: dict) -> str:
    records = structured.get("risk_projects", [])
    rows = []
    for record in records:
        flags = _flag_labels(record)
        if flags:
            project_id = str(record.get("project_id") or "Unknown").replace("|", "\\|")
            project_name = str(record.get("project_name") or "Unknown").replace("|", "\\|")
            status = str(record.get("status") or "Unknown").replace("|", "\\|")
            flag_text = "; ".join(flags).replace("|", "\\|")
            rows.append(f"| {project_id} | {project_name} | {status} | {flag_text} |")
    if not rows:
        return "No projects with review flags were returned."
    return "\n".join([
        "## Projects requiring procurement data-quality review",
        "",
        "| Project ID | Project | Status | Review flags |",
        "|---|---|---|---|",
        *rows,
    ])


def _clean_url(url):
    if not url:
        return None
    url = str(url).strip()
    match = re.fullmatch(r"\[.*?\]\((https?://[^)]+)\)", url)
    if match:
        url = match.group(1)
    return url if url.startswith(("https://", "http://")) else None


def answer_procurement_question(question: str) -> dict:
    route = detect_route(question)
    documents = retrieve_documents(question) if route in ("documents", "hybrid") else []
    structured = retrieve_structured(question, limit=10) if route in ("structured", "hybrid") else {}

    document_registry = []
    document_context = []
    for i, doc in enumerate(documents, start=1):
        ref = f"D{i}"
        doc["citation_id"] = ref
        title = doc.get("document_title") or "Untitled document"
        project_id = doc.get("project_id") or "Unknown"
        page = doc.get("page_number")
        passage = doc.get("chunk_text") or ""
        document_context.append(
            f"[{ref}]\nDocument: {title}\nProject ID: {project_id}\n"
            f"Page: {page}\nPassage:\n{passage}"
        )
        document_registry.append({
            "citation_id": ref,
            "document_title": title,
            "project_id": project_id,
            "page_number": page,
            "source_url": _clean_url(doc.get("source_url")),
        })

    structured_registry = []
    structured_context = []
    ref_number = 0
    for group_name, records in structured.items():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            ref_number += 1
            ref = f"R{ref_number}"
            with_ref = dict(record)
            with_ref["citation_id"] = ref
            structured_registry.append({
                "citation_id": ref,
                "group": group_name,
                "project_id": record.get("project_id") or "Unknown",
                "project_name": record.get("project_name") or "Unknown",
            })
            structured_context.append(
                f"[{ref}]\nResult group: {group_name}\nRecord:\n"
                + json.dumps(with_ref, default=str, ensure_ascii=False)
            )

    if route == "structured":
        evidence = "\n\n".join(structured_context)
        task = "Answer using the supplied structured records. Cite structured claims with [R#]."
    elif route == "hybrid":
        evidence = "\n\n".join(document_context)
        task = (
            "Answer only the document-based part, citing [D#]. Do not list projects "
            "or interpret structured flags. The app appends authoritative findings separately."
        )
    else:
        evidence = "\n\n".join(document_context)
        task = "Answer using document passages and cite factual claims with [D#]."

    if not evidence.strip():
        evidence = "No evidence was retrieved for this route."

    response = ChatDatabricks(endpoint=MODEL_NAME, temperature=0.1).invoke([
        ("system", SYSTEM_PROMPT),
        ("user", f"Question:\n{question}\n\nRoute: {route}\n\nTask:\n{task}\n\nEvidence:\n{evidence}"),
    ])
    content = response.content

    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                # Include only user-facing text blocks; do not expose reasoning metadata.
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        answer = "\n".join(text_parts).strip()
    else:
        answer = str(content).strip()

    if route in ("structured", "hybrid") and structured:
        answer = answer.rstrip() + "\n\n" + _format_review_summary(structured)

    source_lines = []
    for source in document_registry:
        line = (
            f"[{source['citation_id']}] {source['document_title']} "
            f"(Project: {source['project_id']}, page: {source['page_number']})"
        )
        if source["source_url"]:
            line += f" URL: {source['source_url']}"
        source_lines.append(line)
    for source in structured_registry:
        source_lines.append(
            f"[{source['citation_id']}] Gold record — {source['project_name']} "
            f"(Project: {source['project_id']})"
        )
    if source_lines:
        answer = answer.rstrip() + "\n\n## Retrieved sources\n" + "\n".join(source_lines)

    return {
        "question": question,
        "route": route,
        "answer": answer,
        "structured_results": structured,
        "sources": documents,
        "structured_sources": structured_registry,
    }
