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
interpretation. Cite document claims with the exact [D#] IDs provided.
Genie responses are source evidence, but do not invent row-level citation IDs.
Review flags identify records needing validation; they are not evidence of
fraud, misconduct, or project failure.

For structured answers, preserve project IDs, names, statuses, counts, and flag
values stated by Genie. Do not treat missing records as proof an activity never occurred.
For hybrid answers, combine Genie’s structured analysis with relevant document
passages. Map lessons to risk indicators or patterns actually stated by Genie.
Cite documentary claims with [D#]. Distinguish structured findings from
document recommendations and do not claim causation without evidence.
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


def _requested_display_type(question: str) -> str:
    """Infer a useful display from explicit instructions and common analytical intent."""
    q = question.lower()

    # Explicit user requests take precedence.
    if any(term in q for term in ("timeline", "chronological sequence", "event history")):
        return "timeline"
    if any(term in q for term in ("bar chart", "bar graph")):
        return "bar_chart"
    if any(term in q for term in ("line chart", "line graph", "trend chart", "trend graph")):
        return "line_chart"
    if any(term in q for term in (
        "tabular", "as a table", "in a table", "show a table", "table format",
        "in table form", "display as table",
    )):
        return "table"

    # Infer a chart for comparisons of numeric measures across entities.
    comparison_terms = ("compare", "comparison", "compare across", "highest", "lowest", "top ")
    numeric_terms = ("contract value", "contract amount", "award value", "amount", "count", "number of")
    if any(term in q for term in comparison_terms) and any(term in q for term in numeric_terms):
        return "bar_chart"

    # Project/record listings and risk-flag summaries are most readable as tables.
    if any(term in q for term in (
        "risk indicator", "review flag", "review flags", "which projects",
        "list all", "list the", "show projects", "projects with",
        "missing contracts", "missing notices", "top 10", "top 20",
    )):
        return "table"

    if any(term in q for term in ("over time", "trend", "monthly", "yearly", "by date")):
        return "line_chart"

    return "narrative"


def _build_display_data(question: str, requested_type: str, answer: str, document_context: list[str]) -> dict:
    """Extract displayable records from retrieved evidence; never create new facts."""
    if requested_type == "narrative":
        return {"display_type": "narrative", "title": "", "data": []}

    schema_guidance = {
        "timeline": (
            'Return JSON with keys display_type, title, data. display_type must be "timeline". '
            'Each data item must contain date, event, and may contain details and source. '
            'Only include events and dates explicitly supported by the evidence.'
        ),
        "table": (
            'Return JSON with keys display_type, title, data. display_type must be "table". '
            'data must be an array of row objects containing the fields and values requested. '
            'Preserve IDs, dates, counts, and values exactly as stated in the evidence.'
        ),
        "bar_chart": (
            'Return JSON with keys display_type, title, x_key, y_key, data. display_type must be "bar_chart". '
            'Use only a numeric measure explicitly present in evidence; data must be row objects.'
        ),
        "line_chart": (
            'Return JSON with keys display_type, title, x_key, y_key, data. display_type must be "line_chart". '
            'Use a real date/time series and numeric measure explicitly present in evidence; data must be row objects.'
        ),
    }
    evidence = (
        "ANSWER FROM RETRIEVAL:\n" + (answer or "No answer returned.")
        + "\n\nDOCUMENT EVIDENCE:\n"
        + ("\n\n".join(document_context) if document_context else "No document passages.")
    )[:14000]
    prompt = (
        "Convert the supplied evidence into display data for a procurement assistant. "
        "Return JSON only, without Markdown fences. Never infer missing events, dates, counts, "
        "or values. If evidence does not contain enough data for the requested visualization, "
        'return {"display_type":"narrative","title":"","data":[]} instead. '
        + schema_guidance[requested_type]
        + "\n\nUser question:\n" + question
        + "\n\nEvidence:\n" + evidence
    )
    try:
        response = ChatDatabricks(endpoint=MODEL_NAME, temperature=0.0).invoke([
            ("system", "You are a strict evidence-to-JSON formatter. Do not answer from general knowledge."),
            ("user", prompt),
        ])
        raw = response.content
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^\`\`\`(?:json)?\s*|\s*\`\`\`$", "", raw, flags=re.IGNORECASE)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Display payload must be a JSON object.")
        display_type = parsed.get("display_type")
        if display_type not in {"timeline", "table", "bar_chart", "line_chart", "narrative"}:
            raise ValueError("Unsupported display type.")
        data = parsed.get("data", [])
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("Display data must be a list of row objects.")
        if display_type == "timeline" and data:
            if any(not row.get("date") or not row.get("event") for row in data):
                raise ValueError("Timeline rows require date and event.")
        if display_type in {"bar_chart", "line_chart"} and data:
            x_key, y_key = parsed.get("x_key"), parsed.get("y_key")
            if not x_key or not y_key or any(x_key not in row or y_key not in row for row in data):
                raise ValueError("Chart keys must exist in each row.")
        if not data:
            return {"display_type": "narrative", "title": "", "data": []}
        return {
            "display_type": display_type,
            "title": str(parsed.get("title") or ""),
            "x_key": parsed.get("x_key"),
            "y_key": parsed.get("y_key"),
            "data": data,
        }
    except Exception:
        # A formatting failure must not prevent the evidence-based answer from displaying.
        return {"display_type": "narrative", "title": "", "data": []}


def answer_procurement_question(question: str) -> dict:
    route = detect_route(question)
    requested_type = _requested_display_type(question)
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
    for key, value in structured.items():
        if isinstance(value, str) and value.strip():
            structured_context.append(
                f"Genie response for {key} (structured analytics evidence):\\n{value.strip()}"
            )

    if route == "structured":
        evidence = "\\n\\n".join(structured_context)
        task = (
            "Answer using the supplied Genie response. Preserve its project IDs, "
            "metrics, and caveats. Do not claim direct SQL result rows were parsed."
        )
    elif route == "hybrid":
        evidence = (
            "STRUCTURED ANALYTICS FROM PROCUREMENT GENIE:\\n"
            + ("\\n\\n".join(structured_context) or "No Genie answer was returned.")
            + "\\n\\nDOCUMENT RETRIEVAL EVIDENCE:\\n"
            + ("\\n\\n".join(document_context) or "No document passages were retrieved.")
        )
        task = (
            "Combine Genie’s structured analysis with retrieved document passages. "
            "Map lessons to specific risk indicators or patterns stated by Genie. "
            "Cite documentary claims with [D#]. If either source is missing, say so."
        )
    else:
        evidence = "\\n\\n".join(document_context)
        task = "Answer using document passages and cite factual claims with [D#]."

    if not evidence.strip():
        evidence = "No evidence was retrieved for this route."

    response = ChatDatabricks(endpoint=MODEL_NAME, temperature=0.1).invoke([
        ("system", SYSTEM_PROMPT),
        ("user", f"Question:\n{question}\n\nRoute: {route}\n\nTask:\n{task}\n\nEvidence:\n{evidence}"),
    ])
    content = response.content

    def extract_answer(value):
        # Some endpoints return content blocks as a JSON-encoded string
        # instead of a Python list. Parse that form before rendering it.
        if isinstance(value, str):
            stripped = value.strip()
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped

            if isinstance(parsed, list):
                return extract_answer(parsed)
            if isinstance(parsed, dict):
                for key in ("answer", "final_answer", "output", "text"):
                    candidate = parsed.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                return stripped
            return stripped

        if isinstance(value, list):
            text_parts = []
            for block in value:
                if isinstance(block, dict):
                    # Ignore reasoning/metadata; display only user-facing text.
                    if block.get("type") == "text":
                        text_value = block.get("text")
                        if isinstance(text_value, str) and text_value.strip():
                            text_parts.append(text_value.strip())
                elif isinstance(block, str) and block.strip():
                    text_parts.append(block.strip())
            return "\n\n".join(text_parts).strip()

        if isinstance(value, dict):
            for key in ("answer", "final_answer", "output", "text"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            return str(value)

        return str(value).strip()

    answer = extract_answer(content)

    # For structured-only questions, preserve Genie’s own answer rather than
    # asking a second model to reconstruct results from nonexistent SQL rows.
    if route == "structured":
        genie_answer = next(
            (value.strip() for value in structured.values()
             if isinstance(value, str) and value.strip()),
            "",
        )
        if genie_answer:
            answer = genie_answer

    display_data = _build_display_data(question, requested_type, answer, document_context)

    review_summary = ""

    source_registry = []
    genie_text = next(
        (value.strip() for value in structured.values()
         if isinstance(value, str) and value.strip()),
        "",
    )
    if genie_text:
        source_registry.append({
            "citation_id": "Genie",
            "source_type": "Databricks Genie",
            "title": "Procurement structured analytics",
            "project_id": "Multiple or not specified",
            "detail": "Natural-language answer from the Procurement Genie Agent",
            "source_url": None,
        })
    for source in document_registry:
        source_registry.append({
            "citation_id": source["citation_id"],
            "source_type": "Document",
            "title": source["document_title"],
            "project_id": source["project_id"],
            "detail": f"Page: {source['page_number']}",
            "source_url": source["source_url"],
        })
    for source in structured_registry:
        source_registry.append({
            "citation_id": source["citation_id"],
            "source_type": "Gold record",
            "title": source["project_name"],
            "project_id": source["project_id"],
            "detail": source["group"],
            "source_url": None,
        })

    return {
        "question": question,
        "route": route,
        "answer": answer,
        "display_data": display_data,
        "review_summary": review_summary,
        "source_registry": source_registry,
        "structured_results": structured,
        "sources": documents,
        "structured_sources": structured_registry,
    }
