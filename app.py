import asyncio
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from supply_chain_agent.agent import create_supply_chain_agent, ask_agent
from procurement_agent.qa import answer_procurement_question


def render_procurement_response(result: dict) -> None:
    """Render answer text, structured visual output, and source metadata."""
    st.markdown(result.get("answer") or "No answer text was returned.")

    display = result.get("display_data") or {}
    display_type = display.get("display_type", "narrative")
    rows = display.get("data") or []

    if display_type == "timeline" and rows:
        df = pd.DataFrame(rows)
        if {"date", "event"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if not df.empty:
                st.markdown(f"#### {display.get('title') or 'Project timeline'}")
                hover_cols = [c for c in ("details", "source") if c in df.columns]
                plot_df = df.copy()
                plot_df["position"] = 1
                fig = px.scatter(
                    plot_df,
                    x="date",
                    y="position",
                    text="event",
                    hover_data=hover_cols,
                )
                fig.update_traces(mode="markers+text", textposition="top center")
                fig.update_yaxes(visible=False, title=None)
                fig.update_layout(xaxis_title="Date", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Timeline events were returned, but no valid dates were available.")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Timeline data did not contain the required date and event fields.")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif display_type == "table" and rows:
        st.markdown(f"#### {display.get('title') or 'Results'}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif display_type in ("bar_chart", "line_chart") and rows:
        df = pd.DataFrame(rows)
        x_key = display.get("x_key")
        y_key = display.get("y_key")
        if x_key in df.columns and y_key in df.columns:
            st.markdown(f"#### {display.get('title') or 'Results'}")
            chart_fn = px.bar if display_type == "bar_chart" else px.line
            fig = chart_fn(df, x=x_key, y=y_key)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    review_summary = result.get("review_summary")
    if review_summary:
        st.markdown(review_summary)

    sources = result.get("source_registry") or []
    if sources:
        with st.expander(f"Retrieved sources ({len(sources)})"):
            for source in sources:
                citation = source.get("citation_id", "Source")
                title = source.get("title") or "Untitled source"
                project_id = source.get("project_id") or "Unknown"
                detail = source.get("detail") or source.get("source_type") or ""
                url = source.get("source_url")
                st.markdown(f"**[{citation}] {title}**")
                st.caption(f"{source.get('source_type', 'Source')} · Project: {project_id} · {detail}")
                if url:
                    st.markdown(f"[Open source document]({url})")


TOOL_LABELS = {
    "get_material": "Material Data",
    "get_supplier": "Supplier Data",
    "get_purchase_order": "Purchase Order Data",
    "get_shipment": "Shipment Data",
    "get_inventory": "Inventory Data",
}

SUPPLY_CHAIN_QUESTIONS = [
    "For material M001 at plant P001, show the latest inventory across all storage locations and summarize the total available, blocked, and in-transit quantities.",
    "For purchase order PO001 line 1, show the purchase order details and tell me whether there is any related shipment information available.",
    "Which suppliers currently have the highest number of open purchase orders, and what are the top 5 suppliers?",
    "If shipment SH001 arrives with damaged material, what shipment information do we have and what procedure should the warehouse follow according to the receiving SOP?",
    "Which suppliers have the most open purchase orders, and according to the supplier policy, under what conditions should a supplier be considered at risk?",
]

PROCUREMENT_QUESTIONS = [
    "What procurement challenges occurred in Angola projects, and what lessons can future projects learn?",
    "Which projects need procurement data-quality review in Angola?",
    "What lessons learned describe delays in contract awards?",
    "Show the timeline for project P000034.",
    "What procurement challenges occurred in Angola, and which projects have review flags?",
]

st.set_page_config(
    page_title="Supply Chain & Procurement Agents",
    page_icon="🤖",
    layout="wide",
)

with st.sidebar:
    st.title("AI Agents")
    selected_agent = st.selectbox(
        "Choose an agent",
        ["Supply Chain Agent", "Procurement Agent"],
        key="selected_agent",
    )
    history_key = (
        "supply_chain_messages"
        if selected_agent == "Supply Chain Agent"
        else "procurement_messages"
    )
    st.caption(
        "Supply Chain: materials, suppliers, POs, shipments, inventory, policies and SOPs."
        if selected_agent == "Supply Chain Agent"
        else "Procurement: World Bank projects, timelines, risk indicators, lessons learned and source documents."
    )
    if st.button("Clear this chat", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun()

if "supply_chain_messages" not in st.session_state:
    st.session_state.supply_chain_messages = []
if "procurement_messages" not in st.session_state:
    st.session_state.procurement_messages = []

messages = st.session_state[history_key]

if selected_agent == "Supply Chain Agent":
    st.title("📦 Supply Chain Agent")
    st.caption(
        "Ask questions about materials, suppliers, purchase orders, shipments, "
        "inventory, analytics, policies, and SOPs."
    )
    sample_questions = SUPPLY_CHAIN_QUESTIONS
    chat_placeholder = "Ask a supply chain question..."
else:
    st.title("🌍 World Bank Procurement Agent")
    st.caption(
        "Explore transport procurement projects, timelines, data-quality flags, "
        "lessons learned, and evidence from project documents."
    )
    sample_questions = PROCUREMENT_QUESTIONS
    chat_placeholder = "Ask a procurement question..."

for message in messages:
    with st.chat_message(message["role"]):
        if message.get("procurement_result"):
            render_procurement_response(message["procurement_result"])
        else:
            st.markdown(message["content"])

st.markdown("#### Try an example")
selected_question = None
col1, col2 = st.columns(2)
for i, sample in enumerate(sample_questions):
    column = col1 if i % 2 == 0 else col2
    with column:
        if st.button(
            sample,
            key=f"{history_key}_sample_{i}",
            use_container_width=True,
        ):
            selected_question = sample

typed_question = st.chat_input(
    chat_placeholder,
    key=f"{history_key}_input",
)
question = selected_question or typed_question

if question:
    st.session_state[history_key].append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        procurement_result = None
        with st.status("Working on your request...", expanded=False) as status:
            try:
                if selected_agent == "Supply Chain Agent":
                    if "supply_chain_agent_instance" not in st.session_state:
                        st.session_state.supply_chain_agent_instance = asyncio.run(
                            create_supply_chain_agent()
                        )
                    response = asyncio.run(
                        ask_agent(
                            st.session_state.supply_chain_agent_instance,
                            question,
                        )
                    )
                    answer = response["answer"]
                    tools_used = response.get("tools", [])
                    if tools_used:
                        source_labels = []
                        for tool in tools_used:
                            if tool in TOOL_LABELS:
                                label = TOOL_LABELS[tool]
                            elif "genie" in tool.lower():
                                label = "Supply Chain Analytics"
                            elif "search" in tool.lower() or "vector" in tool.lower():
                                label = "Policies & SOPs"
                            else:
                                label = tool
                            if label not in source_labels:
                                source_labels.append(label)
                        st.caption("Tools used: " + ", ".join(source_labels))
                else:
                    result = answer_procurement_question(question)
                    procurement_result = {
                        "answer": result.get("answer", ""),
                        "route": result.get("route", ""),
                        "review_summary": result.get("review_summary", ""),
                        "source_registry": result.get("source_registry", []),
                    }
                    answer = result.get("answer", "")
                    st.caption(f"Retrieval route: {result['route']}")

                status.update(label="Completed", state="complete")
            except Exception as exc:
                status.update(label="Request failed", state="error")
                st.error(
                    "I couldn't complete this request. Check the app logs and "
                    "confirm the selected agent's Databricks permissions and configuration."
                )
                st.exception(exc)
                answer = (
                    "The request failed because a data source or model could not be reached. "
                    "Please check the application configuration and try again."
                )

        if procurement_result:
            render_procurement_response(procurement_result)
        else:
            st.markdown(answer)

    assistant_message = {"role": "assistant", "content": answer}
    if procurement_result:
        assistant_message["procurement_result"] = procurement_result
    st.session_state[history_key].append(assistant_message)
