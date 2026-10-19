import asyncio
import sys
from pathlib import Path

import streamlit as st

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from supply_chain_agent.agent import create_supply_chain_agent, ask_agent
from procurement_agent.qa import answer_procurement_question

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
                    answer = result["answer"]
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

        st.markdown(answer)

    st.session_state[history_key].append(
        {"role": "assistant", "content": answer}
    )
