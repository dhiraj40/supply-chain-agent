import asyncio
import streamlit as st

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from supply_chain_agent.agent import (
    create_supply_chain_agent,
    ask_agent,
)

TOOL_LABELS = {
    "get_material": "Material Data",
    "get_supplier": "Supplier Data",
    "get_purchase_order": "Purchase Order Data",
    "get_shipment": "Shipment Data",
    "get_inventory": "Inventory Data",
}

st.set_page_config(
    page_title="Supply Chain Agent",
    page_icon="📦",
    layout="centered",
)

st.title("Supply Chain Agent")
st.caption(
    "Ask questions about materials, suppliers, "
    "purchase orders, shipments, inventory, "
    "analytics, policies, and SOPs."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = asyncio.run(
            create_supply_chain_agent()
        )


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

st.markdown("#### Try an example")

sample_questions = [
    "For material M001 at plant P001, show the latest inventory across all storage locations and summarize the total available, blocked, and in-transit quantities.",

    "For purchase order PO001 line 1, show the purchase order details and tell me whether there is any related shipment information available.",

    "Which suppliers currently have the highest number of open purchase orders, and what are the top 5 suppliers?",

    "If shipment SH001 arrives with damaged material, what shipment information do we have and what procedure should the warehouse follow according to the receiving SOP?",

    "Which suppliers have the most open purchase orders, and according to the supplier policy, under what conditions should a supplier be considered at risk?",
]


selected_question = None
col1, col2 = st.columns(2)

for i, sample in enumerate(sample_questions):
    column = col1 if i % 2 == 0 else col2
    with column:
        if st.button(
            sample,
            key=f"sample_question_{i}",
            use_container_width=True,
        ):
            selected_question = sample


typed_question = st.chat_input(
    "Ask a supply chain question..."
)

question = (
    selected_question
    if selected_question
    else typed_question
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):
        with st.status(
            "Working on your request...",
            expanded=False,
        ) as status:
            try:
                response = asyncio.run(
                    ask_agent(
                        st.session_state.agent,
                        question,
                    )
                )

                answer = response["answer"]
                tools_used = response["tools"]
                st.markdown(answer)
                if tools_used:
                    source_labels = []
                    for tool in tools_used:
                        if tool in TOOL_LABELS:
                            label = TOOL_LABELS[tool]
                        elif "genie" in tool.lower():
                            label = "Supply Chain Analytics"
                        elif (
                            "search" in tool.lower()
                            or "vector" in tool.lower()
                        ):
                            label = "Policies & SOPs"
                        else:
                            label = tool
                        if label not in source_labels:
                            source_labels.append(label)
                    st.caption("Source: " + ", ".join(tools_used)   )

                status.update(
                    label="Completed",
                    state="complete",
                )
            except Exception as e:
                status.update(
                    label="Request failed",
                    state="error",
                )
                st.exception(e)

                answer = (
                    "I couldn't complete the request. "
                    "Please try again."
                )
        st.markdown(answer)
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )