import json
import warnings

from databricks.sdk import WorkspaceClient

from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from supply_chain_agent.config import (
    MODEL_NAME,
    PRODUCT_CATALOG,
    AGENT_SCHEMA,
    GENIE_SPACE_ID,
    AI_SEARCH_INDEX,
)


warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
)


SYSTEM_PROMPT = """
You are a supply chain assistant.

Use deterministic lookup tools for questions about a
specific material, supplier, purchase order, shipment,
or material/plant inventory combination.

Use the Genie analytics tool for aggregations,
comparisons, rankings, counts, trends, and questions
involving multiple business records.

Use document search for policies, procedures, SOPs,
guidelines, and other document-based questions.

Never invent business data.

Base business answers on information returned by tools.

When answering from documents, mention the source
document when source information is available.
"""


async def create_supply_chain_agent():

    workspace_client = WorkspaceClient()
    host = workspace_client.config.host

    servers = [
        DatabricksMCPServer(
            name="supply-chain-tools",
            url=(
                f"{host}/api/2.0/mcp/functions/"
                f"{PRODUCT_CATALOG}/{AGENT_SCHEMA}"
            ),
            workspace_client=workspace_client,
        ),
        DatabricksMCPServer(
            name="supply-chain-documents",
            url=(
                f"{host}/api/2.0/mcp/ai-search/"
                f"{PRODUCT_CATALOG}/"
                f"{AGENT_SCHEMA}/"
                f"{AI_SEARCH_INDEX}"
            ),
            workspace_client=workspace_client,
        ),
    ]

    if GENIE_SPACE_ID:
        servers.append(
            DatabricksMCPServer(
                name="supply-chain-analytics",
                url=(
                    f"{host}/api/2.0/mcp/genie/"
                    f"{GENIE_SPACE_ID}"
                ),
                workspace_client=workspace_client,
            )
        )

    mcp_client = DatabricksMultiServerMCPClient(
        servers
    )

    tools = await mcp_client.get_tools()
    # print(
    #     "AVAILABLE AGENT TOOLS:",
    #     [tool.name for tool in tools],
    # )

    agent = create_agent(
        model=ChatDatabricks(
            endpoint=MODEL_NAME
        ),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent

def get_text_output(result):

    content = result["messages"][-1].content

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return content

    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if block.get("type") == "text"
        )

    return str(content)

def get_tool_names(result):

    return list(
        dict.fromkeys(
            message.name
            for message in result["messages"]
            if isinstance(message, ToolMessage)
            and message.name
        )
    )

async def ask_agent( agent, question: str):
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    return {
        "answer": get_text_output(result),
        "tools": get_tool_names(result)
    }