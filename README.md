# Supply Chain Agent

An AI-powered supply chain assistant built on Databricks.

The agent combines structured enterprise data, analytical querying, and document retrieval to answer supply chain questions through a conversational interface.

## Architecture

```text
                    Supply Chain Agent
                           |
                     Streamlit UI
                    Databricks Apps
                           |
                    LangChain Agent
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    UC Functions         Genie          AI Search
          |                |                |
   Exact lookups       Analytics       Documents / RAG
          |                |                |
          +----------------+----------------+
                           |
                           v
                Common Data Platform
                           |
                        Trusted
```

## Capabilities

The agent supports three main categories of questions.

### 1. Structured Data Lookups

Unity Catalog functions provide deterministic access to certified business data.

Available tools include:

- Material lookup
- Supplier lookup
- Purchase order lookup
- Shipment lookup
- Inventory lookup

Examples:

```text
Tell me about material M001
```

```text
Show me the latest inventory for material M001 at plant P001
```

### 2. Supply Chain Analytics

Databricks Genie is used for analytical questions involving multiple records, aggregations, comparisons, rankings, and trends.

Example:

```text
Which suppliers currently have the highest number of open purchase orders?
```

### 3. Document Retrieval / RAG

Databricks AI Search provides semantic retrieval over supply chain documents such as:

- Standard Operating Procedures
- Supplier policies
- Shipment handling guides
- Operational manuals

Example:

```text
What should happen when received goods are damaged?
```

The agent retrieves relevant document chunks and uses them to generate a grounded response.

## Example Questions

```text
Tell me about material M001
```

```text
Show me the latest inventory for material M001 at plant P001
```

```text
For purchase order PO001 line 1, show the purchase order details and tell me whether there is any related shipment information available.
```

```text
If shipment SH001 arrives with damaged material, what shipment information do we have and what procedure should the warehouse follow according to the receiving SOP?
```

```text
Which suppliers have the most open purchase orders, and according to the supplier policy, under what conditions should a supplier be considered at risk?
```

## Data Platform

The Supply Chain Agent consumes certified datasets from the Common Data Platform.

```text
Source Systems
      |
      v
     Raw
      |
      v
 Foundation
      |
      v
   Trusted
      |
      v
Supply Chain Agent
```

The agent does not directly own core enterprise data.

Trusted datasets currently include:

- Material
- Plant
- Supplier
- Purchase Order
- Shipment
- Inventory

Product-specific assets such as AI Search indexes, RAG chunks, agent tools, evaluation data, and application logic are maintained separately in the Supply Chain Agent product layer.

## Technology Stack

- Databricks
- Unity Catalog
- Delta Lake
- Databricks SQL
- Databricks Genie
- Databricks AI Search
- Databricks Model Serving
- Databricks Managed MCP
- LangChain
- LangGraph
- MLflow
- Streamlit
- Databricks Apps

## Agent Tool Architecture

The agent dynamically selects between different tools depending on the question.

```text
Specific business entity
        |
        v
   UC Function

Aggregation / ranking / trend
        |
        v
      Genie

Policy / SOP / manual
        |
        v
    AI Search
```

This allows deterministic lookups and analytical or unstructured retrieval to coexist within the same agent.

## Project Structure

```text
Supply Chain Agent/
|
+-- src/
|   +-- supply_chain_agent/
|       +-- __init__.py
|       +-- config.py
|       +-- agent.py
|
+-- notebooks/
|   +-- 01_test_tools.ipynb
|   +-- 02_test_agent.ipynb
|   +-- 03_test_document_rag.ipynb
|   +-- 04_test_evaluation.ipynb
|
+-- app.py
+-- app.yaml
+-- requirements.txt
+-- README.md
```

## Agent Implementation

The main agent implementation lives in:

```text
src/supply_chain_agent/agent.py
```

The agent uses:

```python
ChatDatabricks
DatabricksMCPServer
DatabricksMultiServerMCPClient
LangChain create_agent
```

The current model serving endpoint is configurable through environment variables.

## Configuration

Application configuration is defined in:

```text
src/supply_chain_agent/config.py
```

Typical environment variables include:

```text
ENVIRONMENT
MODEL_NAME
GENIE_SPACE_ID
AI_SEARCH_INDEX
```

Environment-specific catalog names are generated from the selected environment.

Example:

```text
common_data_platform_dev
common_data_products_dev
```

## Python Dependencies

The project uses pinned versions of the core agent libraries to avoid compatibility issues.

```text
langchain==1.3.15
langgraph==1.2.11
langgraph-prebuilt==1.1.0
langchain-mcp-adapters==0.3.2
databricks-langchain==0.20.0
databricks-mcp==0.9.2
mlflow==3.15.1
```

Streamlit is used for the application UI.

## Databricks App

The application is deployed using Databricks Apps.

The app provides:

- Conversational chat interface
- Sample questions
- Chat history
- Clear chat option
- Agent execution status
- Tool/source attribution

The Databricks App runs using its own service principal.

Required resource permissions include access to:

- Model Serving endpoint
- Unity Catalog functions
- Genie Space
- AI Search index

## Running the App

The application entry point is:

```text
app.py
```

The Databricks App runtime is configured using:

```text
app.yaml
```

Example:

```yaml
command:
  - streamlit
  - run
  - app.py
```

Deploy the project folder through Databricks Apps.

## Document RAG Pipeline

Documents are stored in the Common Data Platform volume:

```text
/Volumes/common_data_platform_dev/raw/documents/
```

The document pipeline uses:

```text
PDF
 |
 v
ai_parse_document
 |
 v
ai_prep_search
 |
 v
document_chunks
 |
 v
AI Search Index
 |
 v
Supply Chain Agent
```

The product-owned chunk table is:

```text
common_data_products_dev.agent.document_chunks
```

The AI Search index is built over these chunks.

## Observability and Evaluation

MLflow is used for agent observability and evaluation.

Tracing captures:

- User input
- Agent execution
- Tool calls
- Model responses
- Final output

Evaluation currently covers:

- Response relevance
- Safety
- No-fabrication guidelines
- Correctness

Additional RAG-specific evaluation can be added for retrieval quality and groundedness.

## Current Status

Implemented:

- Structured UC function tools
- Genie analytics integration
- AI Search document retrieval
- LangChain/LangGraph agent
- MCP tool integration
- Streamlit UI
- Databricks App deployment
- MLflow tracing
- Initial agent evaluation
- Source/tool attribution

Planned improvements:

- Genie query performance optimization
- Additional evaluation datasets
- RAG retrieval quality evaluation
- Better source citation display
- Conversation memory
- Streaming responses
- External/public deployment option
- Additional external datasets such as World Bank logistics and economic data

## Design Principles

1. The Common Data Platform owns certified reusable data.
2. Product-specific AI assets stay within the Supply Chain Agent product layer.
3. Exact entity lookups use deterministic functions.
4. Analytical questions use Genie.
5. Document questions use semantic retrieval.
6. Agent responses should be grounded in available tools and data.
7. Business facts should never be fabricated.
8. Development logic remains easy to inspect and debug.

## Disclaimer

The current project uses synthetic supply chain data and synthetic operational documents for development and demonstration purposes.

It is intended as a reference implementation for building governed enterprise AI agents using Databricks.
