import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

MODEL_NAME = os.getenv("MODEL_NAME", "databricks-gpt-oss-120b")

PLATFORM_CATALOG = f"common_data_platform_{ENVIRONMENT}"
PRODUCT_CATALOG = f"common_data_products_{ENVIRONMENT}"

AGENT_SCHEMA = "agent"

GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
AI_SEARCH_INDEX = os.getenv("AI_SEARCH_INDEX", "document_chunks_index")