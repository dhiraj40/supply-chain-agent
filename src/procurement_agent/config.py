import os

# Catalog names are configurable so dev/uat/prod can point at different layers.
UC_SILVER_LAYER = os.getenv("UC_SILVER_LAYER", "wb_transport_silver")
PROCUREMENT_SCHEMA = os.getenv("PROCUREMENT_SCHEMA", "procurement")

VECTOR_INDEX_NAME = (
    f"{UC_SILVER_LAYER}.{PROCUREMENT_SCHEMA}.document_chunks_index"
)
MODEL_NAME = os.getenv("MODEL_NAME", "databricks-gpt-oss-120b")

# Dedicated Genie Agent for procurement analytics.
# Configure this value in the Databricks App environment.
PROCUREMENT_GENIE_SPACE_ID = os.getenv("PROCUREMENT_GENIE_SPACE_ID")
