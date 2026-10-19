import os

# Catalog names are configurable so dev/uat/prod can point at different layers.
UC_GOLD_LAYER = os.getenv("UC_GOLD_LAYER", "wb_transport_gold")
UC_SILVER_LAYER = os.getenv("UC_SILVER_LAYER", "wb_transport_silver")
PROCUREMENT_SCHEMA = os.getenv("PROCUREMENT_SCHEMA", "procurement")

PROJECT_TABLE = (
    f"{UC_GOLD_LAYER}.{PROCUREMENT_SCHEMA}.project_procurement_summary"
)
TIMELINE_TABLE = (
    f"{UC_GOLD_LAYER}.{PROCUREMENT_SCHEMA}.procurement_timeline"
)
RISK_TABLE = (
    f"{UC_GOLD_LAYER}.{PROCUREMENT_SCHEMA}.procurement_risk_indicators"
)
VECTOR_INDEX_NAME = (
    f"{UC_SILVER_LAYER}.{PROCUREMENT_SCHEMA}.document_chunks_index"
)
MODEL_NAME = os.getenv("MODEL_NAME", "databricks-gpt-oss-120b")

# Required only if structured queries use the Databricks SQL connector.
# Keep unset while the app is configured to use a different query path.
SQL_WAREHOUSE_HTTP_PATH = os.getenv("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH")
