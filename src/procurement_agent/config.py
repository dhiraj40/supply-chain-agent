import os

# These are Unity Catalog three-part table names.
PROJECT_TABLE = os.getenv(
    "PROCUREMENT_PROJECT_TABLE",
    "wb_transport_gold.procurement.project_procurement_summary",
)
TIMELINE_TABLE = os.getenv(
    "PROCUREMENT_TIMELINE_TABLE",
    "wb_transport_gold.procurement.procurement_timeline",
)
RISK_TABLE = os.getenv(
    "PROCUREMENT_RISK_TABLE",
    "wb_transport_gold.procurement.procurement_risk_indicators",
)
VECTOR_INDEX_NAME = os.getenv(
    "PROCUREMENT_VECTOR_INDEX",
    "wb_transport_silver.procurement.document_chunks_index",
)
MODEL_NAME = os.getenv("MODEL_NAME", "databricks-gpt-oss-120b")
# A SQL Warehouse is required for app-side structured SQL queries.
SQL_WAREHOUSE_HTTP_PATH = os.getenv("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH")
