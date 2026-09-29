import duckdb

print("Fetching all substantive laws from LOCUS-v1 across all topics...")

# Fetch all substantive laws directly via DuckDB
query = """
SELECT 
    header,
    content,
    is_substantive,
    function,
    topic,
    source_jurisdiction_type,
    state,
    city,
    county,
    enforcement_discretion,
    opacity,
    paternalism,
    problem_salience
FROM 'hf://datasets/LocalLaws/LOCUS-v1/**/*.parquet'
WHERE is_substantive = TRUE
"""

# Execute query and write directly to a local Parquet file
duckdb.query(f"COPY ({query}) TO 'locus_substantive.parquet' (FORMAT PARQUET)")

print("Done! All substantive laws saved locally to 'locus_substantive.parquet'.")