import duckdb
import pandas as pd

# 1. Connect to the local Parquet file downloaded by download.py
con = duckdb.connect()

# Total count per topic in the substantive dataset
print("--- Total Substantive Laws by Topic ---")
topic_counts = con.query("""
    SELECT topic, COUNT(*) as count 
    FROM 'locus_substantive.parquet' 
    GROUP BY topic 
    ORDER BY count DESC
""").to_df()
print(topic_counts)

print("\n--------------------------------------------------")

# 2. Search for Housing-related laws across ALL topics
housing_query = """
SELECT 
    header,
    content,
    topic,
    state,
    city,
    county,
    CASE 
        WHEN content ~* '\\b(housing|dwelling|residential|apartment|condo|single-family|multi-family|adu|accessory dwelling)\\b' THEN 'Core Housing'
        WHEN content ~* '\\b(tenant|occupan|landlord|rent control|lease|eviction|habitability|fair housing)\\b' THEN 'Tenancy & Rights'
        WHEN content ~* '\\b(short-term rental|str|airbnb|boarding house|shelter|unhoused)\\b' THEN 'STRs & Shelters'
        ELSE 'Other Housing'
    END as housing_category
FROM 'locus_substantive.parquet'
WHERE 
    content ~* '\\b(housing|dwelling|residential|apartment|condo|single-family|multi-family|adu|accessory dwelling)\\b'
    OR content ~* '\\b(tenant|occupan|landlord|rent control|lease|eviction|habitability|fair housing)\\b'
    OR content ~* '\\b(short-term rental|str|airbnb|boarding house|shelter|unhoused)\\b'
"""

df_housing = con.query(housing_query).to_df()

print(f"\nTotal Housing Candidate Provisions Found: {len(df_housing):,}")

# Breakdown of where Housing laws fall across LOCUS topics
print("\n--- Distribution of Housing Laws Across LOCUS Topics ---")
print(pd.crosstab(df_housing["topic"], df_housing["housing_category"], margins=True))

# 3. Export a sample chunk set for manual codebook labeling
sample_set = df_housing.groupby("topic", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 100), random_state=42)
)
sample_set.to_csv("housing_sampled_chunks.csv", index=False)
print("\nExported 100 sampled chunks per topic to 'housing_sampled_chunks.csv' for codebook review.")