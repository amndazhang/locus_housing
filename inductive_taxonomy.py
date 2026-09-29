import duckdb
import pandas as pd
import json
from openai import OpenAI

client = OpenAI() # Uses OPENAI_API_KEY environment variable

# -------------------------------------------------------------
# Step 1: Draw a Stratified Sample across all 5 original topics
# -------------------------------------------------------------
con = duckdb.connect()

# Draw 300 random chunks from EACH of the 5 topics (1,500 total)
sample_df = con.query("""
    SELECT topic, header, content
    FROM (
        SELECT topic, header, content, ROW_NUMBER() OVER (PARTITION BY topic ORDER BY RANDOM()) as r
        FROM 'locus_substantive.parquet'
    )
    WHERE r <= 300
""").to_df()

print(f"Sampled {len(sample_df)} chunks for topic extraction.")

# -------------------------------------------------------------
# Step 2: Extract Micro-Topics using LLM
# -------------------------------------------------------------
def extract_micro_topics(header, content):
    prompt = f"""Analyze this local ordinance chunk and list 1 to 3 specific sub-topics or regulatory subjects it directly governs.
    Focus on specific subjects (e.g., "Off-street parking requirements", "Short-term rental licensing", "Setback distances", "Noise curfews").

    Header: {header}
    Content: {content[:1000]}

    Return ONLY a JSON array of strings, e.g. ["Topic 1", "Topic 2"].
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("topics", []) if isinstance(data, dict) else data
    except Exception as e:
        return []

extracted_topics = []

print("Running batch extraction...")
for idx, row in sample_df.iterrows():
    topics = extract_micro_topics(row['header'], row['content'])
    extracted_topics.extend(topics)
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(sample_df)} chunks...")

# Deduplicate raw tags
unique_tags = list(set([t.lower().strip() for t in extracted_topics if t]))
print(f"\nExtracted {len(extracted_topics)} total tags ({len(unique_tags)} unique).")

# -------------------------------------------------------------
# Step 3: Synthesize into Hierarchical Taxonomy
# -------------------------------------------------------------
synthesis_prompt = f"""
You are an expert urban policy taxonomist. Below is a raw list of unique sub-topics extracted from local legal codes across the US.
Merge this list with the seed taxonomy provided below to create a comprehensive, 3-tier hierarchy of all topics covered in local codes.

SEED TOPICS:
- Zoning and land use (Density, ADUs, Setbacks, Variances)
- Occupants and tenants (Leases, Rent regulation, Evictions, Occupancy limits)
- Landlords and owners (Registration, Obligations, HOAs)
- Building & Safety (Habitability, Codes, Inspections)
- Affordable Housing, STRs, Homelessness
- Non-Housing (Commercial, Traffic, Public Space)

RAW EXTRACTED TAGS:
{json.dumps(unique_tags[:800], indent=2)}

OUTPUT FORMAT:
Provide a structured Markdown hierarchy containing:
1. Primary Category (Tier 1)
   - Secondary Category (Tier 2)
     * Specific Regulatory Subjects (Tier 3)
2. Indicate for each Tier 1 category whether it is: [CORE HOUSING], [HOUSING-ADJACENT], or [NOT HOUSING].
"""

print("\nSynthesizing full taxonomy hierarchy...")
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": synthesis_prompt}],
    temperature=0.2
)

taxonomy_output = final_response.choices[0].message.content

# Save taxonomy output
with open("extracted_locus_taxonomy.md", "w") as f:
    f.write(taxonomy_output)

print("Done! Taxonomy saved to 'extracted_locus_taxonomy.md'.")