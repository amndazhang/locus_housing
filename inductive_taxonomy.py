import asyncio
import duckdb
import json
import os
import pandas as pd
from openai import AsyncOpenAI, OpenAI

# -------------------------------------------------------------
# JEV (OpenRouter) Configuration
# -------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JEV_API_KEY = os.getenv("JEV_API_KEY", os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"))

# OpenRouter Chat Models (Avoid decision models like jev-1.13 for completions)
FAST_MODEL = "openai/gpt-4o-mini"
REASONING_MODEL = "openai/gpt-4o"

DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/LocalLaws/LOCUS",
    "X-Title": "LOCUS Housing Taxonomy Discovery"
}

async_client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=JEV_API_KEY,
    default_headers=DEFAULT_HEADERS
)

sync_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=JEV_API_KEY,
    default_headers=DEFAULT_HEADERS
)

# -------------------------------------------------------------
# Step 1: Draw Stratified Sample
# -------------------------------------------------------------
def get_sample_data(db_path: str = "locus_substantive.parquet", samples_per_topic: int = 300) -> pd.DataFrame:
    print(f"Sampling {samples_per_topic} chunks per topic from {db_path}...")
    con = duckdb.connect()
    
    query = f"""
        SELECT topic, header, content
        FROM (
            SELECT topic, header, content, ROW_NUMBER() OVER (PARTITION BY topic ORDER BY RANDOM()) as r
            FROM '{db_path}'
        )
        WHERE r <= {samples_per_topic}
    """
    df = con.query(query).to_df()
    print(f"Loaded {len(df):,} total sampled chunks.")
    return df

# -------------------------------------------------------------
# Step 2: Async Concurrent Batch Extraction
# -------------------------------------------------------------
async def extract_single_chunk(semaphore: asyncio.Semaphore, header: str, content: str, original_topic: str) -> list[dict]:
    async with semaphore:
        prompt = f"""Analyze this local ordinance chunk and list 1 to 3 specific sub-topics or regulatory subjects it directly governs.
Focus on concrete regulatory subjects (e.g., "Off-street parking requirements", "Short-term rental licensing", "Setback distances", "Noise curfews").

Header: {header}
Content: {content[:1200]}

Return JSON format with a "topics" array:
{{"topics": ["Topic 1", "Topic 2"]}}
"""

        try:
            response = await async_client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content
            data = json.loads(raw_text)
            topics = data.get("topics", []) if isinstance(data, dict) else []
            return [{"tag": t.strip().lower(), "locus_topic": original_topic} for t in topics if isinstance(t, str) and t.strip()]
        
        except Exception as e:
            print(f"[OpenRouter API Error]: {type(e).__name__} - {e}")
            return []

async def run_batch_extraction(df: pd.DataFrame, max_concurrent: int = 20) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        extract_single_chunk(semaphore, row['header'], row['content'], row['topic'])
        for _, row in df.iterrows()
    ]
    
    print(f"Extracting micro-topics via OpenRouter (Max concurrency: {max_concurrent})...")
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

# -------------------------------------------------------------
# Step 3: Synthesize Taxonomy (With Repetition Prevention)
# -------------------------------------------------------------
def synthesize_taxonomy(extracted_data: list[dict], output_file: str = "extracted_locus_taxonomy.md"):
    if not extracted_data:
        return

    df_tags = pd.DataFrame(extracted_data)
    tag_counts = df_tags['tag'].value_counts().head(250).to_dict()

    prompt = f"""You are an expert urban policy taxonomist.
Analyze the empirical micro-topic frequency distribution below from local legal codes across the US, and construct a 3-tier taxonomy.

TOP EXTRACTED MICRO-TOPICS:
{json.dumps(tag_counts, indent=2)}

INSTRUCTIONS:
1. Divide topics into 3 primary tiers:
   - [CORE HOUSING]: Regulations directly governing who can live in a home, housing safety/habitability, or housing unit development.
   - [HOUSING-ADJACENT]: Regulations on property-adjacent land, infrastructure, or nuisances that affect residential living but do not directly create or regulate residential occupancy.
   - [NOT HOUSING]: General municipal, commercial, or public safety regulations unrelated to residential housing.

2. FOR EACH TIER 2 SUB-CATEGORY, explicitly state a **Classification Rationale** explaining WHY it belongs in Core, Adjacent, or Non-Housing.

OUTPUT FORMAT:
# Tier 1: Category Name [CLASSIFICATION]
**Rationale:** <1-2 sentences explaining why this entire category falls into this tier>

- **Tier 2: Sub-category Name**
  - Tier 3 Bullet Points (max 8)
"""

    print(f"\nSynthesizing taxonomy with classification rationales using {REASONING_MODEL}...")
    
    response = sync_client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        presence_penalty=0.5
    )

    taxonomy = response.choices[0].message.content

    with open(output_file, "w") as f:
        f.write(taxonomy)
    
    print(f"Success! Taxonomy with rationales saved to '{output_file}'.")

# -------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------
async def main():
    # Load raw extracted tags if already saved locally from Step 2 to save credits
    if os.path.exists("raw_extracted_micro_topics.csv"):
        print("Found existing 'raw_extracted_micro_topics.csv'. Loading directly...")
        df_raw = pd.read_csv("raw_extracted_micro_topics.csv")
        extracted_data = df_raw.to_dict(orient="records")
    else:
        sample_df = get_sample_data(db_path="locus_substantive.parquet", samples_per_topic=300)
        extracted_data = await run_batch_extraction(sample_df, max_concurrent=20)
        if extracted_data:
            pd.DataFrame(extracted_data).to_csv("raw_extracted_micro_topics.csv", index=False)
            print("Saved raw tags to 'raw_extracted_micro_topics.csv'")

    # Run cleaned synthesis step
    synthesize_taxonomy(extracted_data)

if __name__ == "__main__":
    asyncio.run(main())