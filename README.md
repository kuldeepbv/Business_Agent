# Business Analyst Agent

This is a Gemini-powered agent that answers sales and inventory questions
using local CSVs or Supabase (when configured):

- `data/sales_data.csv`
- `data/inventory_data.csv`

It uses 5 tools under the hood: schema info, filter listing, sales aggregation,
inventory status, and product joins.

## Setup

Create a `.env` file with your API key and (optional) Supabase credentials:

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_EMBED_MODEL=text-embedding-004
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_SCHEMA=public
SUPABASE_SALES_TABLE=sales
SUPABASE_INVENTORY_TABLE=inventory
SUPABASE_PAGE_SIZE=1000
SUPABASE_DOCS_TABLE=documents
SUPABASE_RPC_MATCH=match_documents
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Interactive mode:

```bash
python agent.py
```

Single question:

```bash
python agent.py --question "Top 3 products by revenue from 2024-01-01 to 2024-03-31"
```

## Notes

- For sales questions, the agent requires both a metric (units or revenue) and
  a date range unless you provide a relative range like "last month" or "Q2 2024".
- Inventory questions can filter by warehouse, category, or supplier.
- You can change models with `--model gemini-2.5-flash`.
- If Supabase env vars are present, the agent reads data from Supabase via REST; otherwise it falls back to local CSVs.
- For RAG, create the pgvector table and RPC (see `docs/rag_setup.sql`), then ingest PDFs with `scripts/ingest_pdfs.py`.

## RAG Setup (Supabase pgvector)

1) Run `docs/rag_setup.sql` in Supabase SQL editor.
   - Gemini text-embedding-004 uses 768 dimensions (`vector(768)`).
2) Put PDFs in `docs/`.
3) Ingest:

```bash
python scripts/ingest_pdfs.py --docs-dir docs --replace
```

This will embed and store chunks in `SUPABASE_DOCS_TABLE`.

## Example questions

- "Highest selling product by units from 2024-01-01 to 2024-03-31"
- "Least available product in Dallas warehouse"
- "Top 5 products by revenue last quarter in Online channel"
- "Which products are low stock in New York?"
