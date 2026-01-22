# Business Analyst Agent

This is a Gemini-powered agent that answers sales and inventory questions
using the local CSVs:

- `data/sales_data.csv`
- `data/inventory_data.csv`

It uses 5 tools under the hood: schema info, filter listing, sales aggregation,
inventory status, and product joins.

## Setup

Create a `.env` file with your API key:

```bash
GEMINI_API_KEY=your_api_key_here
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

## Example questions

- "Highest selling product by units from 2024-01-01 to 2024-03-31"
- "Least available product in Dallas warehouse"
- "Top 5 products by revenue last quarter in Online channel"
- "Which products are low stock in New York?"
