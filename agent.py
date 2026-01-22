from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: google-genai. Install with: pip install google-genai"
    ) from exc


SALES_REQUIRED_COLUMNS = {
    "product_id",
    "sale_date",
    "quantity_sold",
    "sale_price",
    "region",
    "sales_channel",
}
SALES_ID_COLUMNS = {"sale_id", "id"}
INVENTORY_REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "supplier",
    "unit_cost",
    "stock_quantity",
    "reorder_level",
    "warehouse",
}

SYSTEM_PROMPT = (
    "You are a Business Analyst agent for sales and inventory data. "
    "Use the available tools to compute answers. "
    "Do not assume missing inputs. If metric or date range is missing for sales questions, "
    "ask a clarification question. "
    "When returning sales results, use join_products to include product_name. "
    "Keep answers concise with a short explanation."
)


class DataStore:
    def __init__(
        self,
        sales_path: Path,
        inventory_path: Path,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_schema: Optional[str] = None,
        sales_table: str = "sales",
        inventory_table: str = "inventory",
        page_size: int = 1000,
    ) -> None:
        self.sales_path = sales_path
        self.inventory_path = inventory_path
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase_schema = supabase_schema
        self.sales_table = sales_table
        self.inventory_table = inventory_table
        self.page_size = page_size
        self.sales: Optional[pd.DataFrame] = None
        self.inventory: Optional[pd.DataFrame] = None

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.sales is None or self.inventory is None:
            if self._use_supabase():
                sales = self._load_supabase_table(self.sales_table)
                inventory = self._load_supabase_table(self.inventory_table)
            else:
                sales = pd.read_csv(self.sales_path)
                inventory = pd.read_csv(self.inventory_path)
            self._validate_columns(sales, inventory)
            self.sales = self._normalize_sales(sales)
            self.inventory = self._normalize_inventory(inventory)
        return self.sales, self.inventory

    def schema_info(self, dataset: str) -> dict[str, Any]:
        sales, inventory = self.load_data()
        if dataset == "sales":
            df = sales
        elif dataset == "inventory":
            df = inventory
        else:
            raise ValueError("dataset must be 'sales' or 'inventory'")
        return {
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isna().sum().to_dict(),
            "sample": df.head(5).to_dict(orient="records"),
        }

    def value_map(self) -> dict[str, list[str]]:
        sales, inventory = self.load_data()
        return {
            "region": sorted(sales["region"].dropna().unique().tolist()),
            "sales_channel": sorted(sales["sales_channel"].dropna().unique().tolist()),
            "warehouse": sorted(inventory["warehouse"].dropna().unique().tolist()),
            "category": sorted(inventory["category"].dropna().unique().tolist()),
            "supplier": sorted(inventory["supplier"].dropna().unique().tolist()),
            "product_name": sorted(inventory["product_name"].dropna().unique().tolist()),
        }

    def _validate_columns(self, sales: pd.DataFrame, inventory: pd.DataFrame) -> None:
        sales_missing = SALES_REQUIRED_COLUMNS.difference(sales.columns)
        inventory_missing = INVENTORY_REQUIRED_COLUMNS.difference(inventory.columns)
        if sales_missing:
            raise ValueError(f"Sales data missing columns: {sorted(sales_missing)}")
        if not any(col in sales.columns for col in SALES_ID_COLUMNS):
            raise ValueError(
                "Sales data missing identifier column: expected one of "
                f"{sorted(SALES_ID_COLUMNS)}"
            )
        if inventory_missing:
            raise ValueError(f"Inventory data missing columns: {sorted(inventory_missing)}")

    def _normalize_sales(self, sales: pd.DataFrame) -> pd.DataFrame:
        sales = sales.copy()
        sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
        sales["quantity_sold"] = pd.to_numeric(sales["quantity_sold"], errors="coerce")
        sales["sale_price"] = pd.to_numeric(sales["sale_price"], errors="coerce")
        return sales

    def _normalize_inventory(self, inventory: pd.DataFrame) -> pd.DataFrame:
        inventory = inventory.copy()
        inventory["unit_cost"] = pd.to_numeric(inventory["unit_cost"], errors="coerce")
        inventory["stock_quantity"] = pd.to_numeric(
            inventory["stock_quantity"], errors="coerce"
        )
        inventory["reorder_level"] = pd.to_numeric(
            inventory["reorder_level"], errors="coerce"
        )
        return inventory

    def _use_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def _load_supabase_table(self, table: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        start = 0
        base_url = self.supabase_url.rstrip("/")
        url = f"{base_url}/rest/v1/{table}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }
        if self.supabase_schema:
            headers["Accept-Profile"] = self.supabase_schema

        while True:
            params = {"select": "*", "limit": self.page_size, "offset": start}
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code >= 300:
                raise ValueError(
                    f"Supabase error {response.status_code}: {response.text}"
                )
            batch = response.json() or []
            rows.extend(batch)
            if len(batch) < self.page_size:
                break
            start += self.page_size
        return pd.DataFrame(rows)


class BusinessAnalystTools:
    def __init__(self, datastore: DataStore) -> None:
        self.datastore = datastore
        self.sales, self.inventory = self.datastore.load_data()

    def schema_info(self, dataset: str) -> dict[str, Any]:
        return self.datastore.schema_info(dataset)

    def list_filters(self) -> dict[str, Any]:
        return self.datastore.value_map()

    def sales_agg(
        self,
        metric: str,
        date_range: str,
        region: Optional[str] = None,
        sales_channel: Optional[str] = None,
        limit: int = 10,
        order: str = "desc",
    ) -> dict[str, Any]:
        if not metric:
            return tool_error("metric is required", needs_clarification=True)
        if not date_range:
            return tool_error("date_range is required", needs_clarification=True)
        metric = metric.lower().strip()
        if metric not in {"units", "revenue"}:
            return tool_error("metric must be 'units' or 'revenue'")

        parsed_range, note = parse_date_range_input(date_range, self.sales)
        if not parsed_range:
            return tool_error(
                "date_range could not be parsed (example: 2024-01-01 to 2024-03-31)"
            )

        start_date, end_date = parsed_range
        df = self.sales.dropna(subset=["sale_date"]).copy()
        df = df[(df["sale_date"] >= start_date) & (df["sale_date"] <= end_date)]
        if region:
            df = df[df["region"].str.lower() == region.lower()]
        if sales_channel:
            df = df[df["sales_channel"].str.lower() == sales_channel.lower()]

        df["revenue"] = df["quantity_sold"] * df["sale_price"]
        metric_column = "units_sold" if metric == "units" else "revenue"
        if metric == "units":
            grouped = df.groupby("product_id", as_index=False)["quantity_sold"].sum()
            grouped = grouped.rename(columns={"quantity_sold": metric_column})
        else:
            grouped = df.groupby("product_id", as_index=False)["revenue"].sum()
            grouped = grouped.rename(columns={"revenue": metric_column})

        ascending = order.lower() == "asc"
        grouped = grouped.sort_values(metric_column, ascending=ascending)
        if limit and limit > 0:
            grouped = grouped.head(limit)

        rows = grouped.to_dict(orient="records")
        return {
            "ok": True,
            "rows": rows,
            "metric": metric,
            "metric_column": metric_column,
            "date_range": {
                "start": start_date.date().isoformat(),
                "end": end_date.date().isoformat(),
                "note": note,
            },
            "filters": {
                "region": region,
                "sales_channel": sales_channel,
            },
            "row_count": len(rows),
        }

    def inventory_status(
        self,
        warehouse: Optional[str] = None,
        low_stock_only: bool = False,
        limit: int = 10,
        order: str = "asc",
    ) -> dict[str, Any]:
        df = self.inventory.copy()
        if warehouse:
            df = df[df["warehouse"].str.lower() == warehouse.lower()]

        df["stock_gap"] = df["reorder_level"] - df["stock_quantity"]
        df["low_stock"] = df["stock_quantity"] <= df["reorder_level"]
        if low_stock_only:
            df = df[df["low_stock"]]

        ascending = order.lower() == "asc"
        df = df.sort_values("stock_quantity", ascending=ascending)
        if limit and limit > 0:
            df = df.head(limit)

        rows = df[
            [
                "product_id",
                "product_name",
                "category",
                "supplier",
                "stock_quantity",
                "reorder_level",
                "warehouse",
                "low_stock",
            ]
        ].to_dict(orient="records")
        return {
            "ok": True,
            "rows": rows,
            "filters": {
                "warehouse": warehouse,
                "low_stock_only": low_stock_only,
            },
            "row_count": len(rows),
        }

    def join_products(self, product_ids: list[str]) -> dict[str, Any]:
        if not product_ids:
            return tool_error("product_ids are required", needs_clarification=True)
        normalized = [pid.strip().upper() for pid in product_ids if pid.strip()]
        df = self.inventory[self.inventory["product_id"].isin(normalized)]
        rows = df[
            [
                "product_id",
                "product_name",
                "category",
                "supplier",
                "warehouse",
                "stock_quantity",
                "reorder_level",
            ]
        ].to_dict(orient="records")
        return {"ok": True, "rows": rows, "row_count": len(rows)}


class GeminiAgent:
    def __init__(self, tools: BusinessAnalystTools, api_key: str, model: str) -> None:
        self.tools = tools
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tool_registry = {
            "schema_info": self.tools.schema_info,
            "list_filters": self.tools.list_filters,
            "sales_agg": self.tools.sales_agg,
            "inventory_status": self.tools.inventory_status,
            "join_products": self.tools.join_products,
        }
        self.tool_config = [
            types.Tool(function_declarations=build_tool_declarations())
        ]

    def ask(self, question: str, history: Optional[list[types.Content]] = None) -> tuple[str, list[types.Content]]:
        contents = history[:] if history else []
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=self.tool_config,
            temperature=0.2,
        )

        for _ in range(8):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            candidate = response.candidates[0].content
            tool_calls = [
                part.function_call for part in candidate.parts if part.function_call
            ]
            if not tool_calls:
                contents.append(candidate)
                return content_to_text(candidate), contents

            contents.append(candidate)
            for call in tool_calls:
                result = self._run_tool(call)
                tool_part = types.Part.from_function_response(
                    name=call.name, response=result
                )
                contents.append(types.Content(role="tool", parts=[tool_part]))

        return "I hit the tool call limit. Please narrow the question.", contents

    def _run_tool(self, call: types.FunctionCall) -> dict[str, Any]:
        handler = self.tool_registry.get(call.name)
        if not handler:
            return tool_error(f"Unknown tool: {call.name}")
        args = call.args or {}
        try:
            return handler(**args)
        except TypeError as exc:
            return tool_error(f"Tool argument error: {exc}")
        except Exception as exc:
            return tool_error(f"Tool error: {exc}")


def tool_error(message: str, needs_clarification: bool = False) -> dict[str, Any]:
    return {"ok": False, "error": message, "needs_clarification": needs_clarification}


def build_tool_declarations() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="schema_info",
            description="Return columns, dtypes, null counts, and sample rows for a dataset.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "dataset": types.Schema(
                        type=types.Type.STRING, enum=["sales", "inventory"]
                    )
                },
                required=["dataset"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_filters",
            description="List available values for region, sales_channel, warehouse, category, supplier, product_name.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="sales_agg",
            description=(
                "Aggregate sales by product_id for a metric over a date range. "
                "Requires metric and date_range. date_range can be 'YYYY-MM-DD to YYYY-MM-DD', "
                "'last month', 'this quarter', or 'Q2 2024'."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "metric": types.Schema(
                        type=types.Type.STRING, enum=["units", "revenue"]
                    ),
                    "date_range": types.Schema(type=types.Type.STRING),
                    "region": types.Schema(type=types.Type.STRING),
                    "sales_channel": types.Schema(type=types.Type.STRING),
                    "limit": types.Schema(type=types.Type.INTEGER),
                    "order": types.Schema(type=types.Type.STRING, enum=["asc", "desc"]),
                },
                required=["metric", "date_range"],
            ),
        ),
        types.FunctionDeclaration(
            name="inventory_status",
            description=(
                "Return inventory rows with stock and reorder info. "
                "Use low_stock_only to filter stock_quantity <= reorder_level."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "warehouse": types.Schema(type=types.Type.STRING),
                    "low_stock_only": types.Schema(type=types.Type.BOOLEAN),
                    "limit": types.Schema(type=types.Type.INTEGER),
                    "order": types.Schema(type=types.Type.STRING, enum=["asc", "desc"]),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="join_products",
            description="Lookup product details for a list of product_ids.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "product_ids": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    )
                },
                required=["product_ids"],
            ),
        ),
    ]


def parse_date_range_input(
    text: str, sales_df: pd.DataFrame
) -> tuple[Optional[tuple[pd.Timestamp, pd.Timestamp]], Optional[str]]:
    if not text:
        return None, None
    return parse_date_range(text, sales_df)


def parse_date_range(
    question: str, sales_df: pd.DataFrame
) -> tuple[Optional[tuple[pd.Timestamp, pd.Timestamp]], Optional[str]]:
    q = question.lower()
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    match = re.search(rf"from {date_pattern} to {date_pattern}", q)
    if not match:
        match = re.search(rf"between {date_pattern} and {date_pattern}", q)
    if not match:
        match = re.search(rf"{date_pattern}\s+to\s+{date_pattern}", q)
    if match:
        start = pd.to_datetime(match.group(1), errors="coerce")
        end = pd.to_datetime(match.group(2), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return None, None
        return (start, end), None

    date_note = None
    reference_date = sales_df["sale_date"].max()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp(datetime.utcnow().date())

    if "last month" in q:
        start, end = month_range(reference_date - pd.DateOffset(months=1))
        date_note = "Relative to latest sale_date in data"
        return (start, end), date_note
    if "this month" in q:
        start, end = month_range(reference_date)
        date_note = "Relative to latest sale_date in data"
        return (start, end), date_note
    if "last quarter" in q:
        start, end = quarter_range(reference_date - pd.DateOffset(months=3))
        date_note = "Relative to latest sale_date in data"
        return (start, end), date_note
    if "this quarter" in q:
        start, end = quarter_range(reference_date)
        date_note = "Relative to latest sale_date in data"
        return (start, end), date_note
    if "last year" in q:
        year = reference_date.year - 1
        return (
            pd.Timestamp(year=year, month=1, day=1),
            pd.Timestamp(year=year, month=12, day=31),
        ), "Relative to latest sale_date in data"
    if "this year" in q:
        year = reference_date.year
        return (
            pd.Timestamp(year=year, month=1, day=1),
            pd.Timestamp(year=year, month=12, day=31),
        ), "Relative to latest sale_date in data"

    quarter_match = re.search(r"q([1-4])\s*(\d{4})", q)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        start, end = quarter_range_for_year(quarter, year)
        return (start, end), None

    return None, None


def month_range(reference: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=reference.year, month=reference.month, day=1)
    next_month = start + pd.DateOffset(months=1)
    end = next_month - pd.Timedelta(days=1)
    return start, end


def quarter_range(reference: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    quarter = (reference.month - 1) // 3 + 1
    return quarter_range_for_year(quarter, reference.year)


def quarter_range_for_year(quarter: int, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_month = (quarter - 1) * 3 + 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end_month = start_month + 2
    end = (
        pd.Timestamp(year=year, month=end_month, day=1) + pd.DateOffset(months=1)
    ) - pd.Timedelta(days=1)
    return start, end


def content_to_text(content: types.Content) -> str:
    parts = []
    for part in content.parts:
        if part.text:
            parts.append(part.text)
    return "".join(parts).strip()


def run_interactive(agent: GeminiAgent) -> None:
    print("Business Analyst Agent (Gemini) ready. Type 'exit' to quit.")
    history: list[types.Content] = []
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer, history = agent.ask(question, history=history)
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Business Analyst Agent (Gemini)")
    parser.add_argument(
        "--sales",
        type=Path,
        default=Path("data/sales_data.csv"),
        help="Path to sales CSV",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/inventory_data.csv"),
        help="Path to inventory CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument("--question", type=str, help="Single question to answer")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing. Set it in a .env file.")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase_schema = os.getenv("SUPABASE_SCHEMA")
    sales_table = os.getenv("SUPABASE_SALES_TABLE", "sales")
    inventory_table = os.getenv("SUPABASE_INVENTORY_TABLE", "inventory")
    page_size = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))

    datastore = DataStore(
        args.sales,
        args.inventory,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        supabase_schema=supabase_schema,
        sales_table=sales_table,
        inventory_table=inventory_table,
        page_size=page_size,
    )
    tools = BusinessAnalystTools(datastore)
    agent = GeminiAgent(tools, api_key=api_key, model=args.model)

    if args.question:
        answer, _ = agent.ask(args.question)
        print(answer)
        return

    run_interactive(agent)


if __name__ == "__main__":
    main()
