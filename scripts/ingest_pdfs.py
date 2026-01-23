from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


def iter_pdf_chunks(
    path: Path,
    chunk_size: int,
    overlap: int,
    max_pages: int | None = None,
    max_chunks: int | None = None,
) -> Iterable[str]:
    reader = PdfReader(str(path))
    if chunk_size <= 0:
        chunk_size = 1200
    overlap = max(0, min(overlap, chunk_size - 1))
    buffer = ""
    chunk_count = 0
    for page_index, page in enumerate(reader.pages, start=1):
        if max_pages and page_index > max_pages:
            break
        text = page.extract_text() or ""
        if not text:
            continue
        if buffer:
            buffer = f"{buffer}\n{text}"
        else:
            buffer = text
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size].strip()
            if chunk:
                chunk_count += 1
                yield chunk
                if max_chunks and chunk_count >= max_chunks:
                    return
            buffer = buffer[chunk_size - overlap :] if overlap else buffer[chunk_size:]
    if buffer.strip():
        chunk_count += 1
        yield buffer.strip()


def embed_text(client: genai.Client, model: str, text: str) -> list[float]:
    response = client.models.embed_content(model=model, contents=[text])
    if not getattr(response, "embeddings", None):
        return []
    embedding = response.embeddings[0]
    return list(getattr(embedding, "values", []))


def flush_batch(
    batch: list[dict],
    insert_url: str,
    headers: dict[str, str],
) -> None:
    if not batch:
        return
    response = requests.post(insert_url, headers=headers, json=batch, timeout=30)
    if response.status_code >= 300:
        raise RuntimeError(f"Insert failed {response.status_code}: {response.text}")
    batch.clear()


def supabase_headers(api_key: str, schema: str | None) -> dict[str, str]:
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    if schema:
        headers["Accept-Profile"] = schema
        headers["Content-Profile"] = schema
    return headers


def delete_existing(url: str, headers: dict[str, str], source: str) -> None:
    params = {"source": f"eq.{source}"}
    response = requests.delete(url, headers=headers, params=params, timeout=30)
    if response.status_code >= 300:
        raise RuntimeError(f"Delete failed {response.status_code}: {response.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF docs into Supabase pgvector.")
    parser.add_argument("--docs-dir", default="docs", help="Directory containing PDFs")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--file", help="Ingest a single PDF file")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages per PDF")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks per PDF")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--replace", action="store_true", help="Replace existing chunks for the same source")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase_schema = os.getenv("SUPABASE_SCHEMA")
    docs_table = os.getenv("SUPABASE_DOCS_TABLE", "documents")

    if not supabase_url or not supabase_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_KEY are required.")

    print(f"Loading embedding model: {embed_model}")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing.")
    client = genai.Client(api_key=api_key)
    print("Model loaded.")
    docs_path = Path(args.docs_dir)
    if args.file:
        pdf_paths = [Path(args.file)]
    else:
        pdf_paths = sorted(docs_path.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {docs_path}")

    base_url = supabase_url.rstrip("/")
    insert_url = f"{base_url}/rest/v1/{docs_table}"
    headers = supabase_headers(supabase_key, supabase_schema)

    for pdf_path in pdf_paths:
        source = pdf_path.name
        print(f"Processing {source}...")
        if args.replace:
            delete_existing(insert_url, headers, source)

        rows: list[dict] = []
        chunk_count = 0
        for index, chunk in enumerate(
            iter_pdf_chunks(
                pdf_path,
                args.chunk_size,
                args.overlap,
                max_pages=args.max_pages or None,
                max_chunks=args.max_chunks or None,
            )
        ):
            embedding = embed_text(client, embed_model, chunk)
            if not embedding:
                raise RuntimeError(f"Failed to embed chunk {index} for {source}")
            rows.append(
                {
                    "content": chunk,
                    "embedding": embedding,
                    "source": source,
                    "chunk_index": index,
                }
            )
            chunk_count += 1
            if args.progress_every and chunk_count % args.progress_every == 0:
                print(f"{source}: embedded {chunk_count} chunks")
            if len(rows) >= args.batch_size:
                flush_batch(rows, insert_url, headers)

        flush_batch(rows, insert_url, headers)
        if chunk_count == 0:
            print(f"Skipping {source}: no text extracted")
            continue
        print(f"Ingested {chunk_count} chunks from {source}")


if __name__ == "__main__":
    main()
