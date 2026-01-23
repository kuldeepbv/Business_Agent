-- Enable pgvector extension
create extension if not exists vector;

-- Create documents table (Gemini text-embedding-004 uses 768-dim embeddings)
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(768) not null,
  source text,
  chunk_index integer,
  created_at timestamptz default now()
);

-- Vector index for faster search (adjust ops based on your similarity)
create index if not exists documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops);

-- Similarity search function
create or replace function match_documents(
  query_embedding vector(768),
  match_count int
)
returns table (
  id uuid,
  content text,
  source text,
  chunk_index integer,
  similarity float
)
language sql stable as $$
  select
    id,
    content,
    source,
    chunk_index,
    1 - (embedding <-> query_embedding) as similarity
  from documents
  order by embedding <-> query_embedding
  limit match_count;
$$;
