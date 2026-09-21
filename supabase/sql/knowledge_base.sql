-- Fase 0 — Repositorio de conocimiento (RAG) admin-only. Voyage embeddings dim 1024.
create extension if not exists vector;

create table if not exists knowledge_docs (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  source_type text not null default 'manual',   -- manual | curso | norma | otro
  machine_model text,                            -- para match por modelo (manuales)
  filename text,
  n_chunks int default 0,
  uploaded_by text,
  created_at timestamptz default now()
);

create table if not exists knowledge_chunks (
  id bigserial primary key,
  doc_id uuid references knowledge_docs(id) on delete cascade,
  chunk_index int not null,
  content text not null,
  embedding vector(1024),
  created_at timestamptz default now()
);

create index if not exists idx_knowledge_chunks_doc on knowledge_chunks(doc_id);
create index if not exists idx_knowledge_chunks_embedding
  on knowledge_chunks using hnsw (embedding vector_cosine_ops);

create or replace function match_knowledge(
  query_embedding vector(1024),
  match_count int default 6,
  filter_model text default null
) returns table (
  id bigint, doc_id uuid, title text, source_type text,
  machine_model text, content text, similarity float
) language sql stable as $$
  select c.id, c.doc_id, d.title, d.source_type, d.machine_model,
         c.content, 1 - (c.embedding <=> query_embedding) as similarity
  from knowledge_chunks c
  join knowledge_docs d on d.id = c.doc_id
  where filter_model is null
     or d.machine_model is null
     or d.machine_model ilike '%'||filter_model||'%'
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
