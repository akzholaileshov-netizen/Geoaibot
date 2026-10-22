-- GeoAI textbook search. Leaves the VIP users table unchanged.
-- gemini-embedding-001 vectors from unt_geography_v2 are 3072 dimensions.
-- vector(3072) can be stored but not indexed, so search casts to halfvec.

create extension if not exists vector with schema extensions;

create table if not exists public.textbook_chunks (
    id text primary key,
    content text not null,
    source text,
    page integer,
    embedding extensions.vector(3072) not null
);

alter table public.textbook_chunks enable row level security;

create index if not exists textbook_chunks_embedding_hnsw
on public.textbook_chunks
using hnsw ((embedding::extensions.halfvec(3072)) extensions.halfvec_cosine_ops);

create or replace function public.match_textbook_chunks(
    query_embedding extensions.vector(3072),
    match_count integer
)
returns table (
    id text,
    content text,
    source text,
    page integer,
    similarity double precision
)
language sql
stable
set search_path = public, extensions
as $$
    select
        textbook_chunks.id,
        textbook_chunks.content,
        textbook_chunks.source,
        textbook_chunks.page,
        1 - ((textbook_chunks.embedding::extensions.halfvec(3072)) <=> (query_embedding::extensions.halfvec(3072))) as similarity
    from public.textbook_chunks
    order by (textbook_chunks.embedding::extensions.halfvec(3072)) <=> (query_embedding::extensions.halfvec(3072))
    limit greatest(coalesce(match_count, 1), 1);
$$;

grant select, insert, update, delete on public.textbook_chunks to service_role;
grant execute on function public.match_textbook_chunks(extensions.vector, integer) to service_role;
