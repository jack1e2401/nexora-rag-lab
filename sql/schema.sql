CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    board_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
    overlap INTEGER NOT NULL CHECK (overlap >= 0 AND overlap < chunk_size),
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions = 1024),
    embedding_settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    corpus_fingerprint TEXT NOT NULL,
    extraction_version TEXT NOT NULL DEFAULT 'tika-text-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL CHECK (end_char > start_char),
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (profile_id, document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_profile_document_idx
    ON chunks (profile_id, document_id);

ALTER TABLE profiles ADD COLUMN IF NOT EXISTS extraction_version TEXT NOT NULL DEFAULT 'tika-text-v1';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_char INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_char INTEGER;
