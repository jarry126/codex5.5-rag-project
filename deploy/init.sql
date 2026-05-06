CREATE EXTENSION IF NOT EXISTS vector;

SELECT format('CREATE DATABASE %I', 'rag_medical')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'rag_medical')\gexec

\connect rag_medical

CREATE EXTENSION IF NOT EXISTS vector;
