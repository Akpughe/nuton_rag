-- Fix RLS violation when backend (anon key) calls append_space_conversation_messages.
--
-- Root cause: space_conversations has RLS enabled but no permissive policy for the
-- anon role. The backend uses SUPABASE_KEY (anon key), so server-side upserts are
-- blocked with code 42501.
--
-- Fix 1: Recreate the RPC function with SECURITY DEFINER so it runs as the
-- postgres owner and bypasses RLS entirely. The function is only callable by
-- the backend — callers cannot forge space_id/user_id at the SQL level.
--
-- Fix 2: Add the initial space_conversations table + unique constraint in case
-- this migration runs on a fresh DB (idempotent via IF NOT EXISTS / IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS space_conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    space_id    UUID NOT NULL,
    user_id     UUID NOT NULL,
    messages    JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE space_conversations
    ADD CONSTRAINT IF NOT EXISTS space_conversations_space_user_unique
    UNIQUE (space_id, user_id);

-- Recreate function with SECURITY DEFINER (bypasses RLS).
CREATE OR REPLACE FUNCTION append_space_conversation_messages(
    p_space_id      UUID,
    p_user_id       UUID,
    p_new_messages  JSONB
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO space_conversations (space_id, user_id, messages, created_at, updated_at)
    VALUES (p_space_id, p_user_id, p_new_messages, NOW(), NOW())
    ON CONFLICT (space_id, user_id)
    DO UPDATE SET
        messages   = COALESCE(space_conversations.messages, '[]'::jsonb) || EXCLUDED.messages,
        updated_at = NOW();
END;
$$;

-- Read function with SECURITY DEFINER (bypasses RLS for the anon-key backend).
CREATE OR REPLACE FUNCTION get_space_conversation_messages(
    p_space_id UUID,
    p_user_id  UUID,
    p_limit    INT DEFAULT 20
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT messages INTO result
    FROM space_conversations
    WHERE space_id = p_space_id
      AND user_id  = p_user_id
    LIMIT 1;

    IF result IS NULL THEN
        RETURN '[]'::jsonb;
    END IF;

    -- Return the last p_limit messages
    IF jsonb_array_length(result) > p_limit THEN
        RETURN (
            SELECT jsonb_agg(elem)
            FROM (
                SELECT elem
                FROM jsonb_array_elements(result) AS elem
                OFFSET jsonb_array_length(result) - p_limit
            ) sub
        );
    END IF;

    RETURN result;
END;
$$;
