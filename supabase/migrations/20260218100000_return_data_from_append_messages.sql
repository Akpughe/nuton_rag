-- Change append_space_conversation_messages to return the updated row as JSONB
-- so the backend can log exactly what was persisted.
-- Must DROP first because Postgres cannot change return type via CREATE OR REPLACE.

DROP FUNCTION IF EXISTS append_space_conversation_messages(UUID, UUID, JSONB);

CREATE OR REPLACE FUNCTION append_space_conversation_messages(
    p_space_id      UUID,
    p_user_id       UUID,
    p_new_messages  JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    result JSONB;
BEGIN
    INSERT INTO space_conversations (space_id, user_id, messages, created_at, updated_at)
    VALUES (p_space_id, p_user_id, p_new_messages, NOW(), NOW())
    ON CONFLICT (space_id, user_id)
    DO UPDATE SET
        messages   = COALESCE(space_conversations.messages, '[]'::jsonb) || EXCLUDED.messages,
        updated_at = NOW()
    RETURNING jsonb_build_object(
        'id',         id,
        'space_id',   space_id,
        'user_id',    user_id,
        'messages',   messages,
        'created_at', created_at,
        'updated_at', updated_at
    ) INTO result;

    RETURN result;
END;
$$;
