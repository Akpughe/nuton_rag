-- Add flashcards JSONB column to chapters table (per-chapter flashcards)
ALTER TABLE chapters ADD COLUMN IF NOT EXISTS flashcards jsonb DEFAULT '[]'::jsonb;
