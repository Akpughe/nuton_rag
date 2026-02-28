-- Add visibility column to courses table
ALTER TABLE courses ADD COLUMN IF NOT EXISTS visibility text NOT NULL DEFAULT 'private'::text;

ALTER TABLE courses ADD CONSTRAINT courses_visibility_check
  CHECK (visibility = ANY (ARRAY['private'::text, 'public'::text]));

CREATE INDEX IF NOT EXISTS idx_courses_visibility ON courses (visibility);
