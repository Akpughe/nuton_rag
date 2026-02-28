-- Add summary_md column and widen status check to include 'materials_only'
ALTER TABLE courses ADD COLUMN IF NOT EXISTS summary_md text;

ALTER TABLE courses DROP CONSTRAINT IF EXISTS courses_status_check;
ALTER TABLE courses ADD CONSTRAINT courses_status_check
  CHECK (status = ANY (ARRAY['generating'::text, 'ready'::text, 'error'::text, 'materials_only'::text]));
