-- Add study_guide and flashcards JSONB columns to courses table
ALTER TABLE courses ADD COLUMN IF NOT EXISTS study_guide jsonb;
ALTER TABLE courses ADD COLUMN IF NOT EXISTS flashcards jsonb;

-- Create course_exams table for on-demand final exams
CREATE TABLE IF NOT EXISTS course_exams (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id uuid NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    exam_size integer NOT NULL DEFAULT 30,
    mcq jsonb NOT NULL DEFAULT '[]'::jsonb,
    fill_in_gap jsonb NOT NULL DEFAULT '[]'::jsonb,
    theory jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_course_exams_course_user
    ON course_exams (course_id, user_id);

-- Create course_chat_messages table for persistent chat history
CREATE TABLE IF NOT EXISTS course_chat_messages (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id uuid NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    sources jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_course_chat_messages_lookup
    ON course_chat_messages (course_id, user_id, created_at DESC);
