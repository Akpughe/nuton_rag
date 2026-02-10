-- Create course_exam_attempts table for graded exam submissions
CREATE TABLE IF NOT EXISTS course_exam_attempts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id uuid NOT NULL REFERENCES course_exams(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    answers jsonb NOT NULL,
    results jsonb NOT NULL,
    score numeric(5,2) NOT NULL,
    mcq_score numeric(5,2),
    fill_in_gap_score numeric(5,2),
    theory_score numeric(5,2),
    time_taken_seconds integer,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_course_exam_attempts_lookup
    ON course_exam_attempts (exam_id, user_id);
