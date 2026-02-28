-- Add course_id column to flashcard_sets and quiz_sets
ALTER TABLE flashcard_sets ADD COLUMN IF NOT EXISTS course_id uuid;
ALTER TABLE flashcard_sets ADD CONSTRAINT flashcard_sets_course_id_fkey
  FOREIGN KEY (course_id) REFERENCES courses(id);

ALTER TABLE quiz_sets ADD COLUMN IF NOT EXISTS course_id uuid;
ALTER TABLE quiz_sets ADD CONSTRAINT quiz_sets_course_id_fkey
  FOREIGN KEY (course_id) REFERENCES courses(id);
