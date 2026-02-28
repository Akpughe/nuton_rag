-- Update SELECT policy on courses to allow viewing public courses
DROP POLICY IF EXISTS "Users can view own courses" ON courses;

CREATE POLICY "Users can view own or public courses"
  ON courses FOR SELECT
  USING ((auth.uid() = user_id) OR (visibility = 'public'::text));
