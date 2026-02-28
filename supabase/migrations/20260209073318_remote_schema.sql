drop trigger if exists "update_course_progress_updated_at" on "public"."course_progress";

drop trigger if exists "update_learning_profiles_updated_at" on "public"."learning_profiles";

drop trigger if exists "update_user_settings_updated_at" on "public"."user_settings";

drop trigger if exists "update_writing_sessions_updated_at_trigger" on "public"."writing_sessions";

drop trigger if exists "update_writing_user_sources_updated_at" on "public"."writing_user_sources";

drop policy "Users can view chapters of own courses" on "public"."chapters";

drop policy "Service can insert reflections" on "public"."writing_reflections";

drop policy "Users can view their own reflections" on "public"."writing_reflections";

drop policy "Service can insert section plans" on "public"."writing_section_plans";

drop policy "Service can update section plans" on "public"."writing_section_plans";

drop policy "Users can view their own section plans" on "public"."writing_section_plans";

drop policy "Users can insert sources for their sessions" on "public"."writing_user_sources";

alter table "public"."learning_profiles" drop constraint "learning_profiles_expertise_check";

alter table "public"."chapters" drop constraint "chapters_course_id_fkey";

alter table "public"."course_progress" drop constraint "course_progress_chapter_id_fkey";

alter table "public"."course_progress" drop constraint "course_progress_course_id_fkey";

alter table "public"."course_quiz_attempts" drop constraint "course_quiz_attempts_chapter_id_fkey";

alter table "public"."courses" drop constraint "courses_space_id_fkey";

alter table "public"."flashcard_sets" drop constraint "flashcard_sets_content_id_fkey";

alter table "public"."generated_content" drop constraint "generated_content_audio_id_fkey";

alter table "public"."generated_content" drop constraint "generated_content_pdf_id_fkey";

alter table "public"."generated_content" drop constraint "generated_content_space_id_fkey";

alter table "public"."generated_content" drop constraint "generated_content_yt_id_fkey";

alter table "public"."pdf_chats" drop constraint "pdf_chats_pdf_id_fkey";

alter table "public"."pdf_chats" drop constraint "pdf_chats_user_session_id_fkey";

alter table "public"."pdf_chats" drop constraint "pdf_chats_yt_id_fkey";

alter table "public"."pdf_embeddings" drop constraint "pdf_embeddings_pdf_id_fkey";

alter table "public"."pdf_embeddings" drop constraint "pdf_embeddings_yt_id_fkey";

alter table "public"."pdf_passages" drop constraint "pdf_passages_pdf_id_fkey";

alter table "public"."pdfs" drop constraint "pdfs_space_id_fkey";

alter table "public"."quiz_attempts" drop constraint "quiz_attempts_quiz_set_id_fkey";

alter table "public"."quiz_sets" drop constraint "quiz_sets_content_id_fkey";

alter table "public"."recordings" drop constraint "recordings_space_id_fkey";

alter table "public"."share_link_access" drop constraint "share_link_access_share_link_id_fkey";

alter table "public"."share_links" drop constraint "share_links_space_id_fkey";

alter table "public"."space_shares" drop constraint "space_shares_space_id_fkey";

alter table "public"."subscriptions" drop constraint "subscriptions_plan_id_fkey";

alter table "public"."user_chat_sessions" drop constraint "user_chat_sessions_space_id_fkey";

alter table "public"."user_notes" drop constraint "user_notes_generated_content_id_fkey";

alter table "public"."writing_blocks" drop constraint "writing_blocks_session_id_fkey";

alter table "public"."writing_references" drop constraint "writing_references_session_id_fkey";

alter table "public"."writing_reflections" drop constraint "writing_reflections_session_id_fkey";

alter table "public"."writing_section_plans" drop constraint "writing_section_plans_session_id_fkey";

alter table "public"."writing_structures" drop constraint "writing_structures_session_id_fkey";

alter table "public"."writing_user_sources" drop constraint "writing_user_sources_session_id_fkey";

alter table "public"."yts" drop constraint "yts_space_id_fkey";

drop index if exists "public"."idx_courses_slug";

alter table "public"."conversations" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."courses" drop column "slug";

alter table "public"."document_shares" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."flashcard_sets" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."generated_content" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."learning_profiles" drop column "expertise";

alter table "public"."pdf_chats" alter column "id" set default nextval('public.pdf_chats_id_seq'::regclass);

alter table "public"."pdf_embeddings" alter column "embedding" set data type public.vector(1536) using "embedding"::public.vector(1536);

alter table "public"."pdf_embeddings" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."pdf_passages" alter column "embedding" set data type public.vector(1024) using "embedding"::public.vector(1024);

alter table "public"."pdf_passages" alter column "id" set default nextval('public.pdf_passages_id_seq'::regclass);

alter table "public"."pdfs" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."plans" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."quiz_attempts" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."quiz_sets" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."recordings" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."referral" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."referral_codes" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."reviews" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."share_audit_logs" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."share_link_access" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."share_links" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."space_shares" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."spaces" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."subscriptions" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."subscriptions" alter column "subscription_status" set data type public.subscription_status_type using "subscription_status"::text::public.subscription_status_type;

alter table "public"."user_chat_sessions" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."user_list" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."user_notes" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."user_settings" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."user_settings" alter column "preferred_learning_style" set data type public.learning_style_type using "preferred_learning_style"::text::public.learning_style_type;

alter table "public"."yts" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."chapters" add constraint "chapters_course_id_fkey" FOREIGN KEY (course_id) REFERENCES public.courses(id) ON DELETE CASCADE not valid;

alter table "public"."chapters" validate constraint "chapters_course_id_fkey";

alter table "public"."course_progress" add constraint "course_progress_chapter_id_fkey" FOREIGN KEY (chapter_id) REFERENCES public.chapters(id) ON DELETE CASCADE not valid;

alter table "public"."course_progress" validate constraint "course_progress_chapter_id_fkey";

alter table "public"."course_progress" add constraint "course_progress_course_id_fkey" FOREIGN KEY (course_id) REFERENCES public.courses(id) ON DELETE CASCADE not valid;

alter table "public"."course_progress" validate constraint "course_progress_course_id_fkey";

alter table "public"."course_quiz_attempts" add constraint "course_quiz_attempts_chapter_id_fkey" FOREIGN KEY (chapter_id) REFERENCES public.chapters(id) ON DELETE CASCADE not valid;

alter table "public"."course_quiz_attempts" validate constraint "course_quiz_attempts_chapter_id_fkey";

alter table "public"."courses" add constraint "courses_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE SET NULL not valid;

alter table "public"."courses" validate constraint "courses_space_id_fkey";

alter table "public"."flashcard_sets" add constraint "flashcard_sets_content_id_fkey" FOREIGN KEY (content_id) REFERENCES public.generated_content(id) not valid;

alter table "public"."flashcard_sets" validate constraint "flashcard_sets_content_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_audio_id_fkey" FOREIGN KEY (audio_id) REFERENCES public.recordings(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_audio_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_pdf_id_fkey" FOREIGN KEY (pdf_id) REFERENCES public.pdfs(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_pdf_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_space_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_yt_id_fkey" FOREIGN KEY (yt_id) REFERENCES public.yts(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_yt_id_fkey";

alter table "public"."pdf_chats" add constraint "pdf_chats_pdf_id_fkey" FOREIGN KEY (pdf_id) REFERENCES public.pdfs(id) ON DELETE CASCADE not valid;

alter table "public"."pdf_chats" validate constraint "pdf_chats_pdf_id_fkey";

alter table "public"."pdf_chats" add constraint "pdf_chats_user_session_id_fkey" FOREIGN KEY (user_session_id) REFERENCES public.user_chat_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."pdf_chats" validate constraint "pdf_chats_user_session_id_fkey";

alter table "public"."pdf_chats" add constraint "pdf_chats_yt_id_fkey" FOREIGN KEY (yt_id) REFERENCES public.yts(id) ON DELETE CASCADE not valid;

alter table "public"."pdf_chats" validate constraint "pdf_chats_yt_id_fkey";

alter table "public"."pdf_embeddings" add constraint "pdf_embeddings_pdf_id_fkey" FOREIGN KEY (pdf_id) REFERENCES public.pdfs(id) ON DELETE CASCADE not valid;

alter table "public"."pdf_embeddings" validate constraint "pdf_embeddings_pdf_id_fkey";

alter table "public"."pdf_embeddings" add constraint "pdf_embeddings_yt_id_fkey" FOREIGN KEY (yt_id) REFERENCES public.yts(id) ON DELETE CASCADE not valid;

alter table "public"."pdf_embeddings" validate constraint "pdf_embeddings_yt_id_fkey";

alter table "public"."pdf_passages" add constraint "pdf_passages_pdf_id_fkey" FOREIGN KEY (pdf_id) REFERENCES public.pdfs(id) not valid;

alter table "public"."pdf_passages" validate constraint "pdf_passages_pdf_id_fkey";

alter table "public"."pdfs" add constraint "pdfs_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."pdfs" validate constraint "pdfs_space_id_fkey";

alter table "public"."quiz_attempts" add constraint "quiz_attempts_quiz_set_id_fkey" FOREIGN KEY (quiz_set_id) REFERENCES public.quiz_sets(id) not valid;

alter table "public"."quiz_attempts" validate constraint "quiz_attempts_quiz_set_id_fkey";

alter table "public"."quiz_sets" add constraint "quiz_sets_content_id_fkey" FOREIGN KEY (content_id) REFERENCES public.generated_content(id) not valid;

alter table "public"."quiz_sets" validate constraint "quiz_sets_content_id_fkey";

alter table "public"."recordings" add constraint "recordings_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."recordings" validate constraint "recordings_space_id_fkey";

alter table "public"."share_link_access" add constraint "share_link_access_share_link_id_fkey" FOREIGN KEY (share_link_id) REFERENCES public.share_links(id) ON DELETE CASCADE not valid;

alter table "public"."share_link_access" validate constraint "share_link_access_share_link_id_fkey";

alter table "public"."share_links" add constraint "share_links_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."share_links" validate constraint "share_links_space_id_fkey";

alter table "public"."space_shares" add constraint "space_shares_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."space_shares" validate constraint "space_shares_space_id_fkey";

alter table "public"."subscriptions" add constraint "subscriptions_plan_id_fkey" FOREIGN KEY (plan_id) REFERENCES public.plans(id) not valid;

alter table "public"."subscriptions" validate constraint "subscriptions_plan_id_fkey";

alter table "public"."user_chat_sessions" add constraint "user_chat_sessions_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."user_chat_sessions" validate constraint "user_chat_sessions_space_id_fkey";

alter table "public"."user_notes" add constraint "user_notes_generated_content_id_fkey" FOREIGN KEY (generated_content_id) REFERENCES public.generated_content(id) ON DELETE CASCADE not valid;

alter table "public"."user_notes" validate constraint "user_notes_generated_content_id_fkey";

alter table "public"."writing_blocks" add constraint "writing_blocks_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_blocks" validate constraint "writing_blocks_session_id_fkey";

alter table "public"."writing_references" add constraint "writing_references_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_references" validate constraint "writing_references_session_id_fkey";

alter table "public"."writing_reflections" add constraint "writing_reflections_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_reflections" validate constraint "writing_reflections_session_id_fkey";

alter table "public"."writing_section_plans" add constraint "writing_section_plans_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_section_plans" validate constraint "writing_section_plans_session_id_fkey";

alter table "public"."writing_structures" add constraint "writing_structures_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_structures" validate constraint "writing_structures_session_id_fkey";

alter table "public"."writing_user_sources" add constraint "writing_user_sources_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_user_sources" validate constraint "writing_user_sources_session_id_fkey";

alter table "public"."yts" add constraint "yts_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."yts" validate constraint "yts_space_id_fkey";

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.get_active_section_plan(p_session_id uuid, p_section_id text)
 RETURNS public.writing_section_plans
 LANGUAGE sql
 STABLE
AS $function$
  SELECT *
  FROM writing_section_plans
  WHERE session_id = p_session_id
    AND section_id = p_section_id
    AND is_active = true
  ORDER BY created_at DESC
  LIMIT 1;
$function$
;

CREATE OR REPLACE FUNCTION public.get_latest_reflection(p_session_id uuid, p_section_id text)
 RETURNS public.writing_reflections
 LANGUAGE sql
 STABLE
AS $function$
  SELECT *
  FROM writing_reflections
  WHERE session_id = p_session_id
    AND section_id = p_section_id
  ORDER BY created_at DESC
  LIMIT 1;
$function$
;


  create policy "Backend can insert chapters"
  on "public"."chapters"
  as permissive
  for insert
  to anon
with check (true);



  create policy "Backend can select chapters"
  on "public"."chapters"
  as permissive
  for select
  to anon
using (true);



  create policy "Backend can update chapters"
  on "public"."chapters"
  as permissive
  for update
  to anon
using (true);



  create policy "Backend can insert course_progress"
  on "public"."course_progress"
  as permissive
  for insert
  to anon
with check (true);



  create policy "Backend can select course_progress"
  on "public"."course_progress"
  as permissive
  for select
  to anon
using (true);



  create policy "Backend can update course_progress"
  on "public"."course_progress"
  as permissive
  for update
  to anon
using (true);



  create policy "Backend can insert course_quiz_attempts"
  on "public"."course_quiz_attempts"
  as permissive
  for insert
  to anon
with check (true);



  create policy "Backend can select course_quiz_attempts"
  on "public"."course_quiz_attempts"
  as permissive
  for select
  to anon
using (true);



  create policy "Backend can insert courses"
  on "public"."courses"
  as permissive
  for insert
  to anon
with check (true);



  create policy "Backend can select courses"
  on "public"."courses"
  as permissive
  for select
  to anon
using (true);



  create policy "Backend can update courses"
  on "public"."courses"
  as permissive
  for update
  to anon
using (true);



  create policy "Backend can insert learning_profiles"
  on "public"."learning_profiles"
  as permissive
  for insert
  to anon
with check (true);



  create policy "Backend can select learning_profiles"
  on "public"."learning_profiles"
  as permissive
  for select
  to anon
using (true);



  create policy "Backend can update learning_profiles"
  on "public"."learning_profiles"
  as permissive
  for update
  to anon
using (true);



  create policy "Users can view chapters of own courses"
  on "public"."chapters"
  as permissive
  for select
  to public
using ((EXISTS ( SELECT 1
   FROM public.courses
  WHERE ((courses.id = chapters.course_id) AND (courses.user_id = auth.uid())))));



  create policy "Service can insert reflections"
  on "public"."writing_reflections"
  as permissive
  for insert
  to authenticated
with check ((session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid()))));



  create policy "Users can view their own reflections"
  on "public"."writing_reflections"
  as permissive
  for select
  to authenticated
using ((session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid()))));



  create policy "Service can insert section plans"
  on "public"."writing_section_plans"
  as permissive
  for insert
  to authenticated
with check ((session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid()))));



  create policy "Service can update section plans"
  on "public"."writing_section_plans"
  as permissive
  for update
  to authenticated
using ((session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid()))));



  create policy "Users can view their own section plans"
  on "public"."writing_section_plans"
  as permissive
  for select
  to authenticated
using ((session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid()))));



  create policy "Users can insert sources for their sessions"
  on "public"."writing_user_sources"
  as permissive
  for insert
  to authenticated
with check (((user_id = auth.uid()) AND (session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid())))));


CREATE TRIGGER update_course_progress_updated_at BEFORE UPDATE ON public.course_progress FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_learning_profiles_updated_at BEFORE UPDATE ON public.learning_profiles FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON public.user_settings FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_writing_sessions_updated_at_trigger BEFORE UPDATE ON public.writing_sessions FOR EACH ROW EXECUTE FUNCTION public.update_writing_sessions_updated_at();

CREATE TRIGGER update_writing_user_sources_updated_at BEFORE UPDATE ON public.writing_user_sources FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER objects_delete_delete_prefix AFTER DELETE ON storage.objects FOR EACH ROW EXECUTE FUNCTION storage.delete_prefix_hierarchy_trigger();

CREATE TRIGGER objects_insert_create_prefix BEFORE INSERT ON storage.objects FOR EACH ROW EXECUTE FUNCTION storage.objects_insert_prefix_trigger();

CREATE TRIGGER objects_update_create_prefix BEFORE UPDATE ON storage.objects FOR EACH ROW WHEN (((new.name <> old.name) OR (new.bucket_id <> old.bucket_id))) EXECUTE FUNCTION storage.objects_update_prefix_trigger();

CREATE TRIGGER prefixes_create_hierarchy BEFORE INSERT ON storage.prefixes FOR EACH ROW WHEN ((pg_trigger_depth() < 1)) EXECUTE FUNCTION storage.prefixes_insert_trigger();

CREATE TRIGGER prefixes_delete_hierarchy AFTER DELETE ON storage.prefixes FOR EACH ROW EXECUTE FUNCTION storage.delete_prefix_hierarchy_trigger();


