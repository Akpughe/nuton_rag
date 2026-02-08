-- Migration: Add course generation tables
-- 5 new tables: learning_profiles, courses, chapters, course_progress, course_quiz_attempts
-- These are purely additive — no existing tables are modified.

--------------------------------------------------------------------------------
-- 1. learning_profiles
--------------------------------------------------------------------------------
create table "public"."learning_profiles" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "format_pref" text not null,
    "depth_pref" text not null,
    "role" text not null,
    "learning_goal" text not null,
    "example_pref" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    constraint "learning_profiles_pkey" primary key ("id"),
    constraint "learning_profiles_user_id_key" unique ("user_id"),
    constraint "learning_profiles_user_id_fkey" foreign key ("user_id") references auth.users("id") on delete cascade,
    constraint "learning_profiles_format_pref_check" check (format_pref in ('reading', 'listening', 'testing', 'mixed')),
    constraint "learning_profiles_depth_pref_check" check (depth_pref in ('quick', 'detailed', 'conversational', 'academic')),
    constraint "learning_profiles_role_check" check (role in ('student', 'professional', 'graduate_student')),
    constraint "learning_profiles_learning_goal_check" check (learning_goal in ('exams', 'career', 'curiosity', 'supplement')),
    constraint "learning_profiles_example_pref_check" check (example_pref in ('real_world', 'technical', 'stories', 'analogies'))
);

--------------------------------------------------------------------------------
-- 2. courses
--------------------------------------------------------------------------------
create table "public"."courses" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "space_id" uuid,
    "title" text not null,
    "description" text,
    "topic" text not null,
    "source_type" text not null,
    "source_files" jsonb default '[]'::jsonb,
    "multi_file_organization" text,
    "total_chapters" integer not null,
    "estimated_time" integer,
    "status" text not null default 'generating',
    "personalization_params" jsonb,
    "outline" jsonb,
    "model_used" text,
    "created_at" timestamp with time zone default now(),
    "completed_at" timestamp with time zone,
    constraint "courses_pkey" primary key ("id"),
    constraint "courses_user_id_fkey" foreign key ("user_id") references auth.users("id") on delete cascade,
    constraint "courses_space_id_fkey" foreign key ("space_id") references "public"."spaces"("id") on delete set null,
    constraint "courses_source_type_check" check (source_type in ('topic', 'files', 'youtube', 'web', 'mixed')),
    constraint "courses_multi_file_organization_check" check (multi_file_organization in ('thematic_bridge', 'sequential_sections', 'separate_courses')),
    constraint "courses_status_check" check (status in ('generating', 'ready', 'error'))
);

--------------------------------------------------------------------------------
-- 3. chapters
--------------------------------------------------------------------------------
create table "public"."chapters" (
    "id" uuid not null default gen_random_uuid(),
    "course_id" uuid not null,
    "order_index" integer not null,
    "title" text not null,
    "learning_objectives" jsonb default '[]'::jsonb,
    "content" text,
    "content_format" text default 'markdown',
    "estimated_time" integer,
    "key_concepts" jsonb default '[]'::jsonb,
    "sources" jsonb default '[]'::jsonb,
    "quiz" jsonb,
    "word_count" integer default 0,
    "source_document_id" uuid,
    "source_document_type" text,
    "status" text not null default 'pending',
    "generated_at" timestamp with time zone,
    "created_at" timestamp with time zone default now(),
    constraint "chapters_pkey" primary key ("id"),
    constraint "chapters_course_id_order_index_key" unique ("course_id", "order_index"),
    constraint "chapters_course_id_fkey" foreign key ("course_id") references "public"."courses"("id") on delete cascade,
    constraint "chapters_source_document_type_check" check (source_document_type in ('pdf', 'youtube')),
    constraint "chapters_status_check" check (status in ('pending', 'ready', 'error'))
);

--------------------------------------------------------------------------------
-- 4. course_progress
--------------------------------------------------------------------------------
create table "public"."course_progress" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "course_id" uuid not null,
    "chapter_id" uuid not null,
    "completed" boolean default false,
    "time_spent_minutes" integer default 0,
    "completed_at" timestamp with time zone,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    constraint "course_progress_pkey" primary key ("id"),
    constraint "course_progress_user_id_chapter_id_key" unique ("user_id", "chapter_id"),
    constraint "course_progress_user_id_fkey" foreign key ("user_id") references auth.users("id") on delete cascade,
    constraint "course_progress_course_id_fkey" foreign key ("course_id") references "public"."courses"("id") on delete cascade,
    constraint "course_progress_chapter_id_fkey" foreign key ("chapter_id") references "public"."chapters"("id") on delete cascade
);

--------------------------------------------------------------------------------
-- 5. course_quiz_attempts
--------------------------------------------------------------------------------
create table "public"."course_quiz_attempts" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "chapter_id" uuid not null,
    "score" numeric(5,2),
    "answers" jsonb,
    "started_at" timestamp with time zone default now(),
    "completed_at" timestamp with time zone,
    "time_taken_seconds" integer,
    constraint "course_quiz_attempts_pkey" primary key ("id"),
    constraint "course_quiz_attempts_user_chapter_started_key" unique ("user_id", "chapter_id", "started_at"),
    constraint "course_quiz_attempts_user_id_fkey" foreign key ("user_id") references auth.users("id") on delete cascade,
    constraint "course_quiz_attempts_chapter_id_fkey" foreign key ("chapter_id") references "public"."chapters"("id") on delete cascade
);

--------------------------------------------------------------------------------
-- Indexes
--------------------------------------------------------------------------------
create index idx_courses_user_id on "public"."courses" ("user_id");
create index idx_courses_space_id on "public"."courses" ("space_id");
create index idx_courses_status on "public"."courses" ("status");
create index idx_chapters_course_id on "public"."chapters" ("course_id");
create index idx_course_progress_user_id on "public"."course_progress" ("user_id");
create index idx_course_progress_course_id on "public"."course_progress" ("course_id");
create index idx_course_quiz_attempts_user_id on "public"."course_quiz_attempts" ("user_id");
create index idx_course_quiz_attempts_chapter_id on "public"."course_quiz_attempts" ("chapter_id");
create index idx_learning_profiles_user_id on "public"."learning_profiles" ("user_id");

--------------------------------------------------------------------------------
-- Row-Level Security
--------------------------------------------------------------------------------
alter table "public"."learning_profiles" enable row level security;
alter table "public"."courses" enable row level security;
alter table "public"."chapters" enable row level security;
alter table "public"."course_progress" enable row level security;
alter table "public"."course_quiz_attempts" enable row level security;

-- learning_profiles: users manage their own
create policy "Users can view own profile"
    on "public"."learning_profiles"
    as permissive
    for select
    to public
    using ((auth.uid() = user_id));

create policy "Users can insert own profile"
    on "public"."learning_profiles"
    as permissive
    for insert
    to public
    with check ((auth.uid() = user_id));

create policy "Users can update own profile"
    on "public"."learning_profiles"
    as permissive
    for update
    to public
    using ((auth.uid() = user_id));

-- courses: users manage their own
create policy "Users can view own courses"
    on "public"."courses"
    as permissive
    for select
    to public
    using ((auth.uid() = user_id));

create policy "Users can insert own courses"
    on "public"."courses"
    as permissive
    for insert
    to public
    with check ((auth.uid() = user_id));

create policy "Users can update own courses"
    on "public"."courses"
    as permissive
    for update
    to public
    using ((auth.uid() = user_id));

create policy "Users can delete own courses"
    on "public"."courses"
    as permissive
    for delete
    to public
    using ((auth.uid() = user_id));

-- chapters: visible if user owns the parent course
create policy "Users can view chapters of own courses"
    on "public"."chapters"
    as permissive
    for select
    to public
    using ((exists (select 1 from courses where courses.id = chapters.course_id and courses.user_id = auth.uid())));

create policy "Service can manage chapters"
    on "public"."chapters"
    as permissive
    for all
    to service_role
    using (true);

-- course_progress: users manage their own
create policy "Users can view own progress"
    on "public"."course_progress"
    as permissive
    for select
    to public
    using ((auth.uid() = user_id));

create policy "Users can insert own progress"
    on "public"."course_progress"
    as permissive
    for insert
    to public
    with check ((auth.uid() = user_id));

create policy "Users can update own progress"
    on "public"."course_progress"
    as permissive
    for update
    to public
    using ((auth.uid() = user_id));

-- course_quiz_attempts: users manage their own
create policy "Users can view own quiz attempts"
    on "public"."course_quiz_attempts"
    as permissive
    for select
    to public
    using ((auth.uid() = user_id));

create policy "Users can insert own quiz attempts"
    on "public"."course_quiz_attempts"
    as permissive
    for insert
    to public
    with check ((auth.uid() = user_id));

--------------------------------------------------------------------------------
-- Triggers (reuse existing update_updated_at_column function)
--------------------------------------------------------------------------------
create trigger update_learning_profiles_updated_at
    before update on "public"."learning_profiles"
    for each row execute function public.update_updated_at_column();

create trigger update_course_progress_updated_at
    before update on "public"."course_progress"
    for each row execute function public.update_updated_at_column();

--------------------------------------------------------------------------------
-- Grants (follow existing per-operation pattern)
--------------------------------------------------------------------------------
grant delete on table "public"."learning_profiles" to "anon";
grant insert on table "public"."learning_profiles" to "anon";
grant references on table "public"."learning_profiles" to "anon";
grant select on table "public"."learning_profiles" to "anon";
grant trigger on table "public"."learning_profiles" to "anon";
grant truncate on table "public"."learning_profiles" to "anon";
grant update on table "public"."learning_profiles" to "anon";
grant delete on table "public"."learning_profiles" to "authenticated";
grant insert on table "public"."learning_profiles" to "authenticated";
grant references on table "public"."learning_profiles" to "authenticated";
grant select on table "public"."learning_profiles" to "authenticated";
grant trigger on table "public"."learning_profiles" to "authenticated";
grant truncate on table "public"."learning_profiles" to "authenticated";
grant update on table "public"."learning_profiles" to "authenticated";
grant delete on table "public"."learning_profiles" to "service_role";
grant insert on table "public"."learning_profiles" to "service_role";
grant references on table "public"."learning_profiles" to "service_role";
grant select on table "public"."learning_profiles" to "service_role";
grant trigger on table "public"."learning_profiles" to "service_role";
grant truncate on table "public"."learning_profiles" to "service_role";
grant update on table "public"."learning_profiles" to "service_role";

grant delete on table "public"."courses" to "anon";
grant insert on table "public"."courses" to "anon";
grant references on table "public"."courses" to "anon";
grant select on table "public"."courses" to "anon";
grant trigger on table "public"."courses" to "anon";
grant truncate on table "public"."courses" to "anon";
grant update on table "public"."courses" to "anon";
grant delete on table "public"."courses" to "authenticated";
grant insert on table "public"."courses" to "authenticated";
grant references on table "public"."courses" to "authenticated";
grant select on table "public"."courses" to "authenticated";
grant trigger on table "public"."courses" to "authenticated";
grant truncate on table "public"."courses" to "authenticated";
grant update on table "public"."courses" to "authenticated";
grant delete on table "public"."courses" to "service_role";
grant insert on table "public"."courses" to "service_role";
grant references on table "public"."courses" to "service_role";
grant select on table "public"."courses" to "service_role";
grant trigger on table "public"."courses" to "service_role";
grant truncate on table "public"."courses" to "service_role";
grant update on table "public"."courses" to "service_role";

grant delete on table "public"."chapters" to "anon";
grant insert on table "public"."chapters" to "anon";
grant references on table "public"."chapters" to "anon";
grant select on table "public"."chapters" to "anon";
grant trigger on table "public"."chapters" to "anon";
grant truncate on table "public"."chapters" to "anon";
grant update on table "public"."chapters" to "anon";
grant delete on table "public"."chapters" to "authenticated";
grant insert on table "public"."chapters" to "authenticated";
grant references on table "public"."chapters" to "authenticated";
grant select on table "public"."chapters" to "authenticated";
grant trigger on table "public"."chapters" to "authenticated";
grant truncate on table "public"."chapters" to "authenticated";
grant update on table "public"."chapters" to "authenticated";
grant delete on table "public"."chapters" to "service_role";
grant insert on table "public"."chapters" to "service_role";
grant references on table "public"."chapters" to "service_role";
grant select on table "public"."chapters" to "service_role";
grant trigger on table "public"."chapters" to "service_role";
grant truncate on table "public"."chapters" to "service_role";
grant update on table "public"."chapters" to "service_role";

grant delete on table "public"."course_progress" to "anon";
grant insert on table "public"."course_progress" to "anon";
grant references on table "public"."course_progress" to "anon";
grant select on table "public"."course_progress" to "anon";
grant trigger on table "public"."course_progress" to "anon";
grant truncate on table "public"."course_progress" to "anon";
grant update on table "public"."course_progress" to "anon";
grant delete on table "public"."course_progress" to "authenticated";
grant insert on table "public"."course_progress" to "authenticated";
grant references on table "public"."course_progress" to "authenticated";
grant select on table "public"."course_progress" to "authenticated";
grant trigger on table "public"."course_progress" to "authenticated";
grant truncate on table "public"."course_progress" to "authenticated";
grant update on table "public"."course_progress" to "authenticated";
grant delete on table "public"."course_progress" to "service_role";
grant insert on table "public"."course_progress" to "service_role";
grant references on table "public"."course_progress" to "service_role";
grant select on table "public"."course_progress" to "service_role";
grant trigger on table "public"."course_progress" to "service_role";
grant truncate on table "public"."course_progress" to "service_role";
grant update on table "public"."course_progress" to "service_role";

grant delete on table "public"."course_quiz_attempts" to "anon";
grant insert on table "public"."course_quiz_attempts" to "anon";
grant references on table "public"."course_quiz_attempts" to "anon";
grant select on table "public"."course_quiz_attempts" to "anon";
grant trigger on table "public"."course_quiz_attempts" to "anon";
grant truncate on table "public"."course_quiz_attempts" to "anon";
grant update on table "public"."course_quiz_attempts" to "anon";
grant delete on table "public"."course_quiz_attempts" to "authenticated";
grant insert on table "public"."course_quiz_attempts" to "authenticated";
grant references on table "public"."course_quiz_attempts" to "authenticated";
grant select on table "public"."course_quiz_attempts" to "authenticated";
grant trigger on table "public"."course_quiz_attempts" to "authenticated";
grant truncate on table "public"."course_quiz_attempts" to "authenticated";
grant update on table "public"."course_quiz_attempts" to "authenticated";
grant delete on table "public"."course_quiz_attempts" to "service_role";
grant insert on table "public"."course_quiz_attempts" to "service_role";
grant references on table "public"."course_quiz_attempts" to "service_role";
grant select on table "public"."course_quiz_attempts" to "service_role";
grant trigger on table "public"."course_quiz_attempts" to "service_role";
grant truncate on table "public"."course_quiz_attempts" to "service_role";
grant update on table "public"."course_quiz_attempts" to "service_role";
