create extension if not exists "pg_cron" with schema "pg_catalog";

drop extension if exists "pg_net";

create extension if not exists "vector" with schema "public";

create type "public"."learning_style_type" as enum ('academic_focus', 'deep_dive', 'quick_practical', 'exploratory_curious', 'narrative_reader');

create type "public"."subscription_status_type" as enum ('active', 'cancelled', 'expired', 'pending', 'failed', 'cancelling');

create sequence "public"."pdf_chats_id_seq";

create sequence "public"."pdf_passages_id_seq";


  create table "public"."chat_library" (
    "id" uuid not null,
    "conversation_id" uuid not null,
    "role" text not null,
    "content" text not null,
    "model" text,
    "created_at" timestamp with time zone default now(),
    "metadata" jsonb,
    "group" text,
    "tool_calls" jsonb[],
    "user_id" text not null
      );



  create table "public"."conversations" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "title" text not null,
    "number_of_messages" integer not null default 0,
    "number_of_tokens" integer not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "conversation_id" uuid
      );



  create table "public"."document_shares" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "document_id" uuid not null,
    "document_type" text not null,
    "owner_id" uuid not null,
    "shared_with_user_id" uuid not null,
    "permission_level" text default 'viewer'::text,
    "created_at" timestamp with time zone default now()
      );



  create table "public"."flashcard_sets" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "content_id" uuid,
    "flashcards" jsonb[],
    "created_at" timestamp with time zone default now(),
    "set_number" integer not null,
    "created_by" uuid,
    "is_shared" boolean default false
      );



  create table "public"."generated_content" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "pdf_id" uuid,
    "space_id" uuid,
    "summary" text,
    "flashcards" jsonb[],
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "yt_id" uuid,
    "quiz" jsonb[],
    "audio_id" uuid,
    "new_note" text,
    "content_hash" text
      );



  create table "public"."google_auth_tokens" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" character varying(255) not null,
    "space_id" character varying(255) not null,
    "encrypted_access_token" text not null,
    "encrypted_refresh_token" text not null,
    "token_expires_at" timestamp with time zone,
    "scopes" text[] not null default ARRAY['https://www.googleapis.com/auth/drive.readonly'::text, 'https://www.googleapis.com/auth/documents.readonly'::text],
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
      );



  create table "public"."memories" (
    "id" uuid not null,
    "user_id" text not null,
    "content" text not null,
    "created_at" timestamp with time zone default now(),
    "metadata" jsonb,
    "content_search" tsvector generated always as (to_tsvector('english'::regconfig, content)) stored
      );



  create table "public"."pdf_chats" (
    "id" integer not null default nextval('public.pdf_chats_id_seq'::regclass),
    "user_id" uuid not null,
    "role" text not null,
    "query" text not null,
    "response" text not null,
    "timestamp" timestamp without time zone default CURRENT_TIMESTAMP,
    "space_id" uuid,
    "refs" jsonb,
    "pdf_id" uuid,
    "yt_id" uuid,
    "chat_context" text,
    "conversation_pair_id" uuid,
    "user_session_id" uuid,
    "is_shared" boolean default false
      );



  create table "public"."pdf_embeddings" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "pdf_id" uuid,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "yt_id" uuid,
    "embedding" public.vector(1536)
      );



  create table "public"."pdf_passages" (
    "id" integer not null default nextval('public.pdf_passages_id_seq'::regclass),
    "pdf_id" uuid,
    "passage_text" text,
    "embedding" public.vector(1024)
      );



  create table "public"."pdfs" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "space_id" uuid,
    "file_path" text not null,
    "extracted_text" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "file_type" text not null,
    "file_name" text,
    "source" character varying(50) default 'upload'::character varying,
    "drive_file_id" character varying(255),
    "mime_type" character varying(100),
    "file_size" bigint,
    "external_created_at" timestamp with time zone,
    "external_modified_at" timestamp with time zone,
    "visibility" text default 'private'::text,
    "uploaded_by" uuid
      );



  create table "public"."plans" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "name" text not null,
    "description" text,
    "monthly_price" numeric(10,2),
    "yearly_price" numeric(10,2),
    "features" jsonb,
    "highlighted" boolean default false,
    "disabled" boolean default false,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "monthly_price_ngn" numeric(10,2),
    "yearly_price_ngn" numeric(10,2)
      );



  create table "public"."quiz_attempts" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "quiz_set_id" uuid,
    "user_id" uuid,
    "score" numeric(5,2),
    "answers" jsonb,
    "started_at" timestamp with time zone default now(),
    "completed_at" timestamp with time zone,
    "time_taken_seconds" integer
      );



  create table "public"."quiz_sets" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "content_id" uuid,
    "quiz" jsonb,
    "created_at" timestamp with time zone default now(),
    "set_number" integer not null,
    "title" text,
    "description" text,
    "created_by" uuid,
    "is_shared" boolean default false
      );



  create table "public"."recordings" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "space_id" uuid,
    "audio_url" text not null,
    "extracted_text" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "file_name" text
      );



  create table "public"."referral" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "code" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
      );



  create table "public"."referral_codes" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "code" text not null,
    "description" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "slots" numeric not null default 500,
    "available_slots" numeric not null default 500
      );



  create table "public"."reviews" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "text" text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "rating" integer,
    "enjoyed_features" text[],
    "improvement_suggestion" text,
    "nps_score" integer,
    "allow_contact" boolean default true,
    "feedback_source" text default 'feedback_dialog'::text
      );



  create table "public"."share_audit_logs" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "action" text not null,
    "resource_type" text,
    "resource_id" uuid,
    "share_token" text,
    "metadata" jsonb,
    "created_at" timestamp with time zone default now()
      );



  create table "public"."share_link_access" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "share_link_id" uuid not null,
    "user_id" uuid not null,
    "accessed_at" timestamp with time zone default now()
      );



  create table "public"."share_links" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "share_token" text not null,
    "resource_id" uuid not null,
    "resource_type" text not null,
    "owner_id" uuid not null,
    "visibility" text default 'private'::text,
    "permission_level" text default 'viewer'::text,
    "expires_at" timestamp with time zone,
    "access_count" integer default 0,
    "max_uses" integer,
    "created_at" timestamp with time zone default now(),
    "last_accessed_at" timestamp with time zone,
    "space_id" uuid
      );



  create table "public"."space_shares" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "space_id" uuid not null,
    "owner_id" uuid not null,
    "shared_with_user_id" uuid not null,
    "permission_level" text default 'viewer'::text,
    "created_at" timestamp with time zone default now()
      );



  create table "public"."spaces" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "name" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "summary" text,
    "thumbnail" text,
    "number_of_files" integer not null default 0,
    "file_types" text[],
    "visibility" text default 'private'::text,
    "created_by" uuid
      );



  create table "public"."subscriptions" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "subscription_status" public.subscription_status_type,
    "subscription_start_date" timestamp without time zone,
    "subscription_end_date" timestamp without time zone,
    "renewal_date" timestamp without time zone,
    "cancellation_date" timestamp without time zone,
    "last_payment_status" text,
    "payment_platform" text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "plan_id" uuid,
    "duration" text,
    "paystack_sub_code" text,
    "stripe_sub_code" text,
    "email_token" text
      );



  create table "public"."user_chat_sessions" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid not null,
    "space_id" uuid not null,
    "is_private" boolean default true,
    "created_at" timestamp with time zone default now()
      );



  create table "public"."user_list" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "email" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "full_name" text,
    "country" text
      );



  create table "public"."user_notes" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid not null,
    "generated_content_id" uuid not null,
    "note_content" text,
    "is_private" boolean default true,
    "forked_from_shared" boolean default false,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
      );



  create table "public"."user_settings" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid,
    "preferred_learning_style" public.learning_style_type,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "role" text
      );


alter table "public"."user_settings" enable row level security;


  create table "public"."writing_blocks" (
    "id" uuid not null default gen_random_uuid(),
    "session_id" uuid not null,
    "section_id" text not null,
    "content" text not null,
    "status" text not null default 'pending'::text,
    "order_index" integer not null,
    "created_at" timestamp with time zone default now()
      );



  create table "public"."writing_references" (
    "id" uuid not null default gen_random_uuid(),
    "session_id" uuid not null,
    "citation_json" jsonb not null,
    "used_in_sections" text[] default ARRAY[]::text[],
    "created_at" timestamp with time zone default now()
      );



  create table "public"."writing_reflections" (
    "id" uuid not null default gen_random_uuid(),
    "session_id" uuid not null,
    "section_id" text not null,
    "reflection_type" text default 'section'::text,
    "coherence_score" numeric(3,1),
    "quality_score" numeric(3,1),
    "issues" jsonb default '[]'::jsonb,
    "sources_cited" jsonb default '[]'::jsonb,
    "unused_sources" jsonb default '[]'::jsonb,
    "source_distribution" jsonb default '{}'::jsonb,
    "concepts_introduced" jsonb default '[]'::jsonb,
    "repeated_concepts" jsonb default '[]'::jsonb,
    "transitions_quality" text,
    "next_section_guidance" text,
    "suggestions" jsonb default '[]'::jsonb,
    "word_count" integer,
    "paragraph_count" integer,
    "citation_count" integer,
    "processing_time_ms" integer,
    "created_at" timestamp with time zone default now()
      );


alter table "public"."writing_reflections" enable row level security;


  create table "public"."writing_section_plans" (
    "id" uuid not null default gen_random_uuid(),
    "session_id" uuid not null,
    "section_id" text not null,
    "plan_version" integer default 1,
    "is_active" boolean default true,
    "key_concepts" jsonb default '[]'::jsonb,
    "sources_to_cite" jsonb default '[]'::jsonb,
    "connection_strategy" text,
    "avoid_repeating" jsonb default '[]'::jsonb,
    "estimated_blocks" integer,
    "estimated_word_count" integer,
    "tone_guidance" text,
    "special_instructions" text,
    "processing_time_ms" integer,
    "created_at" timestamp with time zone default now()
      );


alter table "public"."writing_section_plans" enable row level security;


  create table "public"."writing_sessions" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "writing_type" text not null,
    "status" text not null default 'understanding'::text,
    "initial_prompt" text,
    "conversation_context" jsonb default '[]'::jsonb,
    "word_count_range" text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "editor_content" text,
    "current_section_id" text,
    "section_progress" jsonb default '{}'::jsonb,
    "document_format" text default 'general'::text,
    "citation_style" text default 'APA'::text,
    "section_states" jsonb default '{}'::jsonb,
    "writing_memory" jsonb default '{"last_updated": null, "source_usage": {}, "style_profile": {"tone_keywords": [], "common_transitions": [], "avg_sentence_length": 0}, "concepts_covered": {}, "paragraph_hashes": []}'::jsonb
      );



  create table "public"."writing_structures" (
    "id" uuid not null default gen_random_uuid(),
    "session_id" uuid not null,
    "outline_json" jsonb not null,
    "version" integer default 1,
    "is_active" boolean default true,
    "created_at" timestamp with time zone default now()
      );



  create table "public"."writing_user_sources" (
    "id" uuid not null default gen_random_uuid(),
    "session_id" uuid not null,
    "user_id" uuid not null,
    "type" text not null,
    "title" text not null,
    "original_filename" text,
    "original_url" text,
    "file_path" text,
    "file_hash" text,
    "file_size" integer,
    "processing_status" text default 'pending'::text,
    "processing_error" text,
    "extracted_text" text,
    "extracted_markdown" text,
    "ai_summary" text,
    "key_quotes" jsonb default '[]'::jsonb,
    "user_notes" text,
    "highlights" jsonb default '[]'::jsonb,
    "author" text,
    "year" text,
    "page_count" integer,
    "metadata" jsonb default '{}'::jsonb,
    "cited_in_sections" jsonb default '[]'::jsonb,
    "citation_count" integer default 0,
    "citation_number" integer,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
      );


alter table "public"."writing_user_sources" enable row level security;


  create table "public"."yts" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "space_id" uuid,
    "yt_url" text not null,
    "extracted_text" text not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "thumbnail" text,
    "file_name" text,
    "source" character varying(50) default 'youtube'::character varying,
    "drive_file_id" character varying(255),
    "mime_type" character varying(100),
    "file_size" bigint,
    "external_created_at" timestamp with time zone,
    "external_modified_at" timestamp with time zone,
    "visibility" text default 'private'::text,
    "added_by" uuid
      );


alter sequence "public"."pdf_chats_id_seq" owned by "public"."pdf_chats"."id";

alter sequence "public"."pdf_passages_id_seq" owned by "public"."pdf_passages"."id";

CREATE UNIQUE INDEX chat_library_pkey ON public.chat_library USING btree (id);

CREATE UNIQUE INDEX conversations_conversation_id_key ON public.conversations USING btree (conversation_id);

CREATE UNIQUE INDEX conversations_pkey ON public.conversations USING btree (id);

CREATE UNIQUE INDEX document_shares_document_id_document_type_shared_with_user__key ON public.document_shares USING btree (document_id, document_type, shared_with_user_id);

CREATE UNIQUE INDEX document_shares_pkey ON public.document_shares USING btree (id);

CREATE UNIQUE INDEX flashcard_sets_content_id_set_number_key ON public.flashcard_sets USING btree (content_id, set_number);

CREATE UNIQUE INDEX flashcard_sets_pkey ON public.flashcard_sets USING btree (id);

CREATE UNIQUE INDEX generated_content_pkey ON public.generated_content USING btree (id);

CREATE UNIQUE INDEX google_auth_tokens_pkey ON public.google_auth_tokens USING btree (id);

CREATE UNIQUE INDEX google_auth_tokens_user_id_space_id_key ON public.google_auth_tokens USING btree (user_id, space_id);

CREATE INDEX idx_chat_library_conversation_id ON public.chat_library USING btree (conversation_id);

CREATE INDEX idx_chat_library_user_id ON public.chat_library USING btree (user_id);

CREATE INDEX idx_document_shares_document ON public.document_shares USING btree (document_id, document_type);

CREATE INDEX idx_document_shares_user_id ON public.document_shares USING btree (shared_with_user_id);

CREATE INDEX idx_flashcard_sets_created_by ON public.flashcard_sets USING btree (created_by);

CREATE INDEX idx_flashcard_sets_shared ON public.flashcard_sets USING btree (is_shared);

CREATE INDEX idx_google_tokens_expires ON public.google_auth_tokens USING btree (token_expires_at);

CREATE INDEX idx_google_tokens_user_space ON public.google_auth_tokens USING btree (user_id, space_id);

CREATE INDEX idx_memories_user_id ON public.memories USING btree (user_id);

CREATE INDEX idx_pdf_chats_context ON public.pdf_chats USING btree (chat_context);

CREATE INDEX idx_pdf_chats_conversation_pair_id ON public.pdf_chats USING btree (conversation_pair_id);

CREATE INDEX idx_pdf_chats_pdf_id ON public.pdf_chats USING btree (pdf_id);

CREATE INDEX idx_pdf_chats_shared ON public.pdf_chats USING btree (is_shared);

CREATE INDEX idx_pdf_chats_user_session ON public.pdf_chats USING btree (user_session_id);

CREATE INDEX idx_pdf_chats_yt_id ON public.pdf_chats USING btree (yt_id);

CREATE INDEX idx_pdfs_drive_file_id ON public.pdfs USING btree (drive_file_id) WHERE (drive_file_id IS NOT NULL);

CREATE INDEX idx_pdfs_source ON public.pdfs USING btree (source);

CREATE INDEX idx_pdfs_uploaded_by ON public.pdfs USING btree (uploaded_by);

CREATE INDEX idx_pdfs_visibility ON public.pdfs USING btree (visibility);

CREATE INDEX idx_quiz_sets_created_by ON public.quiz_sets USING btree (created_by);

CREATE INDEX idx_quiz_sets_shared ON public.quiz_sets USING btree (is_shared);

CREATE INDEX idx_share_audit_logs_action ON public.share_audit_logs USING btree (action);

CREATE INDEX idx_share_audit_logs_created_at ON public.share_audit_logs USING btree (created_at DESC);

CREATE INDEX idx_share_audit_logs_resource ON public.share_audit_logs USING btree (resource_id, resource_type);

CREATE INDEX idx_share_audit_logs_share_token ON public.share_audit_logs USING btree (share_token);

CREATE INDEX idx_share_audit_logs_user_id ON public.share_audit_logs USING btree (user_id);

CREATE INDEX idx_share_link_access_link_id ON public.share_link_access USING btree (share_link_id);

CREATE INDEX idx_share_link_access_user_id ON public.share_link_access USING btree (user_id);

CREATE INDEX idx_share_links_owner_id ON public.share_links USING btree (owner_id);

CREATE INDEX idx_share_links_resource ON public.share_links USING btree (resource_id, resource_type);

CREATE INDEX idx_share_links_space_id ON public.share_links USING btree (space_id);

CREATE INDEX idx_share_links_token ON public.share_links USING btree (share_token);

CREATE INDEX idx_space_shares_owner_id ON public.space_shares USING btree (owner_id);

CREATE INDEX idx_space_shares_space_id ON public.space_shares USING btree (space_id);

CREATE INDEX idx_space_shares_user_id ON public.space_shares USING btree (shared_with_user_id);

CREATE INDEX idx_spaces_created_by ON public.spaces USING btree (created_by);

CREATE INDEX idx_spaces_visibility ON public.spaces USING btree (visibility);

CREATE INDEX idx_subscriptions_end_date ON public.subscriptions USING btree (subscription_end_date);

CREATE INDEX idx_subscriptions_plan_id ON public.subscriptions USING btree (plan_id);

CREATE INDEX idx_subscriptions_status ON public.subscriptions USING btree (subscription_status);

CREATE INDEX idx_subscriptions_status_end_date ON public.subscriptions USING btree (subscription_status, subscription_end_date);

CREATE INDEX idx_user_chat_sessions_space_id ON public.user_chat_sessions USING btree (space_id);

CREATE INDEX idx_user_chat_sessions_user_space ON public.user_chat_sessions USING btree (user_id, space_id);

CREATE INDEX idx_user_notes_content_id ON public.user_notes USING btree (generated_content_id);

CREATE INDEX idx_user_notes_user_content ON public.user_notes USING btree (user_id, generated_content_id);

CREATE INDEX idx_user_settings_user_id ON public.user_settings USING btree (user_id);

CREATE INDEX idx_writing_blocks_section_id ON public.writing_blocks USING btree (session_id, section_id);

CREATE INDEX idx_writing_blocks_session_id ON public.writing_blocks USING btree (session_id);

CREATE INDEX idx_writing_references_session_id ON public.writing_references USING btree (session_id);

CREATE INDEX idx_writing_reflections_created_at ON public.writing_reflections USING btree (created_at DESC);

CREATE INDEX idx_writing_reflections_section_id ON public.writing_reflections USING btree (section_id);

CREATE INDEX idx_writing_reflections_session_id ON public.writing_reflections USING btree (session_id);

CREATE INDEX idx_writing_section_plans_active ON public.writing_section_plans USING btree (session_id, is_active) WHERE (is_active = true);

CREATE INDEX idx_writing_section_plans_created_at ON public.writing_section_plans USING btree (created_at DESC);

CREATE INDEX idx_writing_section_plans_section_id ON public.writing_section_plans USING btree (section_id);

CREATE INDEX idx_writing_section_plans_session_id ON public.writing_section_plans USING btree (session_id);

CREATE INDEX idx_writing_sessions_content ON public.writing_sessions USING btree (id) WHERE (editor_content IS NOT NULL);

CREATE INDEX idx_writing_sessions_document_format ON public.writing_sessions USING btree (document_format);

CREATE INDEX idx_writing_sessions_memory_concepts ON public.writing_sessions USING gin (((writing_memory -> 'concepts_covered'::text)));

CREATE INDEX idx_writing_sessions_memory_sources ON public.writing_sessions USING gin (((writing_memory -> 'source_usage'::text)));

CREATE INDEX idx_writing_sessions_status ON public.writing_sessions USING btree (status);

CREATE INDEX idx_writing_sessions_user_id ON public.writing_sessions USING btree (user_id);

CREATE INDEX idx_writing_structures_active ON public.writing_structures USING btree (session_id, is_active) WHERE (is_active = true);

CREATE INDEX idx_writing_structures_session_id ON public.writing_structures USING btree (session_id);

CREATE INDEX idx_writing_user_sources_file_hash ON public.writing_user_sources USING btree (file_hash) WHERE (file_hash IS NOT NULL);

CREATE INDEX idx_writing_user_sources_processing_status ON public.writing_user_sources USING btree (processing_status);

CREATE INDEX idx_writing_user_sources_session_id ON public.writing_user_sources USING btree (session_id);

CREATE INDEX idx_writing_user_sources_type ON public.writing_user_sources USING btree (type);

CREATE INDEX idx_writing_user_sources_user_id ON public.writing_user_sources USING btree (user_id);

CREATE INDEX idx_yts_added_by ON public.yts USING btree (added_by);

CREATE INDEX idx_yts_source ON public.yts USING btree (source);

CREATE INDEX idx_yts_visibility ON public.yts USING btree (visibility);

CREATE INDEX memories_content_search_idx ON public.memories USING gin (content_search);

CREATE UNIQUE INDEX memories_pkey ON public.memories USING btree (id);

CREATE UNIQUE INDEX pdf_chats_pkey ON public.pdf_chats USING btree (id);

CREATE UNIQUE INDEX pdf_embeddings_pkey ON public.pdf_embeddings USING btree (id);

CREATE UNIQUE INDEX pdf_passages_pkey ON public.pdf_passages USING btree (id);

CREATE UNIQUE INDEX pdfs_pkey ON public.pdfs USING btree (id);

CREATE UNIQUE INDEX plans_pkey ON public.plans USING btree (id);

CREATE UNIQUE INDEX quiz_attempts_pkey ON public.quiz_attempts USING btree (id);

CREATE UNIQUE INDEX quiz_attempts_quiz_set_id_user_id_started_at_key ON public.quiz_attempts USING btree (quiz_set_id, user_id, started_at);

CREATE UNIQUE INDEX quiz_sets_content_id_set_number_key ON public.quiz_sets USING btree (content_id, set_number);

CREATE UNIQUE INDEX quiz_sets_pkey ON public.quiz_sets USING btree (id);

CREATE UNIQUE INDEX recordings_pkey ON public.recordings USING btree (id);

CREATE UNIQUE INDEX referral_codes_code_key ON public.referral_codes USING btree (code);

CREATE UNIQUE INDEX referral_codes_pkey ON public.referral_codes USING btree (id);

CREATE UNIQUE INDEX referral_pkey ON public.referral USING btree (id);

CREATE UNIQUE INDEX reviews_pkey ON public.reviews USING btree (id);

CREATE UNIQUE INDEX share_audit_logs_pkey ON public.share_audit_logs USING btree (id);

CREATE UNIQUE INDEX share_link_access_pkey ON public.share_link_access USING btree (id);

CREATE UNIQUE INDEX share_link_access_share_link_id_user_id_key ON public.share_link_access USING btree (share_link_id, user_id);

CREATE UNIQUE INDEX share_links_pkey ON public.share_links USING btree (id);

CREATE UNIQUE INDEX share_links_share_token_key ON public.share_links USING btree (share_token);

CREATE UNIQUE INDEX space_shares_pkey ON public.space_shares USING btree (id);

CREATE UNIQUE INDEX space_shares_space_id_shared_with_user_id_key ON public.space_shares USING btree (space_id, shared_with_user_id);

CREATE UNIQUE INDEX spaces_pkey ON public.spaces USING btree (id);

CREATE UNIQUE INDEX subscriptions_pkey ON public.subscriptions USING btree (id);

CREATE UNIQUE INDEX subscriptions_user_id_key ON public.subscriptions USING btree (user_id);

CREATE UNIQUE INDEX user_chat_sessions_pkey ON public.user_chat_sessions USING btree (id);

CREATE UNIQUE INDEX user_list_email_key ON public.user_list USING btree (email);

CREATE UNIQUE INDEX user_list_pkey ON public.user_list USING btree (id);

CREATE UNIQUE INDEX user_notes_pkey ON public.user_notes USING btree (id);

CREATE UNIQUE INDEX user_notes_user_id_generated_content_id_key ON public.user_notes USING btree (user_id, generated_content_id);

CREATE UNIQUE INDEX user_settings_pkey ON public.user_settings USING btree (id);

CREATE UNIQUE INDEX user_settings_user_id_key ON public.user_settings USING btree (user_id);

CREATE UNIQUE INDEX writing_blocks_pkey ON public.writing_blocks USING btree (id);

CREATE UNIQUE INDEX writing_references_pkey ON public.writing_references USING btree (id);

CREATE UNIQUE INDEX writing_reflections_pkey ON public.writing_reflections USING btree (id);

CREATE UNIQUE INDEX writing_section_plans_pkey ON public.writing_section_plans USING btree (id);

CREATE UNIQUE INDEX writing_sessions_pkey ON public.writing_sessions USING btree (id);

CREATE UNIQUE INDEX writing_structures_pkey ON public.writing_structures USING btree (id);

CREATE UNIQUE INDEX writing_user_sources_pkey ON public.writing_user_sources USING btree (id);

CREATE UNIQUE INDEX yts_pkey ON public.yts USING btree (id);

alter table "public"."chat_library" add constraint "chat_library_pkey" PRIMARY KEY using index "chat_library_pkey";

alter table "public"."conversations" add constraint "conversations_pkey" PRIMARY KEY using index "conversations_pkey";

alter table "public"."document_shares" add constraint "document_shares_pkey" PRIMARY KEY using index "document_shares_pkey";

alter table "public"."flashcard_sets" add constraint "flashcard_sets_pkey" PRIMARY KEY using index "flashcard_sets_pkey";

alter table "public"."generated_content" add constraint "generated_content_pkey" PRIMARY KEY using index "generated_content_pkey";

alter table "public"."google_auth_tokens" add constraint "google_auth_tokens_pkey" PRIMARY KEY using index "google_auth_tokens_pkey";

alter table "public"."memories" add constraint "memories_pkey" PRIMARY KEY using index "memories_pkey";

alter table "public"."pdf_chats" add constraint "pdf_chats_pkey" PRIMARY KEY using index "pdf_chats_pkey";

alter table "public"."pdf_embeddings" add constraint "pdf_embeddings_pkey" PRIMARY KEY using index "pdf_embeddings_pkey";

alter table "public"."pdf_passages" add constraint "pdf_passages_pkey" PRIMARY KEY using index "pdf_passages_pkey";

alter table "public"."pdfs" add constraint "pdfs_pkey" PRIMARY KEY using index "pdfs_pkey";

alter table "public"."plans" add constraint "plans_pkey" PRIMARY KEY using index "plans_pkey";

alter table "public"."quiz_attempts" add constraint "quiz_attempts_pkey" PRIMARY KEY using index "quiz_attempts_pkey";

alter table "public"."quiz_sets" add constraint "quiz_sets_pkey" PRIMARY KEY using index "quiz_sets_pkey";

alter table "public"."recordings" add constraint "recordings_pkey" PRIMARY KEY using index "recordings_pkey";

alter table "public"."referral" add constraint "referral_pkey" PRIMARY KEY using index "referral_pkey";

alter table "public"."referral_codes" add constraint "referral_codes_pkey" PRIMARY KEY using index "referral_codes_pkey";

alter table "public"."reviews" add constraint "reviews_pkey" PRIMARY KEY using index "reviews_pkey";

alter table "public"."share_audit_logs" add constraint "share_audit_logs_pkey" PRIMARY KEY using index "share_audit_logs_pkey";

alter table "public"."share_link_access" add constraint "share_link_access_pkey" PRIMARY KEY using index "share_link_access_pkey";

alter table "public"."share_links" add constraint "share_links_pkey" PRIMARY KEY using index "share_links_pkey";

alter table "public"."space_shares" add constraint "space_shares_pkey" PRIMARY KEY using index "space_shares_pkey";

alter table "public"."spaces" add constraint "spaces_pkey" PRIMARY KEY using index "spaces_pkey";

alter table "public"."subscriptions" add constraint "subscriptions_pkey" PRIMARY KEY using index "subscriptions_pkey";

alter table "public"."user_chat_sessions" add constraint "user_chat_sessions_pkey" PRIMARY KEY using index "user_chat_sessions_pkey";

alter table "public"."user_list" add constraint "user_list_pkey" PRIMARY KEY using index "user_list_pkey";

alter table "public"."user_notes" add constraint "user_notes_pkey" PRIMARY KEY using index "user_notes_pkey";

alter table "public"."user_settings" add constraint "user_settings_pkey" PRIMARY KEY using index "user_settings_pkey";

alter table "public"."writing_blocks" add constraint "writing_blocks_pkey" PRIMARY KEY using index "writing_blocks_pkey";

alter table "public"."writing_references" add constraint "writing_references_pkey" PRIMARY KEY using index "writing_references_pkey";

alter table "public"."writing_reflections" add constraint "writing_reflections_pkey" PRIMARY KEY using index "writing_reflections_pkey";

alter table "public"."writing_section_plans" add constraint "writing_section_plans_pkey" PRIMARY KEY using index "writing_section_plans_pkey";

alter table "public"."writing_sessions" add constraint "writing_sessions_pkey" PRIMARY KEY using index "writing_sessions_pkey";

alter table "public"."writing_structures" add constraint "writing_structures_pkey" PRIMARY KEY using index "writing_structures_pkey";

alter table "public"."writing_user_sources" add constraint "writing_user_sources_pkey" PRIMARY KEY using index "writing_user_sources_pkey";

alter table "public"."yts" add constraint "yts_pkey" PRIMARY KEY using index "yts_pkey";

alter table "public"."conversations" add constraint "conversations_conversation_id_key" UNIQUE using index "conversations_conversation_id_key";

alter table "public"."conversations" add constraint "conversations_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."conversations" validate constraint "conversations_user_id_fkey";

alter table "public"."document_shares" add constraint "document_shares_document_id_document_type_shared_with_user__key" UNIQUE using index "document_shares_document_id_document_type_shared_with_user__key";

alter table "public"."document_shares" add constraint "document_shares_document_type_check" CHECK ((document_type = ANY (ARRAY['pdf'::text, 'youtube'::text]))) not valid;

alter table "public"."document_shares" validate constraint "document_shares_document_type_check";

alter table "public"."document_shares" add constraint "document_shares_owner_id_fkey" FOREIGN KEY (owner_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."document_shares" validate constraint "document_shares_owner_id_fkey";

alter table "public"."document_shares" add constraint "document_shares_permission_level_check" CHECK ((permission_level = ANY (ARRAY['viewer'::text, 'editor'::text]))) not valid;

alter table "public"."document_shares" validate constraint "document_shares_permission_level_check";

alter table "public"."document_shares" add constraint "document_shares_shared_with_user_id_fkey" FOREIGN KEY (shared_with_user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."document_shares" validate constraint "document_shares_shared_with_user_id_fkey";

alter table "public"."flashcard_sets" add constraint "flashcard_sets_content_id_fkey" FOREIGN KEY (content_id) REFERENCES public.generated_content(id) not valid;

alter table "public"."flashcard_sets" validate constraint "flashcard_sets_content_id_fkey";

alter table "public"."flashcard_sets" add constraint "flashcard_sets_content_id_set_number_key" UNIQUE using index "flashcard_sets_content_id_set_number_key";

alter table "public"."flashcard_sets" add constraint "flashcard_sets_created_by_fkey" FOREIGN KEY (created_by) REFERENCES auth.users(id) not valid;

alter table "public"."flashcard_sets" validate constraint "flashcard_sets_created_by_fkey";

alter table "public"."generated_content" add constraint "generated_content_audio_id_fkey" FOREIGN KEY (audio_id) REFERENCES public.recordings(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_audio_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_pdf_id_fkey" FOREIGN KEY (pdf_id) REFERENCES public.pdfs(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_pdf_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_space_id_fkey";

alter table "public"."generated_content" add constraint "generated_content_yt_id_fkey" FOREIGN KEY (yt_id) REFERENCES public.yts(id) ON DELETE CASCADE not valid;

alter table "public"."generated_content" validate constraint "generated_content_yt_id_fkey";

alter table "public"."google_auth_tokens" add constraint "google_auth_tokens_user_id_space_id_key" UNIQUE using index "google_auth_tokens_user_id_space_id_key";

alter table "public"."memories" add constraint "content_length" CHECK ((char_length(content) > 0)) not valid;

alter table "public"."memories" validate constraint "content_length";

alter table "public"."pdf_chats" add constraint "check_document_context" CHECK ((((chat_context = 'space'::text) AND (pdf_id IS NULL) AND (yt_id IS NULL)) OR ((chat_context = 'pdf'::text) AND (pdf_id IS NOT NULL) AND (yt_id IS NULL)) OR ((chat_context = 'youtube'::text) AND (yt_id IS NOT NULL) AND (pdf_id IS NULL)))) not valid;

alter table "public"."pdf_chats" validate constraint "check_document_context";

alter table "public"."pdf_chats" add constraint "pdf_chats_chat_context_check" CHECK ((chat_context = ANY (ARRAY['space'::text, 'pdf'::text, 'youtube'::text]))) not valid;

alter table "public"."pdf_chats" validate constraint "pdf_chats_chat_context_check";

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

alter table "public"."pdfs" add constraint "pdfs_uploaded_by_fkey" FOREIGN KEY (uploaded_by) REFERENCES auth.users(id) not valid;

alter table "public"."pdfs" validate constraint "pdfs_uploaded_by_fkey";

alter table "public"."pdfs" add constraint "pdfs_visibility_check" CHECK ((visibility = ANY (ARRAY['public'::text, 'private'::text]))) not valid;

alter table "public"."pdfs" validate constraint "pdfs_visibility_check";

alter table "public"."quiz_attempts" add constraint "quiz_attempts_quiz_set_id_fkey" FOREIGN KEY (quiz_set_id) REFERENCES public.quiz_sets(id) not valid;

alter table "public"."quiz_attempts" validate constraint "quiz_attempts_quiz_set_id_fkey";

alter table "public"."quiz_attempts" add constraint "quiz_attempts_quiz_set_id_user_id_started_at_key" UNIQUE using index "quiz_attempts_quiz_set_id_user_id_started_at_key";

alter table "public"."quiz_attempts" add constraint "quiz_attempts_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."quiz_attempts" validate constraint "quiz_attempts_user_id_fkey";

alter table "public"."quiz_sets" add constraint "quiz_sets_content_id_fkey" FOREIGN KEY (content_id) REFERENCES public.generated_content(id) not valid;

alter table "public"."quiz_sets" validate constraint "quiz_sets_content_id_fkey";

alter table "public"."quiz_sets" add constraint "quiz_sets_content_id_set_number_key" UNIQUE using index "quiz_sets_content_id_set_number_key";

alter table "public"."quiz_sets" add constraint "quiz_sets_created_by_fkey" FOREIGN KEY (created_by) REFERENCES auth.users(id) not valid;

alter table "public"."quiz_sets" validate constraint "quiz_sets_created_by_fkey";

alter table "public"."recordings" add constraint "recordings_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."recordings" validate constraint "recordings_space_id_fkey";

alter table "public"."referral" add constraint "referral_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."referral" validate constraint "referral_user_id_fkey";

alter table "public"."referral_codes" add constraint "referral_codes_code_key" UNIQUE using index "referral_codes_code_key";

alter table "public"."reviews" add constraint "reviews_nps_score_check" CHECK (((nps_score >= 0) AND (nps_score <= 10))) not valid;

alter table "public"."reviews" validate constraint "reviews_nps_score_check";

alter table "public"."reviews" add constraint "reviews_rating_check" CHECK (((rating >= 1) AND (rating <= 5))) not valid;

alter table "public"."reviews" validate constraint "reviews_rating_check";

alter table "public"."reviews" add constraint "reviews_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."reviews" validate constraint "reviews_user_id_fkey";

alter table "public"."share_audit_logs" add constraint "share_audit_logs_resource_type_check" CHECK ((resource_type = ANY (ARRAY['space'::text, 'pdf'::text, 'youtube'::text]))) not valid;

alter table "public"."share_audit_logs" validate constraint "share_audit_logs_resource_type_check";

alter table "public"."share_audit_logs" add constraint "share_audit_logs_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE SET NULL not valid;

alter table "public"."share_audit_logs" validate constraint "share_audit_logs_user_id_fkey";

alter table "public"."share_link_access" add constraint "share_link_access_share_link_id_fkey" FOREIGN KEY (share_link_id) REFERENCES public.share_links(id) ON DELETE CASCADE not valid;

alter table "public"."share_link_access" validate constraint "share_link_access_share_link_id_fkey";

alter table "public"."share_link_access" add constraint "share_link_access_share_link_id_user_id_key" UNIQUE using index "share_link_access_share_link_id_user_id_key";

alter table "public"."share_link_access" add constraint "share_link_access_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."share_link_access" validate constraint "share_link_access_user_id_fkey";

alter table "public"."share_links" add constraint "share_links_owner_id_fkey" FOREIGN KEY (owner_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."share_links" validate constraint "share_links_owner_id_fkey";

alter table "public"."share_links" add constraint "share_links_permission_level_check" CHECK ((permission_level = ANY (ARRAY['viewer'::text, 'editor'::text]))) not valid;

alter table "public"."share_links" validate constraint "share_links_permission_level_check";

alter table "public"."share_links" add constraint "share_links_resource_type_check" CHECK ((resource_type = ANY (ARRAY['space'::text, 'pdf'::text, 'youtube'::text]))) not valid;

alter table "public"."share_links" validate constraint "share_links_resource_type_check";

alter table "public"."share_links" add constraint "share_links_share_token_key" UNIQUE using index "share_links_share_token_key";

alter table "public"."share_links" add constraint "share_links_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."share_links" validate constraint "share_links_space_id_fkey";

alter table "public"."share_links" add constraint "share_links_visibility_check" CHECK ((visibility = ANY (ARRAY['public'::text, 'private'::text]))) not valid;

alter table "public"."share_links" validate constraint "share_links_visibility_check";

alter table "public"."space_shares" add constraint "space_shares_owner_id_fkey" FOREIGN KEY (owner_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."space_shares" validate constraint "space_shares_owner_id_fkey";

alter table "public"."space_shares" add constraint "space_shares_permission_level_check" CHECK ((permission_level = ANY (ARRAY['viewer'::text, 'editor'::text, 'admin'::text]))) not valid;

alter table "public"."space_shares" validate constraint "space_shares_permission_level_check";

alter table "public"."space_shares" add constraint "space_shares_shared_with_user_id_fkey" FOREIGN KEY (shared_with_user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."space_shares" validate constraint "space_shares_shared_with_user_id_fkey";

alter table "public"."space_shares" add constraint "space_shares_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."space_shares" validate constraint "space_shares_space_id_fkey";

alter table "public"."space_shares" add constraint "space_shares_space_id_shared_with_user_id_key" UNIQUE using index "space_shares_space_id_shared_with_user_id_key";

alter table "public"."spaces" add constraint "spaces_created_by_fkey" FOREIGN KEY (created_by) REFERENCES auth.users(id) not valid;

alter table "public"."spaces" validate constraint "spaces_created_by_fkey";

alter table "public"."spaces" add constraint "spaces_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."spaces" validate constraint "spaces_user_id_fkey";

alter table "public"."spaces" add constraint "spaces_visibility_check" CHECK ((visibility = ANY (ARRAY['public'::text, 'private'::text]))) not valid;

alter table "public"."spaces" validate constraint "spaces_visibility_check";

alter table "public"."subscriptions" add constraint "subscriptions_plan_id_fkey" FOREIGN KEY (plan_id) REFERENCES public.plans(id) not valid;

alter table "public"."subscriptions" validate constraint "subscriptions_plan_id_fkey";

alter table "public"."subscriptions" add constraint "subscriptions_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."subscriptions" validate constraint "subscriptions_user_id_fkey";

alter table "public"."subscriptions" add constraint "subscriptions_user_id_key" UNIQUE using index "subscriptions_user_id_key";

alter table "public"."user_chat_sessions" add constraint "user_chat_sessions_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."user_chat_sessions" validate constraint "user_chat_sessions_space_id_fkey";

alter table "public"."user_chat_sessions" add constraint "user_chat_sessions_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."user_chat_sessions" validate constraint "user_chat_sessions_user_id_fkey";

alter table "public"."user_list" add constraint "user_list_email_key" UNIQUE using index "user_list_email_key";

alter table "public"."user_list" add constraint "user_list_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."user_list" validate constraint "user_list_user_id_fkey";

alter table "public"."user_notes" add constraint "user_notes_generated_content_id_fkey" FOREIGN KEY (generated_content_id) REFERENCES public.generated_content(id) ON DELETE CASCADE not valid;

alter table "public"."user_notes" validate constraint "user_notes_generated_content_id_fkey";

alter table "public"."user_notes" add constraint "user_notes_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."user_notes" validate constraint "user_notes_user_id_fkey";

alter table "public"."user_notes" add constraint "user_notes_user_id_generated_content_id_key" UNIQUE using index "user_notes_user_id_generated_content_id_key";

alter table "public"."user_settings" add constraint "user_settings_role_check" CHECK ((role = ANY (ARRAY['student'::text, 'teacher'::text]))) not valid;

alter table "public"."user_settings" validate constraint "user_settings_role_check";

alter table "public"."user_settings" add constraint "user_settings_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."user_settings" validate constraint "user_settings_user_id_fkey";

alter table "public"."user_settings" add constraint "user_settings_user_id_key" UNIQUE using index "user_settings_user_id_key";

alter table "public"."writing_blocks" add constraint "writing_blocks_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_blocks" validate constraint "writing_blocks_session_id_fkey";

alter table "public"."writing_blocks" add constraint "writing_blocks_status_check" CHECK ((status = ANY (ARRAY['pending'::text, 'accepted'::text, 'rejected'::text, 'editing'::text]))) not valid;

alter table "public"."writing_blocks" validate constraint "writing_blocks_status_check";

alter table "public"."writing_references" add constraint "writing_references_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_references" validate constraint "writing_references_session_id_fkey";

alter table "public"."writing_reflections" add constraint "writing_reflections_coherence_score_check" CHECK (((coherence_score >= (0)::numeric) AND (coherence_score <= (10)::numeric))) not valid;

alter table "public"."writing_reflections" validate constraint "writing_reflections_coherence_score_check";

alter table "public"."writing_reflections" add constraint "writing_reflections_quality_score_check" CHECK (((quality_score >= (0)::numeric) AND (quality_score <= (10)::numeric))) not valid;

alter table "public"."writing_reflections" validate constraint "writing_reflections_quality_score_check";

alter table "public"."writing_reflections" add constraint "writing_reflections_reflection_type_check" CHECK ((reflection_type = ANY (ARRAY['section'::text, 'document'::text]))) not valid;

alter table "public"."writing_reflections" validate constraint "writing_reflections_reflection_type_check";

alter table "public"."writing_reflections" add constraint "writing_reflections_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_reflections" validate constraint "writing_reflections_session_id_fkey";

alter table "public"."writing_section_plans" add constraint "writing_section_plans_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_section_plans" validate constraint "writing_section_plans_session_id_fkey";

alter table "public"."writing_sessions" add constraint "writing_sessions_citation_style_check" CHECK ((citation_style = ANY (ARRAY['APA'::text, 'MLA'::text, 'Chicago'::text, 'IEEE'::text]))) not valid;

alter table "public"."writing_sessions" validate constraint "writing_sessions_citation_style_check";

alter table "public"."writing_sessions" add constraint "writing_sessions_document_format_check" CHECK ((document_format = ANY (ARRAY['research_paper'::text, 'technical_report'::text, 'essay'::text, 'project_proposal'::text, 'general'::text]))) not valid;

alter table "public"."writing_sessions" validate constraint "writing_sessions_document_format_check";

alter table "public"."writing_sessions" add constraint "writing_sessions_status_check" CHECK ((status = ANY (ARRAY['understanding'::text, 'structure'::text, 'writing'::text, 'complete'::text]))) not valid;

alter table "public"."writing_sessions" validate constraint "writing_sessions_status_check";

alter table "public"."writing_sessions" add constraint "writing_sessions_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."writing_sessions" validate constraint "writing_sessions_user_id_fkey";

alter table "public"."writing_sessions" add constraint "writing_sessions_writing_type_check" CHECK ((writing_type = ANY (ARRAY['academic'::text, 'technical'::text, 'personal'::text, 'creative'::text]))) not valid;

alter table "public"."writing_sessions" validate constraint "writing_sessions_writing_type_check";

alter table "public"."writing_structures" add constraint "writing_structures_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_structures" validate constraint "writing_structures_session_id_fkey";

alter table "public"."writing_user_sources" add constraint "writing_user_sources_processing_status_check" CHECK ((processing_status = ANY (ARRAY['pending'::text, 'processing'::text, 'complete'::text, 'failed'::text]))) not valid;

alter table "public"."writing_user_sources" validate constraint "writing_user_sources_processing_status_check";

alter table "public"."writing_user_sources" add constraint "writing_user_sources_session_id_fkey" FOREIGN KEY (session_id) REFERENCES public.writing_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."writing_user_sources" validate constraint "writing_user_sources_session_id_fkey";

alter table "public"."writing_user_sources" add constraint "writing_user_sources_type_check" CHECK ((type = ANY (ARRAY['pdf'::text, 'url'::text, 'image'::text, 'note'::text]))) not valid;

alter table "public"."writing_user_sources" validate constraint "writing_user_sources_type_check";

alter table "public"."writing_user_sources" add constraint "writing_user_sources_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."writing_user_sources" validate constraint "writing_user_sources_user_id_fkey";

alter table "public"."yts" add constraint "yts_added_by_fkey" FOREIGN KEY (added_by) REFERENCES auth.users(id) not valid;

alter table "public"."yts" validate constraint "yts_added_by_fkey";

alter table "public"."yts" add constraint "yts_space_id_fkey" FOREIGN KEY (space_id) REFERENCES public.spaces(id) ON DELETE CASCADE not valid;

alter table "public"."yts" validate constraint "yts_space_id_fkey";

alter table "public"."yts" add constraint "yts_visibility_check" CHECK ((visibility = ANY (ARRAY['public'::text, 'private'::text]))) not valid;

alter table "public"."yts" validate constraint "yts_visibility_check";

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.check_expired_subscriptions()
 RETURNS void
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  free_plan_id uuid;
  expired_count int;
BEGIN
  SELECT id INTO free_plan_id FROM plans WHERE name = 'Free' LIMIT 1;
  IF free_plan_id IS NULL THEN
    RAISE EXCEPTION 'Free plan not found in plans table';
  END IF;
  WITH updated AS (
    UPDATE subscriptions
    SET subscription_status = 'expired', plan_id = free_plan_id, updated_at = NOW()
    WHERE subscription_end_date < NOW()
      AND subscription_status = 'active'
      AND plan_id != free_plan_id
    RETURNING *
  )
  SELECT COUNT(*) INTO expired_count FROM updated;
  RAISE NOTICE 'Expired % subscriptions at %', expired_count, NOW();
END;
$function$
;

CREATE OR REPLACE FUNCTION public.cleanup_expired_subscription_plans()
 RETURNS void
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$ DECLARE free_plan_id uuid; fixed_count int; BEGIN SELECT id INTO free_plan_id FROM plans WHERE name = 'Free' LIMIT 1; IF free_plan_id IS NULL THEN RAISE EXCEPTION 'Free plan not found in plans table'; END IF; WITH fixed AS ( UPDATE subscriptions SET plan_id = free_plan_id, updated_at = NOW() WHERE subscription_status = 'expired' AND plan_id != free_plan_id RETURNING * ) SELECT COUNT(*) INTO fixed_count FROM fixed; RAISE NOTICE 'Fixed % expired subscriptions with wrong plan_id at %', fixed_count, NOW(); END; $function$
;

CREATE OR REPLACE FUNCTION public.cleanup_stale_subscriptions()
 RETURNS TABLE(user_id uuid, issue_type text, subscription_status text, end_date timestamp without time zone, plan_name text)
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  free_plan_id uuid;
  fixed_count int;
BEGIN
  -- Get the Free plan ID
  SELECT id INTO free_plan_id
  FROM plans
  WHERE name = 'Free'
  LIMIT 1;

  -- Return subscriptions with data integrity issues
  -- (active status but expired for more than 7 days)
  RETURN QUERY
  SELECT
    s.user_id,
    'Active subscription past end date (7+ days)' AS issue_type,
    s.subscription_status::text,
    s.subscription_end_date AS end_date,
    p.name AS plan_name
  FROM subscriptions s
  JOIN plans p ON s.plan_id = p.id
  WHERE
    s.subscription_status = 'active'
    AND s.subscription_end_date < NOW() - INTERVAL '7 days'
    AND s.plan_id != free_plan_id;

  -- Auto-fix these stale subscriptions
  WITH fixed AS (
    UPDATE subscriptions s
    SET
      subscription_status = 'expired',
      plan_id = free_plan_id,
      updated_at = NOW()
    WHERE
      s.subscription_status = 'active'
      AND s.subscription_end_date < NOW() - INTERVAL '7 days'
      AND s.plan_id != free_plan_id
    RETURNING *
  )
  SELECT COUNT(*) INTO fixed_count FROM fixed;

  -- Log the cleanup action
  RAISE NOTICE 'Cleaned up % stale subscriptions at %', fixed_count, NOW();
END;
$function$
;

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

CREATE OR REPLACE FUNCTION public.get_covered_concepts(p_session_id uuid)
 RETURNS jsonb
 LANGUAGE sql
 STABLE
AS $function$
  SELECT writing_memory->'concepts_covered'
  FROM writing_sessions
  WHERE id = p_session_id;
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

CREATE OR REPLACE FUNCTION public.get_source_usage_stats(p_session_id uuid)
 RETURNS jsonb
 LANGUAGE sql
 STABLE
AS $function$
  SELECT writing_memory->'source_usage'
  FROM writing_sessions
  WHERE id = p_session_id;
$function$
;

CREATE OR REPLACE FUNCTION public.handle_pending_cancellations()
 RETURNS void
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  free_plan_id uuid;
  cancelled_count int;
BEGIN
  SELECT id INTO free_plan_id FROM plans WHERE name = 'Free' LIMIT 1;
  IF free_plan_id IS NULL THEN
    RAISE EXCEPTION 'Free plan not found in plans table';
  END IF;
  WITH updated AS (
    UPDATE subscriptions
    SET subscription_status = 'cancelled', plan_id = free_plan_id, updated_at = NOW()
    WHERE subscription_status = 'cancelling'
      AND subscription_end_date < NOW()
      AND plan_id != free_plan_id
    RETURNING *
  )
  SELECT COUNT(*) INTO cancelled_count FROM updated;
  RAISE NOTICE 'Cancelled % subscriptions at %', cancelled_count, NOW();
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_writing_sessions_updated_at()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$function$
;

grant delete on table "public"."chat_library" to "anon";

grant insert on table "public"."chat_library" to "anon";

grant references on table "public"."chat_library" to "anon";

grant select on table "public"."chat_library" to "anon";

grant trigger on table "public"."chat_library" to "anon";

grant truncate on table "public"."chat_library" to "anon";

grant update on table "public"."chat_library" to "anon";

grant delete on table "public"."chat_library" to "authenticated";

grant insert on table "public"."chat_library" to "authenticated";

grant references on table "public"."chat_library" to "authenticated";

grant select on table "public"."chat_library" to "authenticated";

grant trigger on table "public"."chat_library" to "authenticated";

grant truncate on table "public"."chat_library" to "authenticated";

grant update on table "public"."chat_library" to "authenticated";

grant delete on table "public"."chat_library" to "service_role";

grant insert on table "public"."chat_library" to "service_role";

grant references on table "public"."chat_library" to "service_role";

grant select on table "public"."chat_library" to "service_role";

grant trigger on table "public"."chat_library" to "service_role";

grant truncate on table "public"."chat_library" to "service_role";

grant update on table "public"."chat_library" to "service_role";

grant delete on table "public"."conversations" to "anon";

grant insert on table "public"."conversations" to "anon";

grant references on table "public"."conversations" to "anon";

grant select on table "public"."conversations" to "anon";

grant trigger on table "public"."conversations" to "anon";

grant truncate on table "public"."conversations" to "anon";

grant update on table "public"."conversations" to "anon";

grant delete on table "public"."conversations" to "authenticated";

grant insert on table "public"."conversations" to "authenticated";

grant references on table "public"."conversations" to "authenticated";

grant select on table "public"."conversations" to "authenticated";

grant trigger on table "public"."conversations" to "authenticated";

grant truncate on table "public"."conversations" to "authenticated";

grant update on table "public"."conversations" to "authenticated";

grant delete on table "public"."conversations" to "service_role";

grant insert on table "public"."conversations" to "service_role";

grant references on table "public"."conversations" to "service_role";

grant select on table "public"."conversations" to "service_role";

grant trigger on table "public"."conversations" to "service_role";

grant truncate on table "public"."conversations" to "service_role";

grant update on table "public"."conversations" to "service_role";

grant delete on table "public"."document_shares" to "anon";

grant insert on table "public"."document_shares" to "anon";

grant references on table "public"."document_shares" to "anon";

grant select on table "public"."document_shares" to "anon";

grant trigger on table "public"."document_shares" to "anon";

grant truncate on table "public"."document_shares" to "anon";

grant update on table "public"."document_shares" to "anon";

grant delete on table "public"."document_shares" to "authenticated";

grant insert on table "public"."document_shares" to "authenticated";

grant references on table "public"."document_shares" to "authenticated";

grant select on table "public"."document_shares" to "authenticated";

grant trigger on table "public"."document_shares" to "authenticated";

grant truncate on table "public"."document_shares" to "authenticated";

grant update on table "public"."document_shares" to "authenticated";

grant delete on table "public"."document_shares" to "service_role";

grant insert on table "public"."document_shares" to "service_role";

grant references on table "public"."document_shares" to "service_role";

grant select on table "public"."document_shares" to "service_role";

grant trigger on table "public"."document_shares" to "service_role";

grant truncate on table "public"."document_shares" to "service_role";

grant update on table "public"."document_shares" to "service_role";

grant delete on table "public"."flashcard_sets" to "anon";

grant insert on table "public"."flashcard_sets" to "anon";

grant references on table "public"."flashcard_sets" to "anon";

grant select on table "public"."flashcard_sets" to "anon";

grant trigger on table "public"."flashcard_sets" to "anon";

grant truncate on table "public"."flashcard_sets" to "anon";

grant update on table "public"."flashcard_sets" to "anon";

grant delete on table "public"."flashcard_sets" to "authenticated";

grant insert on table "public"."flashcard_sets" to "authenticated";

grant references on table "public"."flashcard_sets" to "authenticated";

grant select on table "public"."flashcard_sets" to "authenticated";

grant trigger on table "public"."flashcard_sets" to "authenticated";

grant truncate on table "public"."flashcard_sets" to "authenticated";

grant update on table "public"."flashcard_sets" to "authenticated";

grant delete on table "public"."flashcard_sets" to "service_role";

grant insert on table "public"."flashcard_sets" to "service_role";

grant references on table "public"."flashcard_sets" to "service_role";

grant select on table "public"."flashcard_sets" to "service_role";

grant trigger on table "public"."flashcard_sets" to "service_role";

grant truncate on table "public"."flashcard_sets" to "service_role";

grant update on table "public"."flashcard_sets" to "service_role";

grant delete on table "public"."generated_content" to "anon";

grant insert on table "public"."generated_content" to "anon";

grant references on table "public"."generated_content" to "anon";

grant select on table "public"."generated_content" to "anon";

grant trigger on table "public"."generated_content" to "anon";

grant truncate on table "public"."generated_content" to "anon";

grant update on table "public"."generated_content" to "anon";

grant delete on table "public"."generated_content" to "authenticated";

grant insert on table "public"."generated_content" to "authenticated";

grant references on table "public"."generated_content" to "authenticated";

grant select on table "public"."generated_content" to "authenticated";

grant trigger on table "public"."generated_content" to "authenticated";

grant truncate on table "public"."generated_content" to "authenticated";

grant update on table "public"."generated_content" to "authenticated";

grant delete on table "public"."generated_content" to "service_role";

grant insert on table "public"."generated_content" to "service_role";

grant references on table "public"."generated_content" to "service_role";

grant select on table "public"."generated_content" to "service_role";

grant trigger on table "public"."generated_content" to "service_role";

grant truncate on table "public"."generated_content" to "service_role";

grant update on table "public"."generated_content" to "service_role";

grant delete on table "public"."google_auth_tokens" to "anon";

grant insert on table "public"."google_auth_tokens" to "anon";

grant references on table "public"."google_auth_tokens" to "anon";

grant select on table "public"."google_auth_tokens" to "anon";

grant trigger on table "public"."google_auth_tokens" to "anon";

grant truncate on table "public"."google_auth_tokens" to "anon";

grant update on table "public"."google_auth_tokens" to "anon";

grant delete on table "public"."google_auth_tokens" to "authenticated";

grant insert on table "public"."google_auth_tokens" to "authenticated";

grant references on table "public"."google_auth_tokens" to "authenticated";

grant select on table "public"."google_auth_tokens" to "authenticated";

grant trigger on table "public"."google_auth_tokens" to "authenticated";

grant truncate on table "public"."google_auth_tokens" to "authenticated";

grant update on table "public"."google_auth_tokens" to "authenticated";

grant delete on table "public"."google_auth_tokens" to "service_role";

grant insert on table "public"."google_auth_tokens" to "service_role";

grant references on table "public"."google_auth_tokens" to "service_role";

grant select on table "public"."google_auth_tokens" to "service_role";

grant trigger on table "public"."google_auth_tokens" to "service_role";

grant truncate on table "public"."google_auth_tokens" to "service_role";

grant update on table "public"."google_auth_tokens" to "service_role";

grant delete on table "public"."memories" to "anon";

grant insert on table "public"."memories" to "anon";

grant references on table "public"."memories" to "anon";

grant select on table "public"."memories" to "anon";

grant trigger on table "public"."memories" to "anon";

grant truncate on table "public"."memories" to "anon";

grant update on table "public"."memories" to "anon";

grant delete on table "public"."memories" to "authenticated";

grant insert on table "public"."memories" to "authenticated";

grant references on table "public"."memories" to "authenticated";

grant select on table "public"."memories" to "authenticated";

grant trigger on table "public"."memories" to "authenticated";

grant truncate on table "public"."memories" to "authenticated";

grant update on table "public"."memories" to "authenticated";

grant delete on table "public"."memories" to "service_role";

grant insert on table "public"."memories" to "service_role";

grant references on table "public"."memories" to "service_role";

grant select on table "public"."memories" to "service_role";

grant trigger on table "public"."memories" to "service_role";

grant truncate on table "public"."memories" to "service_role";

grant update on table "public"."memories" to "service_role";

grant delete on table "public"."pdf_chats" to "anon";

grant insert on table "public"."pdf_chats" to "anon";

grant references on table "public"."pdf_chats" to "anon";

grant select on table "public"."pdf_chats" to "anon";

grant trigger on table "public"."pdf_chats" to "anon";

grant truncate on table "public"."pdf_chats" to "anon";

grant update on table "public"."pdf_chats" to "anon";

grant delete on table "public"."pdf_chats" to "authenticated";

grant insert on table "public"."pdf_chats" to "authenticated";

grant references on table "public"."pdf_chats" to "authenticated";

grant select on table "public"."pdf_chats" to "authenticated";

grant trigger on table "public"."pdf_chats" to "authenticated";

grant truncate on table "public"."pdf_chats" to "authenticated";

grant update on table "public"."pdf_chats" to "authenticated";

grant delete on table "public"."pdf_chats" to "service_role";

grant insert on table "public"."pdf_chats" to "service_role";

grant references on table "public"."pdf_chats" to "service_role";

grant select on table "public"."pdf_chats" to "service_role";

grant trigger on table "public"."pdf_chats" to "service_role";

grant truncate on table "public"."pdf_chats" to "service_role";

grant update on table "public"."pdf_chats" to "service_role";

grant delete on table "public"."pdf_embeddings" to "anon";

grant insert on table "public"."pdf_embeddings" to "anon";

grant references on table "public"."pdf_embeddings" to "anon";

grant select on table "public"."pdf_embeddings" to "anon";

grant trigger on table "public"."pdf_embeddings" to "anon";

grant truncate on table "public"."pdf_embeddings" to "anon";

grant update on table "public"."pdf_embeddings" to "anon";

grant delete on table "public"."pdf_embeddings" to "authenticated";

grant insert on table "public"."pdf_embeddings" to "authenticated";

grant references on table "public"."pdf_embeddings" to "authenticated";

grant select on table "public"."pdf_embeddings" to "authenticated";

grant trigger on table "public"."pdf_embeddings" to "authenticated";

grant truncate on table "public"."pdf_embeddings" to "authenticated";

grant update on table "public"."pdf_embeddings" to "authenticated";

grant delete on table "public"."pdf_embeddings" to "service_role";

grant insert on table "public"."pdf_embeddings" to "service_role";

grant references on table "public"."pdf_embeddings" to "service_role";

grant select on table "public"."pdf_embeddings" to "service_role";

grant trigger on table "public"."pdf_embeddings" to "service_role";

grant truncate on table "public"."pdf_embeddings" to "service_role";

grant update on table "public"."pdf_embeddings" to "service_role";

grant delete on table "public"."pdf_passages" to "anon";

grant insert on table "public"."pdf_passages" to "anon";

grant references on table "public"."pdf_passages" to "anon";

grant select on table "public"."pdf_passages" to "anon";

grant trigger on table "public"."pdf_passages" to "anon";

grant truncate on table "public"."pdf_passages" to "anon";

grant update on table "public"."pdf_passages" to "anon";

grant delete on table "public"."pdf_passages" to "authenticated";

grant insert on table "public"."pdf_passages" to "authenticated";

grant references on table "public"."pdf_passages" to "authenticated";

grant select on table "public"."pdf_passages" to "authenticated";

grant trigger on table "public"."pdf_passages" to "authenticated";

grant truncate on table "public"."pdf_passages" to "authenticated";

grant update on table "public"."pdf_passages" to "authenticated";

grant delete on table "public"."pdf_passages" to "service_role";

grant insert on table "public"."pdf_passages" to "service_role";

grant references on table "public"."pdf_passages" to "service_role";

grant select on table "public"."pdf_passages" to "service_role";

grant trigger on table "public"."pdf_passages" to "service_role";

grant truncate on table "public"."pdf_passages" to "service_role";

grant update on table "public"."pdf_passages" to "service_role";

grant delete on table "public"."pdfs" to "anon";

grant insert on table "public"."pdfs" to "anon";

grant references on table "public"."pdfs" to "anon";

grant select on table "public"."pdfs" to "anon";

grant trigger on table "public"."pdfs" to "anon";

grant truncate on table "public"."pdfs" to "anon";

grant update on table "public"."pdfs" to "anon";

grant delete on table "public"."pdfs" to "authenticated";

grant insert on table "public"."pdfs" to "authenticated";

grant references on table "public"."pdfs" to "authenticated";

grant select on table "public"."pdfs" to "authenticated";

grant trigger on table "public"."pdfs" to "authenticated";

grant truncate on table "public"."pdfs" to "authenticated";

grant update on table "public"."pdfs" to "authenticated";

grant delete on table "public"."pdfs" to "service_role";

grant insert on table "public"."pdfs" to "service_role";

grant references on table "public"."pdfs" to "service_role";

grant select on table "public"."pdfs" to "service_role";

grant trigger on table "public"."pdfs" to "service_role";

grant truncate on table "public"."pdfs" to "service_role";

grant update on table "public"."pdfs" to "service_role";

grant delete on table "public"."plans" to "anon";

grant insert on table "public"."plans" to "anon";

grant references on table "public"."plans" to "anon";

grant select on table "public"."plans" to "anon";

grant trigger on table "public"."plans" to "anon";

grant truncate on table "public"."plans" to "anon";

grant update on table "public"."plans" to "anon";

grant delete on table "public"."plans" to "authenticated";

grant insert on table "public"."plans" to "authenticated";

grant references on table "public"."plans" to "authenticated";

grant select on table "public"."plans" to "authenticated";

grant trigger on table "public"."plans" to "authenticated";

grant truncate on table "public"."plans" to "authenticated";

grant update on table "public"."plans" to "authenticated";

grant delete on table "public"."plans" to "service_role";

grant insert on table "public"."plans" to "service_role";

grant references on table "public"."plans" to "service_role";

grant select on table "public"."plans" to "service_role";

grant trigger on table "public"."plans" to "service_role";

grant truncate on table "public"."plans" to "service_role";

grant update on table "public"."plans" to "service_role";

grant delete on table "public"."quiz_attempts" to "anon";

grant insert on table "public"."quiz_attempts" to "anon";

grant references on table "public"."quiz_attempts" to "anon";

grant select on table "public"."quiz_attempts" to "anon";

grant trigger on table "public"."quiz_attempts" to "anon";

grant truncate on table "public"."quiz_attempts" to "anon";

grant update on table "public"."quiz_attempts" to "anon";

grant delete on table "public"."quiz_attempts" to "authenticated";

grant insert on table "public"."quiz_attempts" to "authenticated";

grant references on table "public"."quiz_attempts" to "authenticated";

grant select on table "public"."quiz_attempts" to "authenticated";

grant trigger on table "public"."quiz_attempts" to "authenticated";

grant truncate on table "public"."quiz_attempts" to "authenticated";

grant update on table "public"."quiz_attempts" to "authenticated";

grant delete on table "public"."quiz_attempts" to "service_role";

grant insert on table "public"."quiz_attempts" to "service_role";

grant references on table "public"."quiz_attempts" to "service_role";

grant select on table "public"."quiz_attempts" to "service_role";

grant trigger on table "public"."quiz_attempts" to "service_role";

grant truncate on table "public"."quiz_attempts" to "service_role";

grant update on table "public"."quiz_attempts" to "service_role";

grant delete on table "public"."quiz_sets" to "anon";

grant insert on table "public"."quiz_sets" to "anon";

grant references on table "public"."quiz_sets" to "anon";

grant select on table "public"."quiz_sets" to "anon";

grant trigger on table "public"."quiz_sets" to "anon";

grant truncate on table "public"."quiz_sets" to "anon";

grant update on table "public"."quiz_sets" to "anon";

grant delete on table "public"."quiz_sets" to "authenticated";

grant insert on table "public"."quiz_sets" to "authenticated";

grant references on table "public"."quiz_sets" to "authenticated";

grant select on table "public"."quiz_sets" to "authenticated";

grant trigger on table "public"."quiz_sets" to "authenticated";

grant truncate on table "public"."quiz_sets" to "authenticated";

grant update on table "public"."quiz_sets" to "authenticated";

grant delete on table "public"."quiz_sets" to "service_role";

grant insert on table "public"."quiz_sets" to "service_role";

grant references on table "public"."quiz_sets" to "service_role";

grant select on table "public"."quiz_sets" to "service_role";

grant trigger on table "public"."quiz_sets" to "service_role";

grant truncate on table "public"."quiz_sets" to "service_role";

grant update on table "public"."quiz_sets" to "service_role";

grant delete on table "public"."recordings" to "anon";

grant insert on table "public"."recordings" to "anon";

grant references on table "public"."recordings" to "anon";

grant select on table "public"."recordings" to "anon";

grant trigger on table "public"."recordings" to "anon";

grant truncate on table "public"."recordings" to "anon";

grant update on table "public"."recordings" to "anon";

grant delete on table "public"."recordings" to "authenticated";

grant insert on table "public"."recordings" to "authenticated";

grant references on table "public"."recordings" to "authenticated";

grant select on table "public"."recordings" to "authenticated";

grant trigger on table "public"."recordings" to "authenticated";

grant truncate on table "public"."recordings" to "authenticated";

grant update on table "public"."recordings" to "authenticated";

grant delete on table "public"."recordings" to "service_role";

grant insert on table "public"."recordings" to "service_role";

grant references on table "public"."recordings" to "service_role";

grant select on table "public"."recordings" to "service_role";

grant trigger on table "public"."recordings" to "service_role";

grant truncate on table "public"."recordings" to "service_role";

grant update on table "public"."recordings" to "service_role";

grant delete on table "public"."referral" to "anon";

grant insert on table "public"."referral" to "anon";

grant references on table "public"."referral" to "anon";

grant select on table "public"."referral" to "anon";

grant trigger on table "public"."referral" to "anon";

grant truncate on table "public"."referral" to "anon";

grant update on table "public"."referral" to "anon";

grant delete on table "public"."referral" to "authenticated";

grant insert on table "public"."referral" to "authenticated";

grant references on table "public"."referral" to "authenticated";

grant select on table "public"."referral" to "authenticated";

grant trigger on table "public"."referral" to "authenticated";

grant truncate on table "public"."referral" to "authenticated";

grant update on table "public"."referral" to "authenticated";

grant delete on table "public"."referral" to "service_role";

grant insert on table "public"."referral" to "service_role";

grant references on table "public"."referral" to "service_role";

grant select on table "public"."referral" to "service_role";

grant trigger on table "public"."referral" to "service_role";

grant truncate on table "public"."referral" to "service_role";

grant update on table "public"."referral" to "service_role";

grant delete on table "public"."referral_codes" to "anon";

grant insert on table "public"."referral_codes" to "anon";

grant references on table "public"."referral_codes" to "anon";

grant select on table "public"."referral_codes" to "anon";

grant trigger on table "public"."referral_codes" to "anon";

grant truncate on table "public"."referral_codes" to "anon";

grant update on table "public"."referral_codes" to "anon";

grant delete on table "public"."referral_codes" to "authenticated";

grant insert on table "public"."referral_codes" to "authenticated";

grant references on table "public"."referral_codes" to "authenticated";

grant select on table "public"."referral_codes" to "authenticated";

grant trigger on table "public"."referral_codes" to "authenticated";

grant truncate on table "public"."referral_codes" to "authenticated";

grant update on table "public"."referral_codes" to "authenticated";

grant delete on table "public"."referral_codes" to "service_role";

grant insert on table "public"."referral_codes" to "service_role";

grant references on table "public"."referral_codes" to "service_role";

grant select on table "public"."referral_codes" to "service_role";

grant trigger on table "public"."referral_codes" to "service_role";

grant truncate on table "public"."referral_codes" to "service_role";

grant update on table "public"."referral_codes" to "service_role";

grant delete on table "public"."reviews" to "anon";

grant insert on table "public"."reviews" to "anon";

grant references on table "public"."reviews" to "anon";

grant select on table "public"."reviews" to "anon";

grant trigger on table "public"."reviews" to "anon";

grant truncate on table "public"."reviews" to "anon";

grant update on table "public"."reviews" to "anon";

grant delete on table "public"."reviews" to "authenticated";

grant insert on table "public"."reviews" to "authenticated";

grant references on table "public"."reviews" to "authenticated";

grant select on table "public"."reviews" to "authenticated";

grant trigger on table "public"."reviews" to "authenticated";

grant truncate on table "public"."reviews" to "authenticated";

grant update on table "public"."reviews" to "authenticated";

grant delete on table "public"."reviews" to "service_role";

grant insert on table "public"."reviews" to "service_role";

grant references on table "public"."reviews" to "service_role";

grant select on table "public"."reviews" to "service_role";

grant trigger on table "public"."reviews" to "service_role";

grant truncate on table "public"."reviews" to "service_role";

grant update on table "public"."reviews" to "service_role";

grant delete on table "public"."share_audit_logs" to "anon";

grant insert on table "public"."share_audit_logs" to "anon";

grant references on table "public"."share_audit_logs" to "anon";

grant select on table "public"."share_audit_logs" to "anon";

grant trigger on table "public"."share_audit_logs" to "anon";

grant truncate on table "public"."share_audit_logs" to "anon";

grant update on table "public"."share_audit_logs" to "anon";

grant delete on table "public"."share_audit_logs" to "authenticated";

grant insert on table "public"."share_audit_logs" to "authenticated";

grant references on table "public"."share_audit_logs" to "authenticated";

grant select on table "public"."share_audit_logs" to "authenticated";

grant trigger on table "public"."share_audit_logs" to "authenticated";

grant truncate on table "public"."share_audit_logs" to "authenticated";

grant update on table "public"."share_audit_logs" to "authenticated";

grant delete on table "public"."share_audit_logs" to "service_role";

grant insert on table "public"."share_audit_logs" to "service_role";

grant references on table "public"."share_audit_logs" to "service_role";

grant select on table "public"."share_audit_logs" to "service_role";

grant trigger on table "public"."share_audit_logs" to "service_role";

grant truncate on table "public"."share_audit_logs" to "service_role";

grant update on table "public"."share_audit_logs" to "service_role";

grant delete on table "public"."share_link_access" to "anon";

grant insert on table "public"."share_link_access" to "anon";

grant references on table "public"."share_link_access" to "anon";

grant select on table "public"."share_link_access" to "anon";

grant trigger on table "public"."share_link_access" to "anon";

grant truncate on table "public"."share_link_access" to "anon";

grant update on table "public"."share_link_access" to "anon";

grant delete on table "public"."share_link_access" to "authenticated";

grant insert on table "public"."share_link_access" to "authenticated";

grant references on table "public"."share_link_access" to "authenticated";

grant select on table "public"."share_link_access" to "authenticated";

grant trigger on table "public"."share_link_access" to "authenticated";

grant truncate on table "public"."share_link_access" to "authenticated";

grant update on table "public"."share_link_access" to "authenticated";

grant delete on table "public"."share_link_access" to "service_role";

grant insert on table "public"."share_link_access" to "service_role";

grant references on table "public"."share_link_access" to "service_role";

grant select on table "public"."share_link_access" to "service_role";

grant trigger on table "public"."share_link_access" to "service_role";

grant truncate on table "public"."share_link_access" to "service_role";

grant update on table "public"."share_link_access" to "service_role";

grant delete on table "public"."share_links" to "anon";

grant insert on table "public"."share_links" to "anon";

grant references on table "public"."share_links" to "anon";

grant select on table "public"."share_links" to "anon";

grant trigger on table "public"."share_links" to "anon";

grant truncate on table "public"."share_links" to "anon";

grant update on table "public"."share_links" to "anon";

grant delete on table "public"."share_links" to "authenticated";

grant insert on table "public"."share_links" to "authenticated";

grant references on table "public"."share_links" to "authenticated";

grant select on table "public"."share_links" to "authenticated";

grant trigger on table "public"."share_links" to "authenticated";

grant truncate on table "public"."share_links" to "authenticated";

grant update on table "public"."share_links" to "authenticated";

grant delete on table "public"."share_links" to "service_role";

grant insert on table "public"."share_links" to "service_role";

grant references on table "public"."share_links" to "service_role";

grant select on table "public"."share_links" to "service_role";

grant trigger on table "public"."share_links" to "service_role";

grant truncate on table "public"."share_links" to "service_role";

grant update on table "public"."share_links" to "service_role";

grant delete on table "public"."space_shares" to "anon";

grant insert on table "public"."space_shares" to "anon";

grant references on table "public"."space_shares" to "anon";

grant select on table "public"."space_shares" to "anon";

grant trigger on table "public"."space_shares" to "anon";

grant truncate on table "public"."space_shares" to "anon";

grant update on table "public"."space_shares" to "anon";

grant delete on table "public"."space_shares" to "authenticated";

grant insert on table "public"."space_shares" to "authenticated";

grant references on table "public"."space_shares" to "authenticated";

grant select on table "public"."space_shares" to "authenticated";

grant trigger on table "public"."space_shares" to "authenticated";

grant truncate on table "public"."space_shares" to "authenticated";

grant update on table "public"."space_shares" to "authenticated";

grant delete on table "public"."space_shares" to "service_role";

grant insert on table "public"."space_shares" to "service_role";

grant references on table "public"."space_shares" to "service_role";

grant select on table "public"."space_shares" to "service_role";

grant trigger on table "public"."space_shares" to "service_role";

grant truncate on table "public"."space_shares" to "service_role";

grant update on table "public"."space_shares" to "service_role";

grant delete on table "public"."spaces" to "anon";

grant insert on table "public"."spaces" to "anon";

grant references on table "public"."spaces" to "anon";

grant select on table "public"."spaces" to "anon";

grant trigger on table "public"."spaces" to "anon";

grant truncate on table "public"."spaces" to "anon";

grant update on table "public"."spaces" to "anon";

grant delete on table "public"."spaces" to "authenticated";

grant insert on table "public"."spaces" to "authenticated";

grant references on table "public"."spaces" to "authenticated";

grant select on table "public"."spaces" to "authenticated";

grant trigger on table "public"."spaces" to "authenticated";

grant truncate on table "public"."spaces" to "authenticated";

grant update on table "public"."spaces" to "authenticated";

grant delete on table "public"."spaces" to "service_role";

grant insert on table "public"."spaces" to "service_role";

grant references on table "public"."spaces" to "service_role";

grant select on table "public"."spaces" to "service_role";

grant trigger on table "public"."spaces" to "service_role";

grant truncate on table "public"."spaces" to "service_role";

grant update on table "public"."spaces" to "service_role";

grant delete on table "public"."subscriptions" to "anon";

grant insert on table "public"."subscriptions" to "anon";

grant references on table "public"."subscriptions" to "anon";

grant select on table "public"."subscriptions" to "anon";

grant trigger on table "public"."subscriptions" to "anon";

grant truncate on table "public"."subscriptions" to "anon";

grant update on table "public"."subscriptions" to "anon";

grant delete on table "public"."subscriptions" to "authenticated";

grant insert on table "public"."subscriptions" to "authenticated";

grant references on table "public"."subscriptions" to "authenticated";

grant select on table "public"."subscriptions" to "authenticated";

grant trigger on table "public"."subscriptions" to "authenticated";

grant truncate on table "public"."subscriptions" to "authenticated";

grant update on table "public"."subscriptions" to "authenticated";

grant delete on table "public"."subscriptions" to "service_role";

grant insert on table "public"."subscriptions" to "service_role";

grant references on table "public"."subscriptions" to "service_role";

grant select on table "public"."subscriptions" to "service_role";

grant trigger on table "public"."subscriptions" to "service_role";

grant truncate on table "public"."subscriptions" to "service_role";

grant update on table "public"."subscriptions" to "service_role";

grant delete on table "public"."user_chat_sessions" to "anon";

grant insert on table "public"."user_chat_sessions" to "anon";

grant references on table "public"."user_chat_sessions" to "anon";

grant select on table "public"."user_chat_sessions" to "anon";

grant trigger on table "public"."user_chat_sessions" to "anon";

grant truncate on table "public"."user_chat_sessions" to "anon";

grant update on table "public"."user_chat_sessions" to "anon";

grant delete on table "public"."user_chat_sessions" to "authenticated";

grant insert on table "public"."user_chat_sessions" to "authenticated";

grant references on table "public"."user_chat_sessions" to "authenticated";

grant select on table "public"."user_chat_sessions" to "authenticated";

grant trigger on table "public"."user_chat_sessions" to "authenticated";

grant truncate on table "public"."user_chat_sessions" to "authenticated";

grant update on table "public"."user_chat_sessions" to "authenticated";

grant delete on table "public"."user_chat_sessions" to "service_role";

grant insert on table "public"."user_chat_sessions" to "service_role";

grant references on table "public"."user_chat_sessions" to "service_role";

grant select on table "public"."user_chat_sessions" to "service_role";

grant trigger on table "public"."user_chat_sessions" to "service_role";

grant truncate on table "public"."user_chat_sessions" to "service_role";

grant update on table "public"."user_chat_sessions" to "service_role";

grant delete on table "public"."user_list" to "anon";

grant insert on table "public"."user_list" to "anon";

grant references on table "public"."user_list" to "anon";

grant select on table "public"."user_list" to "anon";

grant trigger on table "public"."user_list" to "anon";

grant truncate on table "public"."user_list" to "anon";

grant update on table "public"."user_list" to "anon";

grant delete on table "public"."user_list" to "authenticated";

grant insert on table "public"."user_list" to "authenticated";

grant references on table "public"."user_list" to "authenticated";

grant select on table "public"."user_list" to "authenticated";

grant trigger on table "public"."user_list" to "authenticated";

grant truncate on table "public"."user_list" to "authenticated";

grant update on table "public"."user_list" to "authenticated";

grant delete on table "public"."user_list" to "service_role";

grant insert on table "public"."user_list" to "service_role";

grant references on table "public"."user_list" to "service_role";

grant select on table "public"."user_list" to "service_role";

grant trigger on table "public"."user_list" to "service_role";

grant truncate on table "public"."user_list" to "service_role";

grant update on table "public"."user_list" to "service_role";

grant delete on table "public"."user_notes" to "anon";

grant insert on table "public"."user_notes" to "anon";

grant references on table "public"."user_notes" to "anon";

grant select on table "public"."user_notes" to "anon";

grant trigger on table "public"."user_notes" to "anon";

grant truncate on table "public"."user_notes" to "anon";

grant update on table "public"."user_notes" to "anon";

grant delete on table "public"."user_notes" to "authenticated";

grant insert on table "public"."user_notes" to "authenticated";

grant references on table "public"."user_notes" to "authenticated";

grant select on table "public"."user_notes" to "authenticated";

grant trigger on table "public"."user_notes" to "authenticated";

grant truncate on table "public"."user_notes" to "authenticated";

grant update on table "public"."user_notes" to "authenticated";

grant delete on table "public"."user_notes" to "service_role";

grant insert on table "public"."user_notes" to "service_role";

grant references on table "public"."user_notes" to "service_role";

grant select on table "public"."user_notes" to "service_role";

grant trigger on table "public"."user_notes" to "service_role";

grant truncate on table "public"."user_notes" to "service_role";

grant update on table "public"."user_notes" to "service_role";

grant delete on table "public"."user_settings" to "anon";

grant insert on table "public"."user_settings" to "anon";

grant references on table "public"."user_settings" to "anon";

grant select on table "public"."user_settings" to "anon";

grant trigger on table "public"."user_settings" to "anon";

grant truncate on table "public"."user_settings" to "anon";

grant update on table "public"."user_settings" to "anon";

grant delete on table "public"."user_settings" to "authenticated";

grant insert on table "public"."user_settings" to "authenticated";

grant references on table "public"."user_settings" to "authenticated";

grant select on table "public"."user_settings" to "authenticated";

grant trigger on table "public"."user_settings" to "authenticated";

grant truncate on table "public"."user_settings" to "authenticated";

grant update on table "public"."user_settings" to "authenticated";

grant delete on table "public"."user_settings" to "service_role";

grant insert on table "public"."user_settings" to "service_role";

grant references on table "public"."user_settings" to "service_role";

grant select on table "public"."user_settings" to "service_role";

grant trigger on table "public"."user_settings" to "service_role";

grant truncate on table "public"."user_settings" to "service_role";

grant update on table "public"."user_settings" to "service_role";

grant delete on table "public"."writing_blocks" to "anon";

grant insert on table "public"."writing_blocks" to "anon";

grant references on table "public"."writing_blocks" to "anon";

grant select on table "public"."writing_blocks" to "anon";

grant trigger on table "public"."writing_blocks" to "anon";

grant truncate on table "public"."writing_blocks" to "anon";

grant update on table "public"."writing_blocks" to "anon";

grant delete on table "public"."writing_blocks" to "authenticated";

grant insert on table "public"."writing_blocks" to "authenticated";

grant references on table "public"."writing_blocks" to "authenticated";

grant select on table "public"."writing_blocks" to "authenticated";

grant trigger on table "public"."writing_blocks" to "authenticated";

grant truncate on table "public"."writing_blocks" to "authenticated";

grant update on table "public"."writing_blocks" to "authenticated";

grant delete on table "public"."writing_blocks" to "service_role";

grant insert on table "public"."writing_blocks" to "service_role";

grant references on table "public"."writing_blocks" to "service_role";

grant select on table "public"."writing_blocks" to "service_role";

grant trigger on table "public"."writing_blocks" to "service_role";

grant truncate on table "public"."writing_blocks" to "service_role";

grant update on table "public"."writing_blocks" to "service_role";

grant delete on table "public"."writing_references" to "anon";

grant insert on table "public"."writing_references" to "anon";

grant references on table "public"."writing_references" to "anon";

grant select on table "public"."writing_references" to "anon";

grant trigger on table "public"."writing_references" to "anon";

grant truncate on table "public"."writing_references" to "anon";

grant update on table "public"."writing_references" to "anon";

grant delete on table "public"."writing_references" to "authenticated";

grant insert on table "public"."writing_references" to "authenticated";

grant references on table "public"."writing_references" to "authenticated";

grant select on table "public"."writing_references" to "authenticated";

grant trigger on table "public"."writing_references" to "authenticated";

grant truncate on table "public"."writing_references" to "authenticated";

grant update on table "public"."writing_references" to "authenticated";

grant delete on table "public"."writing_references" to "service_role";

grant insert on table "public"."writing_references" to "service_role";

grant references on table "public"."writing_references" to "service_role";

grant select on table "public"."writing_references" to "service_role";

grant trigger on table "public"."writing_references" to "service_role";

grant truncate on table "public"."writing_references" to "service_role";

grant update on table "public"."writing_references" to "service_role";

grant delete on table "public"."writing_reflections" to "anon";

grant insert on table "public"."writing_reflections" to "anon";

grant references on table "public"."writing_reflections" to "anon";

grant select on table "public"."writing_reflections" to "anon";

grant trigger on table "public"."writing_reflections" to "anon";

grant truncate on table "public"."writing_reflections" to "anon";

grant update on table "public"."writing_reflections" to "anon";

grant delete on table "public"."writing_reflections" to "authenticated";

grant insert on table "public"."writing_reflections" to "authenticated";

grant references on table "public"."writing_reflections" to "authenticated";

grant select on table "public"."writing_reflections" to "authenticated";

grant trigger on table "public"."writing_reflections" to "authenticated";

grant truncate on table "public"."writing_reflections" to "authenticated";

grant update on table "public"."writing_reflections" to "authenticated";

grant delete on table "public"."writing_reflections" to "service_role";

grant insert on table "public"."writing_reflections" to "service_role";

grant references on table "public"."writing_reflections" to "service_role";

grant select on table "public"."writing_reflections" to "service_role";

grant trigger on table "public"."writing_reflections" to "service_role";

grant truncate on table "public"."writing_reflections" to "service_role";

grant update on table "public"."writing_reflections" to "service_role";

grant delete on table "public"."writing_section_plans" to "anon";

grant insert on table "public"."writing_section_plans" to "anon";

grant references on table "public"."writing_section_plans" to "anon";

grant select on table "public"."writing_section_plans" to "anon";

grant trigger on table "public"."writing_section_plans" to "anon";

grant truncate on table "public"."writing_section_plans" to "anon";

grant update on table "public"."writing_section_plans" to "anon";

grant delete on table "public"."writing_section_plans" to "authenticated";

grant insert on table "public"."writing_section_plans" to "authenticated";

grant references on table "public"."writing_section_plans" to "authenticated";

grant select on table "public"."writing_section_plans" to "authenticated";

grant trigger on table "public"."writing_section_plans" to "authenticated";

grant truncate on table "public"."writing_section_plans" to "authenticated";

grant update on table "public"."writing_section_plans" to "authenticated";

grant delete on table "public"."writing_section_plans" to "service_role";

grant insert on table "public"."writing_section_plans" to "service_role";

grant references on table "public"."writing_section_plans" to "service_role";

grant select on table "public"."writing_section_plans" to "service_role";

grant trigger on table "public"."writing_section_plans" to "service_role";

grant truncate on table "public"."writing_section_plans" to "service_role";

grant update on table "public"."writing_section_plans" to "service_role";

grant delete on table "public"."writing_sessions" to "anon";

grant insert on table "public"."writing_sessions" to "anon";

grant references on table "public"."writing_sessions" to "anon";

grant select on table "public"."writing_sessions" to "anon";

grant trigger on table "public"."writing_sessions" to "anon";

grant truncate on table "public"."writing_sessions" to "anon";

grant update on table "public"."writing_sessions" to "anon";

grant delete on table "public"."writing_sessions" to "authenticated";

grant insert on table "public"."writing_sessions" to "authenticated";

grant references on table "public"."writing_sessions" to "authenticated";

grant select on table "public"."writing_sessions" to "authenticated";

grant trigger on table "public"."writing_sessions" to "authenticated";

grant truncate on table "public"."writing_sessions" to "authenticated";

grant update on table "public"."writing_sessions" to "authenticated";

grant delete on table "public"."writing_sessions" to "service_role";

grant insert on table "public"."writing_sessions" to "service_role";

grant references on table "public"."writing_sessions" to "service_role";

grant select on table "public"."writing_sessions" to "service_role";

grant trigger on table "public"."writing_sessions" to "service_role";

grant truncate on table "public"."writing_sessions" to "service_role";

grant update on table "public"."writing_sessions" to "service_role";

grant delete on table "public"."writing_structures" to "anon";

grant insert on table "public"."writing_structures" to "anon";

grant references on table "public"."writing_structures" to "anon";

grant select on table "public"."writing_structures" to "anon";

grant trigger on table "public"."writing_structures" to "anon";

grant truncate on table "public"."writing_structures" to "anon";

grant update on table "public"."writing_structures" to "anon";

grant delete on table "public"."writing_structures" to "authenticated";

grant insert on table "public"."writing_structures" to "authenticated";

grant references on table "public"."writing_structures" to "authenticated";

grant select on table "public"."writing_structures" to "authenticated";

grant trigger on table "public"."writing_structures" to "authenticated";

grant truncate on table "public"."writing_structures" to "authenticated";

grant update on table "public"."writing_structures" to "authenticated";

grant delete on table "public"."writing_structures" to "service_role";

grant insert on table "public"."writing_structures" to "service_role";

grant references on table "public"."writing_structures" to "service_role";

grant select on table "public"."writing_structures" to "service_role";

grant trigger on table "public"."writing_structures" to "service_role";

grant truncate on table "public"."writing_structures" to "service_role";

grant update on table "public"."writing_structures" to "service_role";

grant delete on table "public"."writing_user_sources" to "anon";

grant insert on table "public"."writing_user_sources" to "anon";

grant references on table "public"."writing_user_sources" to "anon";

grant select on table "public"."writing_user_sources" to "anon";

grant trigger on table "public"."writing_user_sources" to "anon";

grant truncate on table "public"."writing_user_sources" to "anon";

grant update on table "public"."writing_user_sources" to "anon";

grant delete on table "public"."writing_user_sources" to "authenticated";

grant insert on table "public"."writing_user_sources" to "authenticated";

grant references on table "public"."writing_user_sources" to "authenticated";

grant select on table "public"."writing_user_sources" to "authenticated";

grant trigger on table "public"."writing_user_sources" to "authenticated";

grant truncate on table "public"."writing_user_sources" to "authenticated";

grant update on table "public"."writing_user_sources" to "authenticated";

grant delete on table "public"."writing_user_sources" to "service_role";

grant insert on table "public"."writing_user_sources" to "service_role";

grant references on table "public"."writing_user_sources" to "service_role";

grant select on table "public"."writing_user_sources" to "service_role";

grant trigger on table "public"."writing_user_sources" to "service_role";

grant truncate on table "public"."writing_user_sources" to "service_role";

grant update on table "public"."writing_user_sources" to "service_role";

grant delete on table "public"."yts" to "anon";

grant insert on table "public"."yts" to "anon";

grant references on table "public"."yts" to "anon";

grant select on table "public"."yts" to "anon";

grant trigger on table "public"."yts" to "anon";

grant truncate on table "public"."yts" to "anon";

grant update on table "public"."yts" to "anon";

grant delete on table "public"."yts" to "authenticated";

grant insert on table "public"."yts" to "authenticated";

grant references on table "public"."yts" to "authenticated";

grant select on table "public"."yts" to "authenticated";

grant trigger on table "public"."yts" to "authenticated";

grant truncate on table "public"."yts" to "authenticated";

grant update on table "public"."yts" to "authenticated";

grant delete on table "public"."yts" to "service_role";

grant insert on table "public"."yts" to "service_role";

grant references on table "public"."yts" to "service_role";

grant select on table "public"."yts" to "service_role";

grant trigger on table "public"."yts" to "service_role";

grant truncate on table "public"."yts" to "service_role";

grant update on table "public"."yts" to "service_role";


  create policy "Users can delete own settings"
  on "public"."user_settings"
  as permissive
  for delete
  to public
using ((auth.uid() = user_id));



  create policy "Users can insert own settings"
  on "public"."user_settings"
  as permissive
  for insert
  to public
with check ((auth.uid() = user_id));



  create policy "Users can insert their own settings"
  on "public"."user_settings"
  as permissive
  for insert
  to public
with check ((auth.uid() = user_id));



  create policy "Users can update own settings"
  on "public"."user_settings"
  as permissive
  for update
  to public
using ((auth.uid() = user_id));



  create policy "Users can update their own settings"
  on "public"."user_settings"
  as permissive
  for update
  to public
using ((auth.uid() = user_id));



  create policy "Users can view own settings"
  on "public"."user_settings"
  as permissive
  for select
  to public
using ((auth.uid() = user_id));



  create policy "Users can view their own settings"
  on "public"."user_settings"
  as permissive
  for select
  to public
using ((auth.uid() = user_id));



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



  create policy "Users can delete their own sources"
  on "public"."writing_user_sources"
  as permissive
  for delete
  to authenticated
using ((user_id = auth.uid()));



  create policy "Users can insert sources for their sessions"
  on "public"."writing_user_sources"
  as permissive
  for insert
  to authenticated
with check (((user_id = auth.uid()) AND (session_id IN ( SELECT writing_sessions.id
   FROM public.writing_sessions
  WHERE (writing_sessions.user_id = auth.uid())))));



  create policy "Users can read their own sources"
  on "public"."writing_user_sources"
  as permissive
  for select
  to authenticated
using ((user_id = auth.uid()));



  create policy "Users can update their own sources"
  on "public"."writing_user_sources"
  as permissive
  for update
  to authenticated
using ((user_id = auth.uid()))
with check ((user_id = auth.uid()));


CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON public.user_settings FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_writing_sessions_updated_at_trigger BEFORE UPDATE ON public.writing_sessions FOR EACH ROW EXECUTE FUNCTION public.update_writing_sessions_updated_at();

CREATE TRIGGER update_writing_user_sources_updated_at BEFORE UPDATE ON public.writing_user_sources FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();



