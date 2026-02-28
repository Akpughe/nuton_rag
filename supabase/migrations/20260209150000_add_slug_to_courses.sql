-- Migration: Add slug column to courses table
-- URL-friendly identifier generated from course title, e.g. "exploring-modern-ai"

alter table "public"."courses"
    add column "slug" text;

-- Create unique index for slug lookups
create unique index idx_courses_slug on "public"."courses" ("slug");
