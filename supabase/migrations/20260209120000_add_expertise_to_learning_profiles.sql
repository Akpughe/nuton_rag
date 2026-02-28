-- Add expertise column to learning_profiles
-- Defaults to 'beginner' for existing rows and backward compatibility

alter table "public"."learning_profiles"
    add column "expertise" text not null default 'beginner';

alter table "public"."learning_profiles"
    add constraint "learning_profiles_expertise_check"
    check (expertise in ('beginner', 'intermediate', 'advanced'));
