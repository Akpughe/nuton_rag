-- Widen resource_type check constraints to include 'course'
ALTER TABLE share_links DROP CONSTRAINT IF EXISTS share_links_resource_type_check;
ALTER TABLE share_links ADD CONSTRAINT share_links_resource_type_check
  CHECK (resource_type = ANY (ARRAY['space'::text, 'pdf'::text, 'youtube'::text, 'course'::text]));

ALTER TABLE share_audit_logs DROP CONSTRAINT IF EXISTS share_audit_logs_resource_type_check;
ALTER TABLE share_audit_logs ADD CONSTRAINT share_audit_logs_resource_type_check
  CHECK (resource_type = ANY (ARRAY['space'::text, 'pdf'::text, 'youtube'::text, 'course'::text]));
