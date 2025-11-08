# Quick Reference: User-Based Data Isolation Implementation

## What Changed?

All flashcard and quiz generation now supports **user-based ownership** to ensure proper data isolation in shared spaces.

## Request Format (NEW)

### Generate Flashcards
```json
POST /generate_flashcards
{
  "document_id": "uuid",
  "space_id": "uuid",
  "user_id": "uuid-of-current-user",  // ← NEW REQUIRED
  "num_questions": 30
}
```

### Regenerate Flashcards
```json
POST /regenerate_flashcards
{
  "document_id": "uuid",
  "space_id": "uuid",
  "user_id": "uuid-of-current-user",  // ← NEW REQUIRED
  "num_questions": 30
}
```

### Generate Quiz
```json
POST /generate_quiz
{
  "document_id": "uuid",
  "space_id": "uuid",
  "user_id": "uuid-of-current-user",  // ← NEW REQUIRED
  "question_type": "both",
  "num_questions": 30
}
```

### Regenerate Quiz
```json
POST /regenerate_quiz
{
  "document_id": "uuid",
  "space_id": "uuid",
  "user_id": "uuid-of-current-user",  // ← NEW REQUIRED
  "question_type": "both",
  "num_questions": 30
}
```

## Key Fields in Response

### Created Flashcard/Quiz Set
```json
{
  "status": "success",
  "flashcards": [
    {
      "set_id": 1,
      "cards": [...]
    }
  ]
}
```

### Database Record
```sql
-- flashcard_sets table
id              UUID
content_id      UUID
flashcards      JSONB
set_number      INT
created_by      UUID          ← User who created it
is_shared       BOOLEAN       ← Shared (true) or Private (false)
created_at      TIMESTAMP

-- quiz_sets table (same)
```

## Logic at a Glance

```
if user_id == space_owner_id:
    is_shared = true   ✓ Visible to all space members
else:
    is_shared = false  ✗ Private to creator only
```

## Database Queries

### Check Ownership for Content
```sql
-- Get space owner
SELECT user_id, created_by FROM spaces WHERE id = '{space_id}';

-- Check if user is owner
SELECT ('{user_id}' = spaces.user_id) as is_owner
FROM spaces
WHERE id = '{space_id}';
```

### View All Flashcards in Space
```sql
-- Shared ones (all members see)
SELECT * FROM flashcard_sets 
WHERE content_id IN (
  SELECT id FROM generated_content WHERE space_id = '{space_id}'
) AND is_shared = true;

-- User's private ones
SELECT * FROM flashcard_sets
WHERE created_by = '{user_id}' AND is_shared = false;
```

## Frontend Integration Example

```typescript
// Get current user ID from auth
const userId = currentUser.id;

// Call endpoint
const response = await fetch('/generate_flashcards', {
  method: 'POST',
  body: JSON.stringify({
    document_id: documentId,
    space_id: spaceId,
    user_id: userId,           // ← ALWAYS PASS THIS
    num_questions: 30
  })
});
```

## Error Scenarios

### Missing user_id
```
Status: 400
{
  "error": "field required",
  "detail": [{
    "loc": ["body", "user_id"],
    "msg": "field required",
    "type": "value_error.missing"
  }]
}
```

### User Not in Space
- Request still processes (no permission check at RAG level)
- Content is marked as private (`is_shared = false`)
- Only that user sees it

### Database Error
```
Status: 500
{
  "error": "Failed to save quiz: ..."
}
```

## Logging

Watch for these in logs:

```
✓ "Generating flashcards for document {doc_id}, space_id: {space_id}, user_id: {user_id}"
✓ "Ownership check: user={user_id}, owner={owner_id}, is_owner={bool}"
✓ "Creating new flashcard set (..., created_by: {user_id}, is_shared: {bool})"
```

## Common Issues

### Issue: Content not visible to other users
**Cause**: `is_shared = false` (creator wasn't space owner)
**Solution**: Space owner needs to create the content

### Issue: "user_id not provided"
**Cause**: Frontend not sending user_id in request
**Solution**: Always include `user_id` in request body

### Issue: Duplicate content across sets
**Cause**: Different users creating content
**Solution**: Expected behavior - each user gets their private set

## Rollback Plan (if needed)

If issues arise:
1. Stop accepting `user_id` in requests (remove from Pydantic models)
2. Default all `is_shared = true` in database
3. Clear any private (`is_shared = false`) entries if not wanted
4. Revert the code changes

## Support Commands

### Check if migration worked
```sql
SELECT COUNT(*) as total, 
       SUM(CASE WHEN is_shared THEN 1 ELSE 0 END) as shared,
       SUM(CASE WHEN NOT is_shared THEN 1 ELSE 0 END) as private
FROM flashcard_sets;
```

### See content creator
```sql
SELECT id, set_number, created_by, is_shared, created_at
FROM flashcard_sets
WHERE content_id = '{content_id}'
ORDER BY created_at DESC;
```

### Find private sets
```sql
SELECT fs.id, fs.set_number, fs.created_by, u.email
FROM flashcard_sets fs
LEFT JOIN auth.users u ON fs.created_by = u.id
WHERE is_shared = false
LIMIT 20;
```

## Version Info

- **Deployed Date**: [To be filled]
- **Files Modified**: 4
- **Lines Changed**: +84
- **Breaking Changes**: None (backward compatible)
- **Migration Required**: Yes (populate existing records)

---

**Questions?** See `ENDPOINT_DOCUMENTATION.md` for detailed API specs or `IMPLEMENTATION_SUMMARY.md` for technical details.

