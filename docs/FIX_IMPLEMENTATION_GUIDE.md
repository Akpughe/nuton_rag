# 🚀 Flashcard/Quiz Visibility Fix - Implementation Guide

## 📌 Quick Summary

Fixed three critical issues:
1. ✅ **`created_by` field empty** → Now always populated
2. ✅ **No visibility filtering** → New functions added to filter by ownership
3. ✅ **Old records orphaned** → Backfill script provided

---

## 🔧 What Was Changed

### 1. File: `supabase_client.py`

#### Updated Functions:
- **`insert_flashcard_set()`** (Lines 77-147)
  - Fixed conditional logic: `if created_by:` → `if created_by is not None:`
  - Always sets `is_shared` explicitly
  - Added enhanced logging

- **`insert_quiz_set()`** (Lines 311-341)
  - Same fixes applied for consistency

#### New Functions Added:
- **`get_visible_flashcard_sets(content_id, user_id=None)`** (Lines 174-219)
  - Returns only flashcard sets user is allowed to see
  - Filters by `is_shared=true` OR `created_by=user_id`

- **`get_visible_quiz_sets(content_id, user_id=None)`** (Lines 258-309)
  - Same logic for quiz sets

### 2. File: `BACKFILL_CREATED_BY.py` (NEW)

Script to populate `created_by=NULL` records with space owner ID.

**Run once to fix existing data:**
```bash
python BACKFILL_CREATED_BY.py
```

---

## 🔐 Visibility Rules

After these fixes, users can see:

### Flashcard Sets
```
✅ VISIBLE:
   - Sets where is_shared = true (shared with everyone)
   - Sets where created_by = current_user_id (own sets)

❌ NOT VISIBLE:
   - Sets where is_shared = false AND created_by ≠ current_user_id
```

### Quiz Sets
- Same rules as flashcards

---

## 📋 Endpoints to Update (FUTURE)

These endpoints should be updated to use the new visibility functions:

### Current Pattern (No Filtering):
```python
@app.post("/generate_flashcards")
async def generate_flashcards_endpoint(request: FlashcardRequest):
    result = generate_flashcards(...)
    return JSONResponse(result)
```

### Recommended Pattern (With Filtering):
```python
@app.get("/get_flashcards/{content_id}")
async def get_flashcards_endpoint(content_id: str, user_id: str):
    visible_sets = get_visible_flashcard_sets(content_id, user_id)
    return JSONResponse({"flashcards": visible_sets})
```

**Note**: The current endpoints return generated flashcards directly, which is fine. The filtering becomes critical when users VIEW previously generated flashcards.

---

## 🔄 Usage Examples

### In Your Code

**Before** (no filtering):
```python
from supabase_client import get_existing_flashcards

# Gets ALL flashcards regardless of ownership
flashcards = get_existing_flashcards(content_id)
```

**After** (with filtering):
```python
from supabase_client import get_visible_flashcard_sets

# Gets only flashcards the user is allowed to see
flashcards = get_visible_flashcard_sets(content_id, user_id)
```

---

## 📊 Database Query Examples

### Check if fix worked:
```sql
-- Recent flashcard sets should have created_by populated
SELECT 
  id, 
  set_number, 
  created_by, 
  is_shared, 
  created_at
FROM flashcard_sets 
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```

### Verify visibility rules:
```sql
-- Show what user A can see
SELECT * FROM flashcard_sets 
WHERE (is_shared = true OR created_by = 'user-a-uuid')
ORDER BY set_number;
```

---

## ✅ Testing Checklist

- [ ] Run backfill script: `python BACKFILL_CREATED_BY.py`
- [ ] Verify backfill completed without errors
- [ ] Generate new flashcards as User A
- [ ] Check database: set_number=latest has `created_by=user_a_id`
- [ ] User B accesses same document
- [ ] User B regenerates flashcards (creates set_number=latest+1)
- [ ] Check User A doesn't see User B's private sets
- [ ] Check User B sees their own sets
- [ ] Generate quiz and repeat for quizzes

---

## 🐛 Troubleshooting

### Problem: `created_by` still NULL after generation

**Solution**:
1. Check that `flashcard_process.py` passes `user_id` to `insert_flashcard_set()`
2. Verify `supabase_client.py` has been updated to fix conditional logic
3. Clear any cached imports: `python -c "import py_compile; py_compile.compile('supabase_client.py')"`

### Problem: Users still see other users' private sets

**Solution**:
1. Ensure retrieval endpoints use `get_visible_flashcard_sets()` instead of `get_existing_flashcards()`
2. Pass `user_id` from request context to visibility function
3. Check that `is_shared` flag is being set correctly during generation

### Problem: Backfill script fails

**Solution**:
1. Verify Supabase connection: `python -c "from supabase_client import supabase; print(supabase.table('spaces').select('id').limit(1).execute())"`
2. Check for permission issues on your Supabase role
3. Ensure `.env` file has correct `SUPABASE_URL` and `SUPABASE_KEY`

---

## 📚 Related Documentation

- `INVESTIGATION_REPORT.md` - Detailed root cause analysis
- `supabase_client.py` - Function implementations
- `flashcard_process.py` - Calls `insert_flashcard_set()` with user_id
- `quiz_process.py` - Calls `insert_quiz_set()` with user_id

---

## 🎯 Summary of Changes

| Issue | Fix | File | Impact |
|-------|-----|------|--------|
| `created_by=NULL` | Updated conditional logic | `supabase_client.py` | Future generations always track owner |
| No visibility filter | New functions added | `supabase_client.py` | Can filter by user permission |
| Old records orphaned | Backfill script | `BACKFILL_CREATED_BY.py` | Can restore ownership retroactively |

---

**Last Updated**: 2025-11-08  
**Status**: ✅ Implementation Complete | ⏳ Testing Pending

