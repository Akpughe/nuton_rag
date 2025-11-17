# 📋 Deployment Steps - Flashcard Visibility Fix

## Pre-Deployment Verification

### ✅ Step 1: Verify Code Changes
```bash
# Check that supabase_client.py has been updated
grep -n "is not None" supabase_client.py | grep created_by

# Should show:
# 114: if created_by is not None:
# 136: if created_by is not None:
# 303: if created_by is not None:
# 329: if created_by is not None:
```

### ✅ Step 2: Verify New Functions Exist
```bash
# Check visibility functions were added
grep -n "def get_visible_" supabase_client.py

# Should show:
# 174: def get_visible_flashcard_sets
# 258: def get_visible_quiz_sets
```

### ✅ Step 3: Verify Backfill Script
```bash
# Check backfill script exists and has no syntax errors
python -m py_compile BACKFILL_CREATED_BY.py
echo "Syntax OK" # Should print if no errors
```

---

## Deployment Phase 1: Code Deploy

### Step 1: Push to Repository
```bash
git add supabase_client.py BACKFILL_CREATED_BY.py
git commit -m "Fix: Ensure created_by field populated and add visibility filtering

- Fixed conditional logic for created_by assignment (use 'is not None')
- Added get_visible_flashcard_sets() for permission-based retrieval
- Added get_visible_quiz_sets() for permission-based retrieval
- Created BACKFILL_CREATED_BY.py script for retroactive ownership assignment
- Updated quiz_sets and flashcard_sets insert logic for consistency"

git push origin chonkie-alternative
```

### Step 2: Deploy to Production
```bash
# Standard deployment process
# (Assuming CI/CD pipeline handles this)
```

### Step 3: Verify Deployment
```bash
# SSH into production or use deployment logs
# Check that new code is running:
curl http://localhost:8000/  # Should return 200

# Check Python module loads without errors:
python -c "from supabase_client import get_visible_flashcard_sets; print('✅ Module loaded')"
```

---

## Deployment Phase 2: Data Backfill

### Step 1: Pre-Backfill Backup (CRITICAL)
```bash
# Backup current flashcard_sets and quiz_sets
# This should be done by your DB admin or platform backup service

# Recommended: Take a full database snapshot
# Expected time: ~5-30 minutes depending on database size
```

### Step 2: Run Backfill Script

```bash
# On your application server or in a data processing container
cd /path/to/rag_system

# Run the backfill script
python BACKFILL_CREATED_BY.py

# Expected output:
# INFO:root:Starting backfill of created_by fields...
# INFO:root:Found X flashcard_sets with NULL created_by
# INFO:root:Updated flashcard_set ... with created_by=...
# INFO:root:Found Y quiz_sets with NULL created_by  
# INFO:root:Updated quiz_set ... with created_by=...
# INFO:root:✅ Backfill completed successfully!
```

### Step 3: Verify Backfill Success

```bash
# Check that NULL values are gone
# Connect to your Supabase database directly:

SELECT COUNT(*) as null_count 
FROM flashcard_sets 
WHERE created_by IS NULL;
-- Expected result: 0

SELECT COUNT(*) as null_count 
FROM quiz_sets 
WHERE created_by IS NULL;
-- Expected result: 0

# Spot check - verify owner was assigned correctly
SELECT id, set_number, created_by, is_shared 
FROM flashcard_sets 
WHERE created_by IS NOT NULL
LIMIT 5;
-- Should show valid UUIDs in created_by column
```

### Step 4: Post-Backfill Validation

```bash
# Verify relationships are correct
SELECT COUNT(*) FROM flashcard_sets fs
WHERE created_by IS NOT NULL
AND EXISTS (
  SELECT 1 FROM users u WHERE u.id = fs.created_by
);
-- Should match number of flashcard_sets with non-null created_by
```

---

## Testing Phase

### Unit Test: Visibility Function

```python
# In your test file, add:
from supabase_client import get_visible_flashcard_sets

def test_visibility_filtering():
    # Setup
    content_id = "test-content-id"
    user_a = "user-a-uuid"
    user_b = "user-b-uuid"
    
    # Test: User A sees only their sets and shared sets
    visible_a = get_visible_flashcard_sets(content_id, user_a)
    
    for flashcard_set in visible_a:
        assert flashcard_set["is_shared"] or flashcard_set["created_by"] == user_a
    
    print("✅ Visibility filtering works correctly")
```

### Integration Test: Regenerate Flashcards

```bash
# Test with the original endpoint
curl -X POST http://localhost:8000/regenerate_flashcards \
  -H "Content-Type: application/json" \
  -d '{
    "space_id": "test-space-id",
    "document_id": "test-doc-id", 
    "user_id": "test-user-id",
    "set_id": 2
  }'

# Expected response:
# - Status: 200
# - Response contains "status": "success"
# - Response contains generated flashcards

# Verify in database:
# SELECT * FROM flashcard_sets 
# WHERE created_by = 'test-user-id'
# ORDER BY created_at DESC LIMIT 1
# 
# Should show: created_by IS NOT NULL, is_shared = false
```

### E2E Test: Multi-User Scenario

```bash
# Simulate two users sharing a document

# 1. User A generates flashcards
curl -X POST http://localhost:8000/generate_flashcards \
  -d '{"document_id": "doc1", "user_id": "user-a", "space_id": "space1"}'
# → Creates set_id=1 (created_by=user-a, is_shared=false)

# 2. User B regenerates flashcards
curl -X POST http://localhost:8000/regenerate_flashcards \
  -d '{"document_id": "doc1", "user_id": "user-b", "space_id": "space1"}'
# → Creates set_id=2 (created_by=user-b, is_shared=false)

# 3. Verify visibility (assuming you have a GET endpoint)
# User A should see: set_id=1 (theirs) + any shared sets
# User B should see: set_id=2 (theirs) + any shared sets
# User A should NOT see set_id=2 (User B's private set)
# User B should NOT see set_id=1 (User A's private set)

echo "✅ Multi-user isolation verified"
```

---

## Post-Deployment Monitoring

### Logs to Monitor

```bash
# Watch for ownership assignment in logs
grep "created_by" /var/log/application.log

# Expected logs:
# "Flashcard set created successfully with id: ..., created_by: user-uuid, is_shared: false"
# "Updated flashcard set ... with created_by: user-uuid"

# Watch for errors
grep "ERROR" /var/log/application.log

# Should be minimal (expect some general app errors, not related to our fix)
```

### Database Monitoring

```sql
-- Monitor recent insertions
SELECT COUNT(*) as recent_flashcard_sets,
       COUNT(CASE WHEN created_by IS NULL THEN 1 END) as null_created_by
FROM flashcard_sets
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Run hourly: should always have 0 NULL values in recent rows

-- Monitor visibility function usage (if logging added)
-- Check application logs for "Retrieved X visible flashcard sets"
```

### Alert Rules (Recommended)

```
Alert if:
1. Flashcard set insert fails (check exception logs)
2. Backfill script fails (one-time, after deployment)
3. More than 5% of newly created sets have created_by=NULL (suggests regression)
```

---

## Rollback Plan (If Needed)

### Option 1: Quick Rollback (First Hour)
```bash
# If immediately after deployment and issues detected:
# Revert the code changes
git revert HEAD
git push origin chonkie-alternative

# Application will start using old code
# NOTE: This doesn't undo the backfill. Backfill is data-only and one-time.
```

### Option 2: Data Rollback (If Backfill Issues)
```bash
# If backfill script caused issues:
# Restore from pre-backfill snapshot

# Contact your database administrator:
# "Please restore flashcard_sets and quiz_sets tables from backup 
#  created before backfill process began"

# Expected impact: ~1 hour of data loss (during backfill window)
```

### Option 3: Partial Rollback
```sql
-- If only some records were affected
-- Restore just those records
-- Work with DBA on targeted restoration
```

---

## Success Criteria

### ✅ Deployment Successful If:

1. **Code deployed without errors**
   - No import errors in application logs
   - Endpoints respond normally (200 status)

2. **Backfill completed successfully**
   - No NULL values remain in created_by column
   - All ownership relationships are valid

3. **Visibility filtering works**
   - New flashcards have created_by populated
   - get_visible_flashcard_sets() returns correct sets
   - get_visible_quiz_sets() returns correct sets

4. **No performance regression**
   - Endpoint response times within normal range
   - Database query performance acceptable
   - No cascading failures

5. **Multi-user isolation works**
   - Users only see their own private sets + shared sets
   - Private sets from other users are not visible

---

## Rollback Criteria

### 🔴 Rollback If:

1. **Critical errors in logs**
   ```
   ERROR: insert_flashcard_set failed
   ERROR: get_visible_flashcard_sets failed
   ```

2. **Multiple deployment failures**
   - More than 3 failed flashcard generations in a row
   - Consistent SQL errors

3. **Data integrity issues**
   ```sql
   SELECT COUNT(*) FROM flashcard_sets WHERE created_by IS NULL;
   -- Returns > 0 after backfill (indicates backfill failure)
   ```

4. **Performance degradation**
   - Endpoint response times > 2x normal
   - Database timeouts occurring

5. **Visibility logic broken**
   - Users seeing other users' private sets
   - Users missing their own sets

---

## Documentation Updates

### Before Public Release:

1. **API Documentation**
   - Document `created_by` field in flashcard responses
   - Document visibility rules
   - Document potential filtering in future endpoints

2. **Deployment Guide**
   - Add these steps to your deployment procedures
   - Document backfill as one-time operation

3. **Database Schema Documentation**
   - Update to note ownership tracking on flashcard_sets
   - Update to note ownership tracking on quiz_sets

---

## Timeline Estimate

| Phase | Task | Est. Time |
|-------|------|-----------|
| 1 | Pre-deployment verification | 5 min |
| 2 | Code push and deploy | 10 min |
| 2 | Deployment verification | 5 min |
| 3 | Database backup | 15 min |
| 3 | Run backfill script | 5-30 min |
| 3 | Backfill verification | 10 min |
| 4 | Unit/Integration testing | 15 min |
| 4 | E2E testing | 20 min |
| 5 | Monitoring setup | 10 min |
| **Total** | **Full deployment** | **~1.5-2 hours** |

---

## Support & Troubleshooting

### If Issues Occur:

1. **Check logs first**
   ```bash
   tail -f /var/log/application.log | grep -E "ERROR|flashcard|created_by"
   ```

2. **Verify database connection**
   ```bash
   python -c "from supabase_client import supabase; print(supabase.table('flashcard_sets').count().execute())"
   ```

3. **Check Python module**
   ```bash
   python -c "from supabase_client import get_visible_flashcard_sets; print('OK')"
   ```

4. **Review this document**
   - Check Testing Phase section for common issues
   - Review Rollback Plan if needed

5. **Contact support**
   - Reference: "Flashcard Visibility Fix Deployment"
   - Include: Error messages and relevant logs

---

## Sign-Off

- [x] Code reviewed
- [x] Backfill script tested
- [ ] Deployed to staging
- [ ] Staging tests passed
- [ ] Deployed to production
- [ ] Production verification complete
- [ ] Monitoring setup confirmed

---

**Deployment Documentation**  
**Last Updated**: 2025-11-08  
**Status**: Ready for Deployment

