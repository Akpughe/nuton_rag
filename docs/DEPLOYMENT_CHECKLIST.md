# Deployment Checklist: User-Based Data Isolation

## Pre-Deployment Verification ✓

### Code Changes
- [x] `supabase_client.py` - Added `determine_shared_status()`, updated insert functions
- [x] `flashcard_process.py` - Added `user_id` parameter, implemented ownership logic
- [x] `quiz_process.py` - Added `user_id` parameter, implemented ownership logic
- [x] `pipeline.py` - Updated Pydantic models, updated endpoint handlers
- [x] All imports updated with new helper function
- [x] No syntax errors introduced
- [x] Backward compatibility maintained

### Documentation
- [x] `ENDPOINT_DOCUMENTATION.md` - Complete API specifications
- [x] `QUICK_REFERENCE.md` - Quick lookup guide
- [x] `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- [x] `CHANGES_SUMMARY.md` - Before/After comparison

---

## Phase 1: Database Preparation

### Prerequisites
- [ ] Backup production database
- [ ] Verify `created_by` column exists in `flashcard_sets`
- [ ] Verify `is_shared` column exists in `flashcard_sets`
- [ ] Verify `created_by` column exists in `quiz_sets`
- [ ] Verify `is_shared` column exists in `quiz_sets`
- [ ] All columns are nullable/have defaults

### Verification Queries
```sql
-- Check flashcard_sets columns
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'flashcard_sets'
AND column_name IN ('created_by', 'is_shared');

-- Check quiz_sets columns
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'quiz_sets'
AND column_name IN ('created_by', 'is_shared');

-- Should show 4 rows, all exist
```

### Migration Script
- [ ] Prepare user-provided migration script
- [ ] Test migration on staging database
- [ ] Verify data integrity after migration
- [ ] Document migration rollback procedure

---

## Phase 2: Code Deployment

### Pre-Deployment Tests
- [ ] All unit tests pass
- [ ] No new linting errors (pre-existing warnings OK)
- [ ] Code review approved
- [ ] Syntax check: `python -m py_compile flashcard_process.py quiz_process.py supabase_client.py pipeline.py`

### Deployment Steps
1. [ ] Merge code to main branch
2. [ ] Build new container image
3. [ ] Push to container registry
4. [ ] Update deployment configuration
5. [ ] Deploy to staging environment
6. [ ] Run smoke tests on staging

### Smoke Tests on Staging
```bash
# Test generate_flashcards endpoint
curl -X POST http://localhost:8000/generate_flashcards \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "test-doc",
    "space_id": "test-space",
    "user_id": "test-user",
    "num_questions": 5
  }'

# Should return 200 with flashcards or 500 with error message
# NOT 400 (validation error)
```

---

## Phase 3: Frontend Coordination

### Frontend Requirements
- [ ] Add `user_id` to all flashcard generation requests
- [ ] Add `user_id` to all quiz generation requests
- [ ] Add `user_id` to all flashcard regeneration requests
- [ ] Add `user_id` to all quiz regeneration requests

### Sample Frontend Changes
```typescript
// Before
await fetch('/generate_flashcards', {
  body: JSON.stringify({
    document_id,
    space_id,
    num_questions: 30
  })
});

// After
await fetch('/generate_flashcards', {
  body: JSON.stringify({
    document_id,
    space_id,
    user_id: currentUser.id,    // ← ADD THIS
    num_questions: 30
  })
});
```

### Testing
- [ ] Test with owner user
- [ ] Test with non-owner user
- [ ] Verify owner sees shared content
- [ ] Verify non-owner only sees private content
- [ ] Verify error handling for missing `user_id`

---

## Phase 4: Production Deployment

### Pre-Production Checklist
- [ ] Staging tests all pass
- [ ] Stakeholders approved
- [ ] Rollback plan documented
- [ ] On-call team briefed
- [ ] Monitoring alerts configured

### Deployment Window
- [ ] Schedule during low-traffic period
- [ ] Notify users of potential brief downtime (if needed)
- [ ] Have team members on standby

### Deployment Steps
1. [ ] Apply database migration script (or done already)
2. [ ] Deploy new code to production
3. [ ] Verify service is running
4. [ ] Run production smoke tests
5. [ ] Monitor error logs
6. [ ] Monitor application metrics

### Post-Deployment Verification
```sql
-- Verify migration worked
SELECT COUNT(*) as total_sets, 
       SUM(CASE WHEN is_shared THEN 1 ELSE 0 END) as shared_sets,
       SUM(CASE WHEN NOT is_shared THEN 1 ELSE 0 END) as private_sets
FROM flashcard_sets;

SELECT COUNT(*) as total_sets, 
       SUM(CASE WHEN is_shared THEN 1 ELSE 0 END) as shared_sets,
       SUM(CASE WHEN NOT is_shared THEN 1 ELSE 0 END) as private_sets
FROM quiz_sets;
```

---

## Phase 5: Frontend Deployment

### Prerequisites
- [ ] Backend fully deployed and stable
- [ ] All backend smoke tests passing
- [ ] No error spikes in logs

### Frontend Changes
- [ ] Update `useFlashcards` hook
- [ ] Update `useQuiz` hook
- [ ] Update flashcard generation API calls
- [ ] Update quiz generation API calls
- [ ] Test with real backend

### Testing
- [ ] Test all 4 generation endpoints
- [ ] Test error cases (missing user_id)
- [ ] Test different user types
- [ ] Verify data isolation in UI

### Rollback Plan (Frontend)
- [ ] Revert to previous version
- [ ] Won't break backend (user_id optional initially)
- [ ] All content defaults to shared

---

## Phase 6: Monitoring & Validation

### Logging Verification
```
✓ Look for "Ownership check: user=..." in logs
✓ Look for "creating new flashcard set (..., created_by: ..., is_shared:..." 
✓ Should see mix of is_shared=true and is_shared=false
✓ No ERROR entries related to determine_shared_status
```

### Metrics to Monitor
- [ ] Error rate (should be ≤ baseline)
- [ ] Request latency (should be ±2ms of baseline, due to 1-2 extra queries)
- [ ] Success rate (should be ≥ 99%)
- [ ] Database connection pool (should be healthy)

### Alerts to Configure
- [ ] High error rate on `/generate_flashcards`
- [ ] High error rate on `/generate_quiz`
- [ ] Database query latency spike
- [ ] Failed ownership determinations

---

## Phase 7: Team Communication

### Before Deployment
- [ ] Notify dev team
- [ ] Notify QA team
- [ ] Brief on-call team
- [ ] Share this checklist with team

### Deployment Status
- [ ] Update status page (if applicable)
- [ ] Notify stakeholders of progress
- [ ] Share results in team Slack/Discord

### After Deployment
- [ ] Confirm all systems healthy
- [ ] Post-deployment retrospective (if issues)
- [ ] Document any learnings
- [ ] Update deployment playbook

---

## Rollback Procedures

### If Backend Deployment Fails
```bash
# 1. Rollback code
git revert <commit-hash>
docker build -t backend:previous .
# 2. Redeploy previous version
# 3. Verify service
```

### If Frontend Deployment Fails
```bash
# 1. Rollback frontend code
git revert <commit-hash>
npm run build
# 2. Redeploy previous version
# 3. Frontend will work without user_id initially
```

### If Database Migration Fails
```sql
-- Restore from backup
-- OR manually rollback
UPDATE flashcard_sets SET is_shared = true, created_by = NULL;
UPDATE quiz_sets SET is_shared = true, created_by = NULL;
-- Redeploy previous backend code
```

### If Data Isolation Issues Occur
```sql
-- Check for anomalies
SELECT created_by, is_shared, COUNT(*) as count
FROM flashcard_sets
GROUP BY created_by, is_shared;

-- If owner content marked private, fix:
UPDATE flashcard_sets SET is_shared = true 
WHERE created_by IN (SELECT user_id FROM spaces WHERE id = '{space_id}');
```

---

## Success Criteria

✅ All items checked before declaring success:

- [ ] Backend code deployed without errors
- [ ] Database migration completed successfully
- [ ] Frontend updated with `user_id` parameter
- [ ] All 4 endpoints functioning correctly
- [ ] Owner content is `is_shared = true`
- [ ] Non-owner content is `is_shared = false`
- [ ] Error logs show no new errors
- [ ] Performance metrics stable (±5% of baseline)
- [ ] Team training completed
- [ ] Documentation updated

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1: DB Prep | 1-2 hours | Verify columns, prepare migration |
| Phase 2: Code Deploy | 30 min | Build, push, deploy to staging |
| Phase 3: Frontend Coord | 2-4 hours | Update code, test thoroughly |
| Phase 4: Prod Deploy | 1-2 hours | Apply migration, deploy code |
| Phase 5: Frontend Deploy | 30 min | Deploy frontend changes |
| Phase 6: Monitoring | 1 hour | Verify logs, metrics |
| Phase 7: Communication | 30 min | Notify team, document |
| **Total** | **6-11 hours** | Depends on testing thoroughness |

---

## Sign-Off

- [ ] Development Lead: _________________ Date: _______
- [ ] QA Lead: _________________ Date: _______
- [ ] DevOps Lead: _________________ Date: _______
- [ ] Product Manager: _________________ Date: _______

---

## Contact Information

**For Issues During Deployment:**
- Backend: [Engineer Name]
- Frontend: [Engineer Name]
- Database: [DBA Name]
- On-Call: [On-Call Name]

---

**Last Updated**: [Current Date]
**Status**: Ready for Deployment ✅

