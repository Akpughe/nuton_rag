# Database Schema Updates for Google Drive Integration

## Required Database Changes

### 1. Add Google Token Storage Table

```sql
-- Table to store encrypted Google OAuth tokens per user/space
CREATE TABLE google_auth_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    space_id VARCHAR(255) NOT NULL,
    encrypted_access_token TEXT NOT NULL,
    encrypted_refresh_token TEXT NOT NULL,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    scopes TEXT[] NOT NULL DEFAULT ARRAY['drive.readonly', 'documents.readonly'],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure one token set per user/space combination
    UNIQUE(user_id, space_id)
);

-- Index for faster lookups
CREATE INDEX idx_google_tokens_user_space ON google_auth_tokens(user_id, space_id);
CREATE INDEX idx_google_tokens_expires ON google_auth_tokens(token_expires_at);
```

### 2. Update Documents Table (PDFs)

```sql
-- Add new columns to existing pdfs table
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'upload';
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS drive_file_id VARCHAR(255) NULL;
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) NULL;
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS file_size BIGINT NULL;
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS external_created_at TIMESTAMP WITH TIME ZONE NULL;
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS external_modified_at TIMESTAMP WITH TIME ZONE NULL;

-- Index for Drive file lookups
CREATE INDEX IF NOT EXISTS idx_pdfs_drive_file_id ON pdfs(drive_file_id) WHERE drive_file_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pdfs_source ON pdfs(source);
```

### 3. Update YouTube Table (YTS)

```sql
-- Add new columns to existing yts table for consistency
ALTER TABLE yts ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'youtube';
ALTER TABLE yts ADD COLUMN IF NOT EXISTS drive_file_id VARCHAR(255) NULL;
ALTER TABLE yts ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) NULL;
ALTER TABLE yts ADD COLUMN IF NOT EXISTS file_size BIGINT NULL;
ALTER TABLE yts ADD COLUMN IF NOT EXISTS external_created_at TIMESTAMP WITH TIME ZONE NULL;
ALTER TABLE yts ADD COLUMN IF NOT EXISTS external_modified_at TIMESTAMP WITH TIME ZONE NULL;

-- Index for consistency
CREATE INDEX IF NOT EXISTS idx_yts_source ON yts(source);
```

## Environment Variables Required

Add these to your `.env` file:

```env
# Google OAuth Credentials
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Encryption key for storing Google tokens (32-byte key)
GOOGLE_TOKEN_ENCRYPTION_KEY=your_32_byte_encryption_key_here
```

## Data Migration Notes

### For Existing Documents
```sql
-- Update existing records to have 'upload' source
UPDATE pdfs SET source = 'upload' WHERE source IS NULL;
UPDATE yts SET source = 'youtube' WHERE source IS NULL;
```

### Source Types
- `'upload'` - Traditional file uploads
- `'google_drive'` - Files imported from Google Drive
- `'youtube'` - YouTube videos (existing)

## Security Considerations

1. **Token Encryption**: All Google OAuth tokens must be encrypted before storage
2. **Scope Limitation**: Only store tokens with minimal required scopes
3. **Token Rotation**: Implement automatic refresh token handling
4. **Access Control**: Tokens are tied to specific user/space combinations

## Implementation in Code

### Token Storage Service
```python
# services/token_storage.py
import os
from cryptography.fernet import Fernet

class TokenStorage:
    def __init__(self):
        self.cipher = Fernet(os.getenv('GOOGLE_TOKEN_ENCRYPTION_KEY').encode())
    
    def encrypt_token(self, token: str) -> str:
        return self.cipher.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        return self.cipher.decrypt(encrypted_token.encode()).decode()
```

### Database Operations
```python
# In supabase_client.py
def store_google_tokens(user_id: str, space_id: str, access_token: str, refresh_token: str):
    # Encrypt tokens before storage
    # Insert/update google_auth_tokens table

def get_google_tokens(user_id: str, space_id: str):
    # Retrieve and decrypt tokens
    # Return decrypted tokens
```

## Required Dependencies

Add to requirements.txt:
```
cryptography>=41.0.0  # For token encryption
```