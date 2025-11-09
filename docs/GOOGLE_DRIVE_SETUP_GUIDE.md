# Google Drive Integration - Setup & Testing Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Google Cloud Console Setup

#### Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the following APIs:
   - Google Drive API
   - Google Docs API

#### Configure OAuth 2.0
1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth 2.0 Client IDs**
3. Choose **Web application**
4. Add authorized redirect URIs:
   - `http://localhost:3000/auth/callback` (for development)
   - `https://yourdomain.com/auth/callback` (for production)
5. Download the client configuration

### 3. Environment Variables

Add to your `.env` file:
```env
# Google OAuth Credentials
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Generate a new encryption key for tokens
GOOGLE_TOKEN_ENCRYPTION_KEY=your_32_byte_fernet_key_here
```

#### Generate Encryption Key
Run this Python script to generate a secure encryption key:
```python
from cryptography.fernet import Fernet
print("GOOGLE_TOKEN_ENCRYPTION_KEY=" + Fernet.generate_key().decode())
```

### 4. Database Setup

Run the SQL commands from `DATABASE_SCHEMA_UPDATES.md`:
```sql
-- Create the google_auth_tokens table
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
    UNIQUE(user_id, space_id)
);

-- Add new columns to existing tables
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'upload';
ALTER TABLE pdfs ADD COLUMN IF NOT EXISTS drive_file_id VARCHAR(255) NULL;
-- ... (see full schema in DATABASE_SCHEMA_UPDATES.md)
```

## 🧪 Testing the Integration

### 1. Start the Backend Server
```bash
python pipeline.py
```
The server will be available at `http://localhost:8000`

### 2. Test API Endpoints

#### Test Drive File Listing
```bash
curl -X POST http://localhost:8000/api/google-drive/files \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "your_user_access_token",
    "refresh_token": "your_user_refresh_token",
    "file_types": ["pdf", "doc"],
    "max_results": 10
  }'
```

#### Test File Import
```bash
curl -X POST http://localhost:8000/api/google-drive/import \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["1ABC123def456GHI789"],
    "space_id": "your_space_id",
    "access_token": "your_user_access_token",
    "refresh_token": "your_user_refresh_token"
  }'
```

### 3. Frontend Integration Testing

Use the React components from `GOOGLE_DRIVE_FRONTEND_GUIDE.md`:

```jsx
// Basic test component
import { GoogleDriveImport } from './components/GoogleDriveImport';

function TestPage() {
  const handleImportComplete = (result) => {
    console.log('Import completed:', result);
  };

  return (
    <div>
      <h1>Test Google Drive Import</h1>
      <GoogleDriveImport 
        spaceId="test-space-123"
        onImportComplete={handleImportComplete}
      />
    </div>
  );
}
```

## 🔧 Debugging & Troubleshooting

### Common Issues

#### 1. Authentication Errors
```
Error: Failed to refresh Google access token
```
**Solution**: Check that `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are correctly set.

#### 2. Permission Errors
```
Error: Failed to download file: insufficient permissions
```
**Solution**: Ensure OAuth scopes include `drive.readonly` and the user has granted permissions.

#### 3. Encryption Errors
```
Error: Invalid encryption key format
```
**Solution**: Generate a new Fernet key using the script above.

#### 4. Database Errors
```
Error: relation "google_auth_tokens" does not exist
```
**Solution**: Run the database schema updates from `DATABASE_SCHEMA_UPDATES.md`.

### Logging

Enable detailed logging for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs for:
- Google API call details
- Token refresh attempts
- File download progress
- Pipeline processing status

### Performance Testing

#### Test with Multiple Files
```python
# Test script for bulk import
import requests

files_to_test = [
    "1ABC123def456GHI789",  # PDF file
    "2DEF456ghi789JKL012",  # Google Doc
    "3GHI789jkl012MNO345"   # Word doc
]

response = requests.post('http://localhost:8000/api/google-drive/import', json={
    "file_ids": files_to_test,
    "space_id": "performance-test",
    "access_token": "your_token",
    "refresh_token": "your_refresh_token"
})

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

## 📊 Monitoring

### Key Metrics to Monitor
- Token refresh rate
- File download success rate  
- Processing time per file
- Error rates by file type

### Health Checks
```python
# Add to your monitoring
def check_google_drive_health():
    try:
        # Test token storage
        from services.token_storage import get_token_storage
        storage = get_token_storage()
        
        # Test Google API connectivity
        from services.google_drive_service import GoogleDriveService
        # ... add basic connectivity test
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 🔒 Security Checklist

- [ ] Google OAuth credentials are secure
- [ ] Encryption key is generated and stored securely
- [ ] Database access is properly configured
- [ ] API endpoints have proper authentication
- [ ] Tokens are encrypted in database
- [ ] Only required OAuth scopes are requested
- [ ] Temporary files are properly cleaned up
- [ ] Error messages don't expose sensitive data

## 🚀 Production Deployment

### Environment Configuration
```env
# Production settings
GOOGLE_CLIENT_ID=prod_client_id
GOOGLE_CLIENT_SECRET=prod_client_secret
GOOGLE_TOKEN_ENCRYPTION_KEY=prod_encryption_key

# Optional: Rate limiting
GOOGLE_API_RATE_LIMIT=100  # requests per minute
```

### Scaling Considerations
- Implement Redis caching for token storage
- Use background job queue for large file processing
- Add rate limiting for Google API calls
- Monitor API quota usage

This completes the Google Drive integration! The system is now ready for testing and deployment.