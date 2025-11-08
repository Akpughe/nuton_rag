# Google Drive Integration - Frontend Implementation Guide

## 🎯 Overview

This guide provides the frontend implementation details for integrating Google Drive file import into your React application. The backend APIs are already implemented in `pipeline.py`.

## 🔧 Backend API Endpoints

The following endpoints are available for frontend integration:

### 1. List Drive Files
```http
POST /api/google-drive/files
Content-Type: application/json

{
  "access_token": "user_access_token",
  "refresh_token": "user_refresh_token", 
  "folder_id": "optional_folder_id",
  "file_types": ["pdf", "doc", "docx"],
  "max_results": 100
}
```

**Response:**
```json
{
  "files": [
    {
      "id": "file_id_123",
      "name": "Document.pdf",
      "mimeType": "application/pdf",
      "size": "1024000",
      "webViewLink": "https://drive.google.com/file/d/...",
      "createdTime": "2024-01-01T00:00:00.000Z",
      "modifiedTime": "2024-01-01T00:00:00.000Z"
    }
  ],
  "updated_tokens": {
    "access_token": "new_access_token",
    "refresh_token": "refresh_token"
  }
}
```

### 2. Import Selected Files
```http
POST /api/google-drive/import
Content-Type: application/json

{
  "file_ids": ["file_id_123", "file_id_456"],
  "space_id": "user_space_id",
  "access_token": "user_access_token",
  "refresh_token": "user_refresh_token"
}
```

**Response:**
```json
{
  "task_id": "import_task_123",
  "status": "processing", 
  "message": "Files are being processed",
  "updated_tokens": {
    "access_token": "new_access_token",
    "refresh_token": "refresh_token"
  }
}
```

### 3. Check Import Status
```http
GET /api/google-drive/import/status/import_task_123
```

**Response:**
```json
{
  "task_id": "import_task_123",
  "status": "completed",
  "processed_files": [
    {
      "file_id": "file_id_123",
      "document_id": "doc_uuid_123",
      "filename": "Document.pdf",
      "status": "success"
    }
  ],
  "errors": []
}
```

## 🚀 React Components Implementation

### 1. Google Drive Connector Component

```jsx
import React, { useState, useContext } from 'react';
import { AuthContext } from './AuthContext';

const GoogleDriveConnector = ({ onFilesSelected }) => {
  const { user, updateTokens } = useContext(AuthContext);
  const [isConnected, setIsConnected] = useState(false);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);

  const connectToDrive = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/google-drive/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          access_token: user.access_token,
          refresh_token: user.refresh_token,
          file_types: ['pdf', 'doc', 'docx'],
          max_results: 50
        })
      });

      const data = await response.json();
      
      if (data.updated_tokens) {
        updateTokens(data.updated_tokens);
      }

      setFiles(data.files || []);
      setIsConnected(true);
    } catch (error) {
      console.error('Failed to connect to Drive:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="google-drive-connector">
      {!isConnected ? (
        <button onClick={connectToDrive} disabled={loading}>
          {loading ? 'Connecting...' : 'Browse Google Drive'}
        </button>
      ) : (
        <DriveFileBrowser files={files} onFilesSelected={onFilesSelected} />
      )}
    </div>
  );
};
```

### 2. File Browser Component

```jsx
const DriveFileBrowser = ({ files, onFilesSelected }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  const filteredFiles = files.filter(file => 
    file.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleFileSelection = (file) => {
    setSelectedFiles(prev => {
      const isSelected = prev.find(f => f.id === file.id);
      if (isSelected) {
        return prev.filter(f => f.id !== file.id);
      } else {
        return [...prev, file];
      }
    });
  };

  const handleImport = () => {
    onFilesSelected(selectedFiles);
  };

  return (
    <div className="drive-file-browser">
      <div className="search-bar">
        <input
          type="text"
          placeholder="Search files..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className="file-list">
        {filteredFiles.map(file => (
          <div 
            key={file.id} 
            className={`file-item ${selectedFiles.find(f => f.id === file.id) ? 'selected' : ''}`}
            onClick={() => toggleFileSelection(file)}
          >
            <div className="file-icon">
              {getFileIcon(file.mimeType)}
            </div>
            <div className="file-info">
              <h4>{file.name}</h4>
              <p>{formatFileSize(file.size)} • {formatDate(file.modifiedTime)}</p>
            </div>
            <input 
              type="checkbox" 
              checked={!!selectedFiles.find(f => f.id === file.id)}
              onChange={() => toggleFileSelection(file)}
            />
          </div>
        ))}
      </div>

      {selectedFiles.length > 0 && (
        <div className="selection-actions">
          <p>{selectedFiles.length} files selected</p>
          <button onClick={handleImport}>
            Import Selected Files
          </button>
        </div>
      )}
    </div>
  );
};
```

### 3. Import Progress Component

```jsx
const DriveImportProgress = ({ taskId, onComplete }) => {
  const [status, setStatus] = useState('processing');
  const [progress, setProgress] = useState([]);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/google-drive/import/status/${taskId}`);
        const data = await response.json();
        
        setStatus(data.status);
        setProgress(data.processed_files || []);

        if (data.status === 'completed' || data.status === 'failed') {
          onComplete(data);
        }
      } catch (error) {
        console.error('Failed to check import status:', error);
      }
    };

    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, [taskId, onComplete]);

  return (
    <div className="import-progress">
      <h3>Importing Files from Google Drive</h3>
      <div className="progress-list">
        {progress.map(file => (
          <div key={file.file_id} className="progress-item">
            <span>{file.filename}</span>
            <span className={`status ${file.status}`}>
              {file.status === 'success' ? '✅' : file.status === 'error' ? '❌' : '⏳'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 4. Complete Import Flow Component

```jsx
const GoogleDriveImport = ({ spaceId, onImportComplete }) => {
  const { user, updateTokens } = useContext(AuthContext);
  const [currentStep, setCurrentStep] = useState('browse'); // browse, importing, complete
  const [taskId, setTaskId] = useState(null);

  const handleFilesSelected = async (selectedFiles) => {
    setCurrentStep('importing');
    
    try {
      const response = await fetch('/api/google-drive/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_ids: selectedFiles.map(f => f.id),
          space_id: spaceId,
          access_token: user.access_token,
          refresh_token: user.refresh_token
        })
      });

      const data = await response.json();
      
      if (data.updated_tokens) {
        updateTokens(data.updated_tokens);
      }

      setTaskId(data.task_id);
    } catch (error) {
      console.error('Import failed:', error);
      setCurrentStep('browse');
    }
  };

  const handleImportComplete = (result) => {
    setCurrentStep('complete');
    onImportComplete(result);
  };

  return (
    <div className="google-drive-import">
      {currentStep === 'browse' && (
        <GoogleDriveConnector onFilesSelected={handleFilesSelected} />
      )}
      
      {currentStep === 'importing' && taskId && (
        <DriveImportProgress 
          taskId={taskId} 
          onComplete={handleImportComplete} 
        />
      )}
      
      {currentStep === 'complete' && (
        <div className="import-complete">
          <h3>Import Complete!</h3>
          <p>Your Google Drive files have been successfully processed.</p>
        </div>
      )}
    </div>
  );
};
```

## 🔐 Authentication Integration

### Update Google Auth Scopes

```javascript
// In your Google OAuth configuration
const GOOGLE_SCOPES = [
  'profile',
  'email',
  'https://www.googleapis.com/auth/drive.readonly',
  'https://www.googleapis.com/auth/documents.readonly'
];

// When initializing Google Auth
gapi.load('auth2', () => {
  gapi.auth2.init({
    client_id: 'your_google_client_id',
    scope: GOOGLE_SCOPES.join(' ')
  });
});
```

### Token Management

```javascript
// AuthContext.js
const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  const updateTokens = (newTokens) => {
    setUser(prev => ({
      ...prev,
      access_token: newTokens.access_token,
      refresh_token: newTokens.refresh_token
    }));
    
    // Store in secure storage
    localStorage.setItem('google_tokens', JSON.stringify(newTokens));
  };

  return (
    <AuthContext.Provider value={{ user, updateTokens }}>
      {children}
    </AuthContext.Provider>
  );
};
```

## 🎨 Styling Guidelines

### CSS Classes Structure

```css
.google-drive-connector {
  /* Main container styles */
}

.drive-file-browser {
  /* File browser container */
}

.file-list {
  /* File list grid/flexbox */
}

.file-item {
  /* Individual file item */
}

.file-item.selected {
  /* Selected file styling */
}

.import-progress {
  /* Progress indicator styles */
}

.progress-item {
  /* Individual progress item */
}

.status.success { color: green; }
.status.error { color: red; }
.status.processing { color: orange; }
```

## 🔧 Utility Functions

```javascript
// Helper functions
const getFileIcon = (mimeType) => {
  if (mimeType.includes('pdf')) return '📄';
  if (mimeType.includes('document')) return '📝';
  if (mimeType.includes('spreadsheet')) return '📊';
  return '📎';
};

const formatFileSize = (bytes) => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString();
};
```

## 🚀 Integration Steps

1. **Update Google OAuth scopes** in your existing auth flow
2. **Add the GoogleDriveImport component** to your document upload page
3. **Style the components** to match your design system  
4. **Test the complete flow** from browsing to import completion
5. **Add error handling** for network failures and permission issues

This provides a complete frontend implementation that works seamlessly with the backend APIs!