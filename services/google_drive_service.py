import os
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
import tempfile
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.INFO)

class GoogleDriveService:
    """
    Google Drive API service for downloading and processing documents.
    Handles authentication, file listing, and content retrieval.
    """
    
    def __init__(self, access_token: str, refresh_token: str):
        """
        Initialize Google Drive service with user credentials.
        
        Args:
            access_token: User's current access token
            refresh_token: User's refresh token for token renewal
        """
        self.credentials = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
            scopes=[
                'https://www.googleapis.com/auth/drive.readonly',
                'https://www.googleapis.com/auth/documents.readonly'
            ]
        )
        
        self._refresh_token_if_needed()
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.docs_service = build('docs', 'v1', credentials=self.credentials)
    
    def _refresh_token_if_needed(self):
        """Refresh access token if it's expired."""
        if self.credentials.expired and self.credentials.refresh_token:
            try:
                self.credentials.refresh(Request())
                logging.info("Successfully refreshed Google access token")
            except Exception as e:
                logging.error(f"Failed to refresh token: {e}")
                raise ValueError("Failed to refresh Google access token")
    
    def list_files(self, 
                   folder_id: Optional[str] = None, 
                   file_types: List[str] = None,
                   max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List files from Google Drive.
        
        Args:
            folder_id: Optional folder ID to search within
            file_types: List of file extensions to filter by
            max_results: Maximum number of files to return
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            # Build query
            query_parts = []
            
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            
            if file_types:
                # Convert extensions to MIME types for common document formats
                mime_conditions = []
                for file_type in file_types:
                    if file_type.lower() == 'pdf':
                        mime_conditions.append("mimeType='application/pdf'")
                    elif file_type.lower() in ['doc', 'docx']:
                        mime_conditions.append("mimeType='application/vnd.google-apps.document'")
                        mime_conditions.append("mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'")
                        mime_conditions.append("mimeType='application/msword'")
                    elif file_type.lower() == 'txt':
                        mime_conditions.append("mimeType='text/plain'")
                
                if mime_conditions:
                    query_parts.append(f"({' or '.join(mime_conditions)})")
            
            # Add common filters
            query_parts.append("trashed=false")
            
            query = " and ".join(query_parts) if query_parts else "trashed=false"
            
            # Execute API call
            results = self.drive_service.files().list(
                q=query,
                pageSize=max_results,
                fields="files(id,name,mimeType,size,webViewLink,createdTime,modifiedTime,parents,permissions)"
            ).execute()
            
            files = results.get('files', [])
            
            logging.info(f"Found {len(files)} files in Google Drive")
            return files
            
        except HttpError as e:
            logging.error(f"Error listing Drive files: {e}")
            raise ValueError(f"Failed to list Drive files: {e}")
    
    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific file.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            File metadata dictionary
        """
        try:
            file_metadata = self.drive_service.files().get(
                fileId=file_id,
                fields="id,name,mimeType,size,webViewLink,createdTime,modifiedTime,parents,permissions"
            ).execute()
            
            return file_metadata
            
        except HttpError as e:
            logging.error(f"Error getting file metadata for {file_id}: {e}")
            raise ValueError(f"Failed to get file metadata: {e}")
    
    def download_file(self, file_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Download file content from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            Tuple of (file_content_bytes, file_metadata)
        """
        try:
            # Get file metadata first
            file_metadata = self.get_file_metadata(file_id)
            mime_type = file_metadata.get('mimeType', '')
            
            logging.info(f"Downloading file: {file_metadata.get('name')} (MIME: {mime_type})")
            
            # Handle Google Docs by exporting to PDF
            if mime_type == 'application/vnd.google-apps.document':
                request = self.drive_service.files().export_media(
                    fileId=file_id,
                    mimeType='application/pdf'
                )
                file_metadata['exportedAs'] = 'application/pdf'
                file_metadata['name'] += '.pdf'  # Add .pdf extension
                
            # Handle Google Sheets by exporting to PDF  
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                request = self.drive_service.files().export_media(
                    fileId=file_id,
                    mimeType='application/pdf'
                )
                file_metadata['exportedAs'] = 'application/pdf'
                file_metadata['name'] += '.pdf'
                
            # Handle regular files (PDFs, Word docs, etc.)
            else:
                request = self.drive_service.files().get_media(fileId=file_id)
            
            # Download the file
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    logging.info(f"Download progress: {int(status.progress() * 100)}%")
            
            file_content = file_io.getvalue()
            logging.info(f"Successfully downloaded {len(file_content)} bytes")
            
            return file_content, file_metadata
            
        except HttpError as e:
            logging.error(f"Error downloading file {file_id}: {e}")
            raise ValueError(f"Failed to download file: {e}")
    
    def save_temp_file(self, content: bytes, filename: str) -> str:
        """
        Save file content to a temporary file.
        
        Args:
            content: File content as bytes
            filename: Original filename for extension detection
            
        Returns:
            Path to temporary file
        """
        # Create temporary file with appropriate extension
        _, ext = os.path.splitext(filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        
        try:
            temp_file.write(content)
            temp_file.flush()
            temp_path = temp_file.name
            
            logging.info(f"Saved {len(content)} bytes to temporary file: {temp_path}")
            return temp_path
            
        finally:
            temp_file.close()
    
    def get_updated_tokens(self) -> Dict[str, str]:
        """
        Get current access and refresh tokens (after potential refresh).
        
        Returns:
            Dictionary with access_token and refresh_token
        """
        return {
            'access_token': self.credentials.token,
            'refresh_token': self.credentials.refresh_token
        }