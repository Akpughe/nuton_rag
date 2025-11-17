import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from supabase_client import supabase

logging.basicConfig(level=logging.INFO)

class TokenStorage:
    """
    Secure storage service for Google OAuth tokens.
    Handles encryption/decryption and database operations.
    """
    
    def __init__(self):
        """Initialize the token storage service with encryption."""
        encryption_key = os.getenv('GOOGLE_TOKEN_ENCRYPTION_KEY')
        if not encryption_key:
            raise ValueError("GOOGLE_TOKEN_ENCRYPTION_KEY environment variable is required")
        
        # Ensure the key is properly formatted (32 bytes, base64 encoded)
        try:
            self.cipher = Fernet(encryption_key.encode())
        except Exception as e:
            raise ValueError(f"Invalid encryption key format: {e}")
        
        self.supabase = supabase
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt a token for secure storage."""
        if not token:
            return ""
        return self.cipher.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt a token from storage."""
        if not encrypted_token:
            return ""
        return self.cipher.decrypt(encrypted_token.encode()).decode()
    
    def store_tokens(self, 
                    user_id: str, 
                    space_id: str, 
                    access_token: str, 
                    refresh_token: str,
                    expires_at: Optional[datetime] = None,
                    scopes: Optional[list] = None) -> bool:
        """
        Store encrypted Google OAuth tokens in the database.
        
        Args:
            user_id: User identifier
            space_id: Space identifier
            access_token: Google access token
            refresh_token: Google refresh token
            expires_at: Token expiration time
            scopes: List of OAuth scopes
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Encrypt the tokens
            encrypted_access = self.encrypt_token(access_token)
            encrypted_refresh = self.encrypt_token(refresh_token)
            
            # Default scopes if not provided
            if scopes is None:
                scopes = ['https://www.googleapis.com/auth/drive.readonly',
                         'https://www.googleapis.com/auth/documents.readonly']
            
            # Prepare data for upsert
            token_data = {
                'user_id': user_id,
                'space_id': space_id,
                'encrypted_access_token': encrypted_access,
                'encrypted_refresh_token': encrypted_refresh,
                'scopes': scopes,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if expires_at:
                token_data['token_expires_at'] = expires_at.isoformat()
            
            # Upsert (insert or update) the tokens
            result = self.supabase.table('google_auth_tokens').upsert(
                token_data,
                on_conflict='user_id,space_id'
            ).execute()
            
            if result.data:
                logging.info(f"Successfully stored Google tokens for user {user_id}, space {space_id}")
                return True
            else:
                logging.error(f"Failed to store tokens: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error storing Google tokens: {e}")
            return False
    
    def get_tokens(self, user_id: str, space_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt Google OAuth tokens from the database.
        
        Args:
            user_id: User identifier
            space_id: Space identifier
            
        Returns:
            Dictionary with decrypted tokens and metadata, or None if not found
        """
        try:
            # Query for tokens
            result = self.supabase.table('google_auth_tokens').select('*').eq(
                'user_id', user_id
            ).eq('space_id', space_id).execute()
            
            if not result.data:
                logging.info(f"No Google tokens found for user {user_id}, space {space_id}")
                return None
            
            token_record = result.data[0]
            
            # Decrypt the tokens
            access_token = self.decrypt_token(token_record['encrypted_access_token'])
            refresh_token = self.decrypt_token(token_record['encrypted_refresh_token'])
            
            # Parse expiration date if present
            expires_at = None
            if token_record.get('token_expires_at'):
                expires_at = datetime.fromisoformat(token_record['token_expires_at'])
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expires_at': expires_at,
                'scopes': token_record.get('scopes', []),
                'created_at': token_record.get('created_at'),
                'updated_at': token_record.get('updated_at')
            }
            
        except Exception as e:
            logging.error(f"Error retrieving Google tokens: {e}")
            return None
    
    def update_access_token(self, 
                           user_id: str, 
                           space_id: str, 
                           new_access_token: str,
                           expires_at: Optional[datetime] = None) -> bool:
        """
        Update just the access token (useful after token refresh).
        
        Args:
            user_id: User identifier
            space_id: Space identifier
            new_access_token: New access token
            expires_at: New expiration time
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Encrypt the new access token
            encrypted_access = self.encrypt_token(new_access_token)
            
            # Prepare update data
            update_data = {
                'encrypted_access_token': encrypted_access,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if expires_at:
                update_data['token_expires_at'] = expires_at.isoformat()
            
            # Update the record
            result = self.supabase.table('google_auth_tokens').update(
                update_data
            ).eq('user_id', user_id).eq('space_id', space_id).execute()
            
            if result.data:
                logging.info(f"Successfully updated access token for user {user_id}, space {space_id}")
                return True
            else:
                logging.error(f"Failed to update access token: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error updating access token: {e}")
            return False
    
    def delete_tokens(self, user_id: str, space_id: str) -> bool:
        """
        Delete stored tokens for a user/space combination.
        
        Args:
            user_id: User identifier
            space_id: Space identifier
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            result = self.supabase.table('google_auth_tokens').delete().eq(
                'user_id', user_id
            ).eq('space_id', space_id).execute()
            
            logging.info(f"Successfully deleted Google tokens for user {user_id}, space {space_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting Google tokens: {e}")
            return False
    
    def is_token_expired(self, user_id: str, space_id: str) -> bool:
        """
        Check if the stored access token is expired.
        
        Args:
            user_id: User identifier
            space_id: Space identifier
            
        Returns:
            True if token is expired or not found, False if still valid
        """
        tokens = self.get_tokens(user_id, space_id)
        if not tokens or not tokens.get('expires_at'):
            return True
        
        return datetime.now(timezone.utc) >= tokens['expires_at']


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key for development/setup.
    This should be called once and the key stored securely.
    """
    return Fernet.generate_key().decode()


# Singleton instance for use across the application
_token_storage = None

def get_token_storage() -> TokenStorage:
    """Get the singleton TokenStorage instance."""
    global _token_storage
    if _token_storage is None:
        _token_storage = TokenStorage()
    return _token_storage