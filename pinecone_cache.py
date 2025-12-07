import sqlite3
import json
import hashlib
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PineconeCache:
    def __init__(self, db_path: str = "pinecone_cache.db", ttl_hours: int = 24):
        self.db_path = db_path
        self.ttl_hours = ttl_hours
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and create the cache table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        cache_key TEXT PRIMARY KEY,
                        results TEXT,
                        created_at TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone cache DB: {e}")

    def _generate_key(self, function_name: str, **kwargs) -> str:
        """Generate a unique cache key based on search parameters."""
        # Create a dictionary of parameters to serialize
        params = {"function": function_name}
        
        # Process kwargs to ensure they are serializable and deterministic
        for k, v in kwargs.items():
            if k == "query_emb" and v:
                # Round embeddings
                params[k] = [round(x, 6) for x in v]
            elif k == "acl_tags" and v:
                params[k] = sorted(v)
            elif k == "document_ids" and v:
                params[k] = sorted(v)
            else:
                params[k] = v
        
        # Serialize to JSON with sorted keys to ensure deterministic output
        serialized = json.dumps(params, sort_keys=True)
        
        # Return SHA256 hash
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, function_name: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """Retrieve results from cache if they exist and are not expired."""
        key = self._generate_key(function_name, **kwargs)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT results, created_at FROM search_cache WHERE cache_key = ?", (key,))
                row = cursor.fetchone()
                
                if row:
                    results_json, created_at_str = row
                    created_at = datetime.fromisoformat(created_at_str)
                    
                    # Check TTL
                    if datetime.now() - created_at < timedelta(hours=self.ttl_hours):
                        logger.info(f"Pinecone cache HIT for key {key[:8]}...")
                        return json.loads(results_json)
                    else:
                        logger.info(f"Pinecone cache EXPIRED for key {key[:8]}...")
                        # Optional: delete expired entry
                        cursor.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
                        conn.commit()
                else:
                    logger.debug(f"Pinecone cache MISS for key {key[:8]}...")
                    
        except Exception as e:
            logger.error(f"Error reading from Pinecone cache: {e}")
            
        return None

    def set(self, function_name: str, results: List[Dict[str, Any]], **kwargs):
        """Store results in cache."""
        key = self._generate_key(function_name, **kwargs)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO search_cache (cache_key, results, created_at) VALUES (?, ?, ?)",
                    (key, json.dumps(results), datetime.now().isoformat())
                )
                conn.commit()
                logger.info(f"Stored results in Pinecone cache for key {key[:8]}...")
        except Exception as e:
            logger.error(f"Error writing to Pinecone cache: {e}")

    def clear(self):
        """Clear all cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM search_cache")
                conn.commit()
                logger.info("Pinecone cache cleared")
        except Exception as e:
            logger.error(f"Error clearing Pinecone cache: {e}")
