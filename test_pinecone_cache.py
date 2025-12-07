import unittest
import os
import shutil
import time
from pinecone_cache import PineconeCache

class TestPineconeCache(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_pinecone_cache.db"
        self.cache = PineconeCache(db_path=self.db_path, ttl_hours=1)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_set_and_get(self):
        query_emb = [0.1, 0.2, 0.3]
        results = [{"id": "1", "score": 0.9}]
        
        # Set cache
        self.cache.set(
            function_name="test_func",
            results=results,
            query_emb=query_emb,
            top_k=10
        )
        
        # Get cache
        cached = self.cache.get(
            function_name="test_func",
            query_emb=query_emb,
            top_k=10
        )
        
        self.assertEqual(cached, results)

    def test_cache_miss(self):
        cached = self.cache.get(
            function_name="test_func",
            query_emb=[0.1, 0.2, 0.3],
            top_k=10
        )
        self.assertIsNone(cached)

    def test_cache_key_sensitivity(self):
        query_emb = [0.1, 0.2, 0.3]
        results = [{"id": "1", "score": 0.9}]
        
        self.cache.set(
            function_name="test_func",
            results=results,
            query_emb=query_emb,
            top_k=10
        )
        
        # Different top_k should be a miss
        cached = self.cache.get(
            function_name="test_func",
            query_emb=query_emb,
            top_k=5
        )
        self.assertIsNone(cached)

    def test_rounding(self):
        # Test that minor floating point differences are handled (if we implemented rounding)
        # In my implementation, I rounded to 6 decimal places.
        
        emb1 = [0.1234567, 0.2, 0.3]
        emb2 = [0.1234568, 0.2, 0.3] # Difference in 7th decimal place
        
        results = [{"id": "1"}]
        
        self.cache.set("test", results, query_emb=emb1)
        cached = self.cache.get("test", query_emb=emb2)
        
        # Since I rounded to 6 places, 0.1234567 -> 0.123457, 0.1234568 -> 0.123457
        # Wait, 0.1234567 rounds to 0.123457. 0.1234568 rounds to 0.123457.
        # So they should match.
        
        self.assertEqual(cached, results)

if __name__ == '__main__':
    unittest.main()
