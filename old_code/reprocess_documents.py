import asyncio
import json
import os
from typing import List, Dict
import httpx
from supabase import create_client, Client
import tempfile
import logging
from urllib.parse import urlparse
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

supabase: Client = create_client(supabase_url, supabase_key)

# Pipeline API endpoint
PIPELINE_URL = os.getenv("PIPELINE_URL", "http://localhost:8000")

# Batch size for processing
BATCH_SIZE = 5
# Delay between batches (in seconds)
BATCH_DELAY = 10

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = os.path.join(LOGS_DIR, f"failed_processes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log_failure(doc_type: str, doc_id: str, error: str, metadata: Dict):
    """Log failure details to the log file."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Document Type: {doc_type}\n")
        f.write(f"Document ID: {doc_id}\n")
        if doc_type == "PDF":
            f.write(f"File Path: {metadata.get('file_path', 'N/A')}\n")
            f.write(f"File Name: {metadata.get('file_name', 'N/A')}\n")
        else:  # YouTube
            f.write(f"YouTube URL: {metadata.get('yt_url', 'N/A')}\n")
            f.write(f"Video Title: {metadata.get('file_name', 'N/A')}\n")
        f.write(f"Space ID: {metadata.get('space_id', 'N/A')}\n")
        f.write(f"Error: {error}\n")
        f.write(f"{'='*80}\n")

async def download_file(url: str) -> str:
    """Download a file from URL and save it temporarily."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        # Get filename from URL or use a default
        filename = os.path.basename(urlparse(url).path) or "downloaded_file"
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(response.content)
            return temp_file.name

async def process_pdf_batch(batch: List[Dict], session: httpx.AsyncClient) -> List[Dict]:
    """Process a batch of PDF documents."""
    results = []
    
    for doc in batch:
        try:
            # Download the file if it's a URL
            if doc["file_path"].startswith(("http://", "https://")):
                temp_file_path = await download_file(doc["file_path"])
            else:
                # If it's a local path, just use it directly
                temp_file_path = doc["file_path"]
                if not os.path.exists(temp_file_path):
                    raise FileNotFoundError(f"File not found: {temp_file_path}")

            # Prepare the files and data for the request
            files = {
                "files": open(temp_file_path, "rb")
            }
            data = {
                "file_urls": json.dumps([doc["file_path"]]),
                "space_id": doc["space_id"] or "",
                "use_openai": "true"  # Using OpenAI for better quality
            }

            # Make request to process_document endpoint
            response = await session.post(
                f"{PIPELINE_URL}/process_document",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            results.append({
                "original_id": doc["id"],
                "status": "success",
                "result": response.json()
            })

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing document {doc['id']}: {error_msg}")
            # Log failure to file
            log_failure("PDF", doc["id"], error_msg, doc)
            results.append({
                "original_id": doc["id"],
                "status": "error",
                "error": error_msg
            })

        finally:
            # Clean up temp file if it was created
            if "temp_file_path" in locals() and temp_file_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logging.warning(f"Error deleting temp file {temp_file_path}: {str(e)}")

    return results

async def process_yt_batch(batch: List[Dict], session: httpx.AsyncClient) -> List[Dict]:
    """Process a batch of YouTube documents."""
    results = []
    
    for doc in batch:
        try:
            # Create a temporary file with the extracted text
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(doc["extracted_text"])
                temp_file_path = temp_file.name

            # Prepare the request data
            data = {
                "youtube_urls": json.dumps([doc["yt_url"]]),
                "space_id": doc["space_id"] or "",
                "embedding_model": "text-embedding-ada-002"
            }

            # Make request to process_youtube endpoint
            response = await session.post(
                f"{PIPELINE_URL}/process_youtube",
                data=data
            )
            response.raise_for_status()
            
            results.append({
                "original_id": doc["id"],
                "status": "success",
                "result": response.json()
            })

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing YouTube document {doc['id']}: {error_msg}")
            # Log failure to file
            log_failure("YouTube", doc["id"], error_msg, doc)
            results.append({
                "original_id": doc["id"],
                "status": "error",
                "error": error_msg
            })

        finally:
            # Clean up temp file
            if "temp_file_path" in locals():
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logging.warning(f"Error deleting temp file {temp_file_path}: {str(e)}")

    return results

async def main():
    # Get all documents from pdfs table
    pdf_response = supabase.table("pdfs").select("*").execute()
    pdf_docs = pdf_response.data

    # Get all documents from yts table
    yt_response = supabase.table("yts").select("*").execute()
    yt_docs = yt_response.data

    logging.info(f"Found {len(pdf_docs)} PDF documents and {len(yt_docs)} YouTube documents to process")
    logging.info(f"Failures will be logged to: {LOG_FILE}")

    # Process in batches
    async with httpx.AsyncClient(timeout=300.0) as session:  # 5 minute timeout
        # Process PDF documents
        for i in range(0, len(pdf_docs), BATCH_SIZE):
            batch = pdf_docs[i:i + BATCH_SIZE]
            logging.info(f"Processing PDF batch {i//BATCH_SIZE + 1}/{(len(pdf_docs) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            results = await process_pdf_batch(batch, session)
            
            # Log results
            for result in results:
                if result["status"] == "success":
                    logging.info(f"Successfully processed PDF document {result['original_id']}")
                else:
                    logging.error(f"Failed to process PDF document {result['original_id']}: {result['error']}")
            
            # Wait between batches
            if i + BATCH_SIZE < len(pdf_docs):
                logging.info(f"Waiting {BATCH_DELAY} seconds before next batch...")
                await asyncio.sleep(BATCH_DELAY)

        # Process YouTube documents
        for i in range(0, len(yt_docs), BATCH_SIZE):
            batch = yt_docs[i:i + BATCH_SIZE]
            logging.info(f"Processing YouTube batch {i//BATCH_SIZE + 1}/{(len(yt_docs) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            results = await process_yt_batch(batch, session)
            
            # Log results
            for result in results:
                if result["status"] == "success":
                    logging.info(f"Successfully processed YouTube document {result['original_id']}")
                else:
                    logging.error(f"Failed to process YouTube document {result['original_id']}: {result['error']}")
            
            # Wait between batches
            if i + BATCH_SIZE < len(yt_docs):
                logging.info(f"Waiting {BATCH_DELAY} seconds before next batch...")
                await asyncio.sleep(BATCH_DELAY)

if __name__ == "__main__":
    asyncio.run(main()) 