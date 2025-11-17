import asyncio
import json
import os
import re
from typing import List, Dict
import httpx
from supabase import create_client, Client
import tempfile
import logging
from urllib.parse import urlparse
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
if not PIPELINE_URL:
    raise ValueError("PIPELINE_URL environment variable must be set")

# Batch size for processing
BATCH_SIZE = 5
# Delay between batches (in seconds)
BATCH_DELAY = 10

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def extract_failed_pdf_ids(log_file_path: str) -> List[str]:
    """Extract failed PDF document IDs from the log file."""
    failed_ids = []
    current_id = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for Document Type and ID lines
            if "Document Type: PDF" in line:
                # Next line should contain the ID
                continue
            elif "Document ID:" in line:
                doc_id = line.split("Document ID:")[1].strip()
                failed_ids.append(doc_id)
    
    return failed_ids

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
            # Add logging for debugging
            logging.info(f"Processing document {doc['id']}: {doc['file_path']}")
            
            # Validate file path
            if not doc["file_path"] or doc["file_path"] == "undefined":
                raise ValueError(f"Invalid file path for document {doc['id']}")

            # Download file with better error handling
            if doc["file_path"].startswith(("http://", "https://")):
                try:
                    temp_file_path = await download_file(doc["file_path"])
                    logging.info(f"Successfully downloaded file to {temp_file_path}")
                except httpx.HTTPError as e:
                    raise ValueError(f"Failed to download file: {str(e)}")
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

async def main():
    # Get the failed IDs from the log file
    log_file = "logs/failed_processes_20250520_010940.txt"
    failed_ids = extract_failed_pdf_ids(log_file)
    
    if not failed_ids:
        logging.info("No failed PDF documents found in the log file.")
        return
    
    logging.info(f"Found {len(failed_ids)} failed PDF documents to retry")
    
    # Create new log file for retry results
    retry_log_file = os.path.join(LOGS_DIR, f"retry_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.info(f"Retry results will be logged to: {retry_log_file}")
    
    # Fetch documents from Supabase
    failed_docs = []
    for doc_id in failed_ids:
        try:
            response = supabase.table("pdfs").select("*").eq("id", doc_id).execute()
            if response.data:
                failed_docs.append(response.data[0])
            else:
                logging.warning(f"Document not found in pdfs table: {doc_id}")
        except Exception as e:
            logging.error(f"Error fetching document {doc_id} from Supabase: {str(e)}")
    
    logging.info(f"Successfully fetched {len(failed_docs)} documents from Supabase")
    
    # Process in batches
    async with httpx.AsyncClient(timeout=300.0) as session:  # 5 minute timeout
        for i in range(0, len(failed_docs), BATCH_SIZE):
            batch = failed_docs[i:i + BATCH_SIZE]
            logging.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(failed_docs) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            results = await process_pdf_batch(batch, session)
            
            # Log results to file
            with open(retry_log_file, 'a') as f:
                for result in results:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Document ID: {result['original_id']}\n")
                    f.write(f"Status: {result['status']}\n")
                    if result['status'] == 'error':
                        f.write(f"Error: {result['error']}\n")
                    f.write(f"{'='*80}\n")
            
            # Log results to console
            for result in results:
                if result["status"] == "success":
                    logging.info(f"Successfully processed PDF document {result['original_id']}")
                else:
                    logging.error(f"Failed to process PDF document {result['original_id']}: {result['error']}")
            
            # Wait between batches
            if i + BATCH_SIZE < len(failed_docs):
                logging.info(f"Waiting {BATCH_DELAY} seconds before next batch...")
                await asyncio.sleep(BATCH_DELAY)

if __name__ == "__main__":
    asyncio.run(main()) 