import io
import re
from datetime import datetime
import PyPDF2
from fastapi import HTTPException
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import logging
import os

logger = logging.getLogger(__name__)

class PDFHandler:
    def __init__(self):
        # Initialize ChromaDB client
        chroma_host = os.getenv('CHROMA_HOST', os.getenv('CHROMA_DB_CONNECTION_STRING'))
        chroma_port = int(os.getenv('CHROMA_PORT', 8000))
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host, 
            port=chroma_port,
            # tenant="default_tenant",
            # database="default_database"
        )
    
    @staticmethod
    def clean_filename(filename):
        """
        Cleans the filename by replacing special characters and spaces with underscores.
        Converts the filename to lowercase.
        """
        cleaned_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        cleaned_name = cleaned_name.lower()
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        return cleaned_name

    @staticmethod
    def determine_page_for_chunk(chunk: str, page_text_map: list[dict]) -> int:
        """
        Determine the page number for a given text chunk
        """
        for page_info in page_text_map:
            if chunk in page_info['text']:
                return page_info['page_num']
        return page_text_map[-1]['page_num']

    async def handle_pdf_upload(self, file, space_id, rag_system, nuton_api):
        try:
            # Read PDF
            pdf_content = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            # Extract text with advanced page tracking
            full_text = ""
            page_text_map = []
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    try:
                        page_content = page.extract_text()
                    except Exception:
                        try:
                            page_content = page.get_text()
                        except Exception as alternative_extract_error:
                            print(f"Failed to extract text from page {page_num}")
                            page_content = ""
                    
                    page_content = ''.join(char for char in page_content if char.isprintable())
                    
                    page_text_map.append({
                        'start': len(full_text),
                        'text': page_content,
                        'page_num': page_num
                    })
                    full_text += page_content + "\n"
                
                except Exception as page_error:
                    print(f"Unexpected error extracting text from page {page_num}: {page_error}")
                    continue
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Save PDF metadata to database
            pdf_upload = rag_system.supabase.table('pdfs').insert({
                'space_id': space_id,
                'extracted_text': full_text
            }).execute()
            
            pdf_id = pdf_upload.data[0]['id']

            # Trigger upload in the background
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{nuton_api}/upload",
                    files={"file": (file.filename, pdf_content, file.content_type)},
                    data={"pdf_id": pdf_id},
                    timeout=60.0
                )

                if response.status_code != 200:
                    print(f"Error from worker service: {response.text}")
            
            # Process text chunks and embeddings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=100
            )
            text_chunks = text_splitter.split_text(full_text)
            
            collection = self.chroma_client.get_or_create_collection("pdf_embeddings")
            
            for i, chunk in enumerate(text_chunks):
                try:
                    page_number = self.determine_page_for_chunk(chunk, page_text_map)
                    embedding = rag_system.generate_embedding(chunk)
                    
                    collection.add(
                        ids=[f"{pdf_id}_{i}"],
                        embeddings=[embedding],
                        metadatas=[{
                            "pdf_id": pdf_id,
                            "chunk_index": i,
                            "page": page_number,
                            "created_at": datetime.now().isoformat()
                        }],
                        documents=[chunk]
                    )
                except Exception as chunk_error:
                    print(f"Error processing chunk {i}: {chunk_error}")
            
            return {
                "status": "success", 
                "pdf_id": pdf_id,
                "extracted_text": full_text, 
                "chunks_processed": len(text_chunks)
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 