import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from supabase import create_client

from langchain_community.document_loaders import (
    PyPDFLoader as PDFLoader, 
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain.docstore.document import Document


class MultiFileDocumentProcessor:
    def __init__(self):
        # Initialize AWS client with region
        self.textract = boto3.client('textract', region_name='us-east-1')
        self.loaders = {
            'pdf': PDFLoader,
            'pptx': UnstructuredPowerPointLoader,
            'docx': UnstructuredWordDocumentLoader,
            'doc': UnstructuredWordDocumentLoader
        }
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL_DEV')
        supabase_key = os.getenv('SUPABASE_KEY_DEV')
        self.supabase = create_client(supabase_url, supabase_key)

    def process_files(self, file_paths: List[str], space_id: str, file_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process multiple files concurrently
        Returns a list of documents from all processed files with their database IDs
        
        Args:
            file_paths: List of paths to files to process
            space_id: ID of the space to associate files with
            file_urls: Optional list of URLs corresponding to each file_path
        """
        all_documents = []
        document_ids = {}
        
        if not file_paths:
            return {
                'documents': [],
                'document_ids': {},
                'status': 'error',
                'message': 'No files provided for processing'
            }
        
        # Create a mapping of file paths to URLs
        file_url_map = {}
        if file_urls:
            file_url_map = {
                file_path: file_urls[i] if i < len(file_urls) else None
                for i, file_path in enumerate(file_paths)
            }
        
        with ThreadPoolExecutor(max_workers=min(10, len(file_paths))) as executor:
            # Submit processing tasks for each file
            future_to_file = {
                executor.submit(self.process_single_file, file_path, space_id, file_url_map.get(file_path)): file_path 
                for file_path in file_paths
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result and result.get('documents'):
                        all_documents.extend(result['documents'])
                        if result.get('document_id'):
                            document_ids[file_path] = result['document_id']
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return {
            'documents': all_documents,
            'document_ids': document_ids,
            'status': 'success' if all_documents else 'error',
            'message': f'Processed {len(all_documents)} documents successfully' if all_documents else 'Failed to process any documents',
            'total_processed': len(document_ids)
        }

    def process_single_file(self, file_path: str, space_id: str, file_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single file with fallback mechanisms and store in Supabase
        
        Args:
            file_path: Path to the file to process
            space_id: ID of the space to associate file with
            file_url: Optional URL corresponding to the file
        """
        try:
            file_extension = file_path.split('.')[-1].lower()
            # Handle file name safely for binary data
            try:
                file_name = os.path.basename(file_path)
                if isinstance(file_name, bytes):
                    try:
                        file_name = file_name.decode('utf-8', errors='replace')
                    except:
                        file_name = f"file.{file_extension}"
            except:
                file_name = f"unknown_file.{file_extension}"
            
            # Validate file type
            if file_extension not in self.loaders:
                return {
                    'documents': [],
                    'document_id': None,
                    'status': 'error',
                    'message': f'Unsupported file type: {file_extension}'
                }
            
            try:
                # Primary: Langchain Loader
                loader = self.loaders.get(file_extension)(file_path)
                documents = loader.load()
                
                # Extract text content - safely handle any encoding issues
                full_text = ""
                for doc in documents:
                    try:
                        if isinstance(doc.page_content, bytes):
                            page_content = doc.page_content.decode('utf-8', errors='replace')
                        else:
                            page_content = doc.page_content
                        
                        # Filter to only printable characters
                        page_content = ''.join(char for char in page_content if char.isprintable() or char in ['\n', '\t'])
                        full_text += page_content + "\n\n"
                    except Exception as text_error:
                        print(f"Error processing document text: {text_error}")
                
                # If full_text is empty after processing, raise error
                if not full_text.strip():
                    raise ValueError("No valid text content extracted from file")
                
                # Prepare Supabase insertion data
                supabase_data = {
                    'space_id': space_id,
                    'extracted_text': full_text,
                    'file_type': file_extension,
                    # 'file_path': file_path  # Always include the file_path
                }
                
                # Add file_url if available
                if file_url:
                    supabase_data['file_path'] = file_url  # Add file_url as a separate field
                
                # Insert into Supabase pdfs table - ensuring all fields are properly encoded
                result = self.supabase.table('pdfs').insert(supabase_data).execute()
                
                document_id = result.data[0]['id']
                
                # Enrich metadata with source file information and IDs
                for doc in documents:
                    doc.metadata['source_file'] = file_name
                    doc.metadata['document_id'] = document_id
                    doc.metadata['space_id'] = space_id
                    # Add file_url to metadata if available
                    if file_url:
                        doc.metadata['file_url'] = file_url
                
                return {
                    'documents': documents,
                    'document_id': document_id,
                    'status': 'success',
                    'message': f'Successfully processed {file_name}'
                }
            
            except Exception as langchain_error:
                # Fallback: AWS Textract for complex documents
                print(f"Langchain processing error, trying fallback: {langchain_error}")
                try:
                    with open(file_path, 'rb') as document:
                        textract_response = self.textract.detect_document_text(
                            Document={'Bytes': document.read()}
                        )
                        # Process Textract response, extract text
                        textract_result = self._parse_textract_response(textract_response, file_path, space_id, file_url)
                        
                        # Insert into Supabase
                        full_text = textract_result['documents'][0].page_content
                        
                        # Ensure text is properly encoded
                        if isinstance(full_text, bytes):
                            full_text = full_text.decode('utf-8', errors='replace')
                        
                        # Filter to only printable characters
                        full_text = ''.join(char for char in full_text if char.isprintable() or char in ['\n', '\t'])
                        
                        # Prepare Supabase data
                        supabase_data = {
                            'name': file_name,
                            'content': full_text,
                            'space_id': space_id,
                            'file_type': file_extension,
                            'extracted_text': full_text,  # Add extracted text to its own column
                            'file_path': file_path
                        }
                        
                        # Add file_url if available
                        if file_url:
                            supabase_data['file_url'] = file_url
                        
                        result = self.supabase.table('pdfs').insert(supabase_data).execute()
                        
                        document_id = result.data[0]['id']
                        
                        # Update metadata
                        for doc in textract_result['documents']:
                            doc.metadata['document_id'] = document_id
                            doc.metadata['space_id'] = space_id
                            # Add file_url to metadata if available
                            if file_url:
                                doc.metadata['file_url'] = file_url
                        
                        return {
                            'documents': textract_result['documents'],
                            'document_id': document_id,
                            'status': 'success',
                            'message': f'Successfully processed {file_name} using Textract fallback'
                        }
                
                except Exception as textract_error:
                    error_msg = f"Document processing failed for {file_path}: {langchain_error}, {textract_error}"
                    print(error_msg)
                    return {
                        'documents': [],
                        'document_id': None,
                        'status': 'error', 
                        'message': error_msg
                    }
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {e}"
            print(error_msg)
            return {
                'documents': [],
                'document_id': None,
                'status': 'error', 
                'message': error_msg
            }

    def _parse_textract_response(self, response, file_path: str, space_id: str, file_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert Textract response to Langchain Documents
        
        Args:
            response: Textract response
            file_path: Path to the file
            space_id: ID of the space to associate with
            file_url: Optional URL corresponding to the file
        """
        # Extract text from Textract response
        text_blocks = [
            block['Text'] 
            for block in response.get('Blocks', []) 
            if block['BlockType'] == 'LINE'
        ]
        
        # Join text and ensure it's properly encoded
        full_text = ' '.join(text_blocks)
        
        # Handle file name safely for binary data
        try:
            source_file = os.path.basename(file_path)
            if isinstance(source_file, bytes):
                try:
                    source_file = source_file.decode('utf-8', errors='replace')
                except:
                    source_file = "unknown_file"
        except:
            source_file = "unknown_file"
        
        # Ensure text is properly encoded
        if isinstance(full_text, bytes):
            full_text = full_text.decode('utf-8', errors='replace')
        
        # Filter to only printable characters
        full_text = ''.join(char for char in full_text if char.isprintable() or char in ['\n', '\t', ' '])
        
        # Prepare metadata
        metadata = {
            'source_file': source_file,
            'extraction_method': 'textract',
            'space_id': space_id
        }
        
        # Add file_url to metadata if available
        if file_url:
            metadata['file_url'] = file_url
        
        documents = [
            Document(
                page_content=full_text,
                metadata=metadata
            )
        ]
        
        return {'documents': documents}