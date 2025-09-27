import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from supabase import create_client
import groq

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
        
        # Generate and update space name based on processed documents
        # if all_documents:
        #     try:
        #         # Extract texts from documents
        #         texts = [doc.page_content for doc in all_documents]
        #         name_result = self.generate_and_update_space_name(space_id, texts)
        #         if name_result['status'] == 'error':
        #             print(f"Warning: Failed to update space name: {name_result['message']}")
        #     except Exception as e:
        #         print(f"Warning: Error generating space name: {e}")
        
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
                page_level_documents = loader.load()
                
                processed_documents = []
                if file_extension == 'pdf': # Apply line-level splitting only for PDFs for now
                    for page_doc in page_level_documents:
                        # Make a mutable copy of the original page's metadata
                        current_page_metadata = page_doc.metadata.copy()
                        
                        # Convert 0-indexed 'page' from PyPDFLoader to 1-indexed 'page_display'
                        raw_page_number = current_page_metadata.get('page')
                        if isinstance(raw_page_number, int):
                            current_page_metadata['page_display'] = raw_page_number + 1
                        else:
                            # If 'page' is not found or not an int, set a placeholder
                            current_page_metadata['page_display'] = -1 # Indicates unknown or not applicable

                        page_content_str = page_doc.page_content
                        if isinstance(page_content_str, bytes):
                            page_content_str = page_content_str.decode('utf-8', errors='replace')
                        
                        lines = page_content_str.splitlines()
                        for line_idx, line_text in enumerate(lines):
                            if line_text.strip(): # Process non-empty lines
                                line_metadata = current_page_metadata.copy() # Inherits 'page' and 'page_display'
                                line_metadata['line_in_page'] = line_idx
                                # 'page' metadata (0-indexed) is still there from PyPDFLoader
                                # 'page_display' (1-indexed) is now also available
                                
                                # Filter to only printable characters for the line
                                printable_line_text = ''.join(char for char in line_text if char.isprintable() or char in ['\\n', '\\t'])
                                
                                processed_documents.append(
                                    Document(page_content=printable_line_text, metadata=line_metadata)
                                )
                    if not processed_documents: # Fallback if all lines were empty or pdf was empty
                        # Add page-level docs if line splitting resulted in nothing (e.g. image-only pdf page)
                        # We still need to clean their content and ensure 'page_display' is set
                        for page_doc in page_level_documents:
                            # Ensure 'page_display' is set if not already (e.g. if this path is taken directly)
                            if 'page_display' not in page_doc.metadata:
                                raw_page_number = page_doc.metadata.get('page')
                                if isinstance(raw_page_number, int):
                                    page_doc.metadata['page_display'] = raw_page_number + 1
                                else:
                                    page_doc.metadata['page_display'] = -1

                            page_content_str = page_doc.page_content
                            if isinstance(page_content_str, bytes):
                                page_content_str = page_content_str.decode('utf-8', errors='replace')
                            printable_page_content = ''.join(char for char in page_content_str if char.isprintable() or char in ['\\n', '\\t'])
                            page_doc.page_content = printable_page_content
                        processed_documents.extend(page_level_documents)

                else: # For non-PDFs, use page_level_documents directly after cleaning
                    # For non-PDFs, 'page' metadata might not exist or be 0-indexed.
                    # We can leave them as is or decide on a convention if needed.
                    # For now, just ensure content cleaning.
                    for doc in page_level_documents:
                        page_content_str = doc.page_content
                        if isinstance(page_content_str, bytes):
                            page_content_str = page_content_str.decode('utf-8', errors='replace')
                        printable_content = ''.join(char for char in page_content_str if char.isprintable() or char in ['\\n', '\\t'])
                        doc.page_content = printable_content
                    processed_documents.extend(page_level_documents)

                documents = processed_documents # These are now more granular for PDFs

                # Extract text content - safely handle any encoding issues
                full_text = ""
                for doc in documents: # Iterate over potentially line-level documents
                    # page_content is already filtered for printable characters
                    full_text += doc.page_content + "\n\n" 
                
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
                for doc in documents: # Ensure this metadata is added to the granular documents
                    doc.metadata['source_file'] = file_name
                    doc.metadata['document_id'] = document_id
                    doc.metadata['space_id'] = space_id
                    # Add file_url to metadata if available
                    if file_url:
                        doc.metadata['file_url'] = file_url
                
                return {
                    'documents': documents, # Return the granular documents
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
                        # For Textract fallback, we might not have line-level granularity easily.
                        # We'll use the full_text from Textract for the Supabase record,
                        # but the 'documents' from _parse_textract_response are page-level.
                        full_text_from_textract = textract_result['documents'][0].page_content
                        
                        # Ensure text is properly encoded
                        if isinstance(full_text_from_textract, bytes):
                            full_text_from_textract = full_text_from_textract.decode('utf-8', errors='replace')
                        
                        # Filter to only printable characters
                        full_text_from_textract = ''.join(char for char in full_text_from_textract if char.isprintable() or char in ['\\n', '\\t'])
                        
                        # Prepare Supabase data
                        supabase_data = {
                            'name': file_name,
                            # 'content': full_text_from_textract, # This was likely a typo, should be extracted_text
                            'space_id': space_id,
                            'file_type': file_extension,
                            'extracted_text': full_text_from_textract,
                            'file_path': file_path # Original file path, or URL if provided earlier
                        }
                        
                        # Add file_url if available
                        if file_url:
                            supabase_data['file_path'] = file_url # if file_url is present, it's the primary identifier
                        
                        result = self.supabase.table('pdfs').insert(supabase_data).execute()
                        
                        document_id = result.data[0]['id']
                        
                        # Update metadata for the Textract-derived documents
                        # These are typically page-level, not line-level from Textract.
                        for doc in textract_result['documents']:
                            doc.metadata['document_id'] = document_id
                            doc.metadata['space_id'] = space_id
                            if file_url:
                                doc.metadata['file_url'] = file_url
                        
                        return {
                            'documents': textract_result['documents'], # These are page-level documents
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
        full_text = ''.join(char for char in full_text if char.isprintable() or char in ['\\n', '\\t', ' '])
        
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

    def generate_and_update_space_name(self, space_id: str, texts: List[str], max_words: int = 5) -> Dict[str, Any]:
        """
        Generate a meaningful name for a space based on extracted texts and update the spaces table.
        
        Args:
            space_id: ID of the space to update
            texts: List of extracted texts from documents
            max_words: Maximum number of words to include in the generated name
            
        Returns:
            Dictionary with status and generated name
        """
        try:
            # Combine all texts and get first 1000 words
            combined_text = ' '.join(texts)
            words = combined_text.split()[:1000]  # Get first 1000 words
            combined_text = ' '.join(words)
            
            # Initialize Groq client
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            groq_client = groq.Client(api_key=groq_api_key)
            
            # Generate title using Groq
            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates concise, meaningful titles based on content. Generate a title that captures the main topic or theme of the following content. The title should be 3-5 words long and be descriptive but concise."
                    },
                    {
                        "role": "user",
                        "content": f"Generate a title for the following content:\n\n{combined_text}"
                    }
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            # Extract and clean the generated title
            generated_name = response.choices[0].message.content.strip()
            # Remove any quotes or special characters
            generated_name = generated_name.strip('"\'')
            
            # If no meaningful title was generated, use a default name
            if not generated_name:
                generated_name = f"Space {space_id[:8]}"
            
            # Update the spaces table
            self.supabase.table('spaces').update({
                'name': generated_name
            }).eq('id', space_id).execute()
            
            return {
                'status': 'success',
                'name': generated_name,
                'message': f'Successfully updated space name to: {generated_name}'
            }
            
        except Exception as e:
            error_msg = f"Error generating and updating space name: {e}"
            print(error_msg)
            return {
                'status': 'error',
                'name': None,
                'message': error_msg
            }