from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)

class MultiFileSemanticChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Semantic chunking for multiple documents
        Preserves source file metadata and enriches chunks with additional info
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
            
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Enrich chunks with additional metadata for better retrieval
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Preserve all original metadata
            original_metadata = chunk.metadata.copy() if chunk.metadata else {}
            
            # Add chunk index and total chunks for this document
            original_metadata["chunk_index"] = i
            
            # Determine source type and set specific metadata
            if "source_url" in original_metadata and "youtube" in str(original_metadata.get("source_url", "")):
                # This is a YouTube video chunk
                original_metadata["source_type"] = "youtube_video"
                
                # Add video-specific metadata if not already present
                if "content_type" not in original_metadata:
                    original_metadata["content_type"] = "youtube_transcript"
                
                if "source" not in original_metadata:
                    original_metadata["source"] = "youtube"
                    
            elif "document_id" in original_metadata:
                # This is a document chunk (PDF, DOCX, etc.)
                original_metadata["source_type"] = "document"
                
                if "content_type" not in original_metadata:
                    file_type = original_metadata.get("file_type", "unknown")
                    original_metadata["content_type"] = file_type
                    
                if "source" not in original_metadata:
                    original_metadata["source"] = "document"
            
            # Create a new chunk with the enriched metadata
            enriched_chunk = Document(
                page_content=chunk.page_content,
                metadata=original_metadata
            )
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f"Chunked {len(documents)} documents into {len(enriched_chunks)} chunks")
        return enriched_chunks