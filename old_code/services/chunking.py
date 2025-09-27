from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)

class MultiFileSemanticChunker:
    """
    Handles semantic chunking of multiple document types with metadata preservation.
    Splits documents into semantic chunks while preserving source metadata and 
    enriching chunks with additional information for better retrieval.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker with configurable chunk size and overlap parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        logger.info(f"Initializing chunker with size={chunk_size}, overlap={chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Semantic chunking for multiple documents.
        Preserves source file metadata and enriches chunks with additional info.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with enriched metadata
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
            
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error during document splitting: {str(e)}")
            return []
        
        # Enrich chunks with additional metadata for better retrieval
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Preserve all original metadata
                original_metadata = chunk.metadata.copy() if chunk.metadata else {}
                
                # Add chunk index
                original_metadata["chunk_index"] = i
                
                # Determine source type and set specific metadata
                self._enrich_metadata_by_source_type(original_metadata)
                
                # Create a new chunk with the enriched metadata
                enriched_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=original_metadata
                )
                
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.error(f"Error enriching chunk {i}: {str(e)}")
                # Still add the original chunk to avoid data loss
                enriched_chunks.append(chunk)
        
        logger.info(f"Produced {len(enriched_chunks)} enriched chunks")
        return enriched_chunks
    
    def _enrich_metadata_by_source_type(self, metadata: dict) -> None:
        """
        Helper method to enrich metadata based on source type
        
        Args:
            metadata: The metadata dictionary to enrich (modified in-place)
        """
        if "source_url" in metadata and "youtube" in str(metadata.get("source_url", "")):
            # YouTube video chunk
            metadata["source_type"] = "youtube_video"
            
            if "content_type" not in metadata:
                metadata["content_type"] = "youtube_transcript"
            
            if "source" not in metadata:
                metadata["source"] = "youtube"
                
        elif "document_id" in metadata:
            # Document chunk (PDF, DOCX, etc.)
            metadata["source_type"] = "document"
            
            if "content_type" not in metadata:
                file_type = metadata.get("file_type", "unknown")
                metadata["content_type"] = file_type
                
            if "source" not in metadata:
                metadata["source"] = "document"