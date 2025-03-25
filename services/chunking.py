from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document

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
        Preserves source file metadata
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks