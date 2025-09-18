
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

class Ingestion:
    def __init__(self, documents: List[str], embeddings: Embeddings):
        self.documents  = documents
        self.embeddings = embeddings

    def ingest(self):
        return FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )