
import uuid
import faiss
import numpy as np

from typing import List
from tqdm import tqdm
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class Ingestion:
    def __init__(self, documents: List[str], embeddings: Embeddings):
        self.documents  = documents
        self.embeddings = embeddings

    def _build_faiss_with_progress(self, docs, embedding_model, batch_size=32, normalize=True):
        texts = [d.page_content for d in docs]
        embeddings = []

        # 1) Calcular embeddings en lotes con barra
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            batch_emb = embedding_model.embed_documents(batch)
            embeddings.extend(batch_emb)

        embeddings = np.array(embeddings, dtype="float32")

        # 2) Normalizar si se quiere coseno
        if normalize:
            faiss.normalize_L2(embeddings)

        # 3) Crear índice FAISS
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # 4) Guardar Document completos con sus metadatos
        id_map = {str(uuid.uuid4()): d for d in docs}
        docstore = InMemoryDocstore(id_map)

        # 5) Devolver FAISS listo para usar
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=list(id_map.keys())
        )

        return vectorstore

    def ingest(self):
        return self._build_faiss_with_progress(
            self.documents, 
            self.embeddings,
        )