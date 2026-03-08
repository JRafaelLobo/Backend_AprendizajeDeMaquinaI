import os
import numpy as np
import logging
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from embeddings import bio_wrapper, _model as sent2vec_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_PATH   = "./Documentos/seccion_renal.pdf"
FAISS_PATH = "./FAISS/faiss_index_renal"

def txt_y_metadatos(pdf_path: str) -> tuple[str, dict]:
    """Extrae texto completo y headers del PDF."""
    doc = fitz.open(pdf_path)
    full_text = []
    headers = []

    for page in doc:
        full_text.append(page.get_text())
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        if span.get("color") == 19072 and span["text"].isupper():
                            headers.append(span["text"].strip())

    texto_completo = "\n".join(full_text)
    metadata = doc.metadata
    metadata["headers_extraidos"] = list(dict.fromkeys(headers))
    return texto_completo, metadata


def chunk_text(text: str) -> list[str]:
    """Divide el texto en chunks con overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.split_text(text)


def generar_embeddings(chunks: list[str]) -> np.ndarray | None:
    """Genera embeddings con BioSentVec para una lista de chunks."""
    try:
        embeddings = sent2vec_model.embed_sentences(chunks)
        return embeddings
    except Exception as e:
        logger.error(f"Error generando embeddings: {e}")
        return None


def guardar_en_faiss(chunks: list[str], metadatos_archivo: dict, faiss_path: str) -> FAISS | None:
    """Crea el vector store FAISS y lo guarda en disco."""
    try:
        documentos = [
            LCDocument(
                page_content=chunk,
                metadata={
                    "source": metadatos_archivo.get("file", "PDF_Renal"),
                    "headers": metadatos_archivo.get("headers_extraidos", [])[:5],
                },
            )
            for chunk in chunks
        ]

        vector_store = FAISS.from_documents(documents=documentos, embedding=bio_wrapper)

        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        vector_store.save_local(faiss_path)
        return vector_store

    except Exception as e:
        logger.error(f"Error en FAISS: {e}")
        return None


def construir_ensemble(chunks: list[str], metadatos: dict, vector_store: FAISS) -> EnsembleRetriever:
    """Combina BM25 + FAISS en un EnsembleRetriever."""
    documentos_bm25 = [
        LCDocument(
            page_content=chunk,
            metadata={
                "source": metadatos.get("file", "PDF_Renal"),
                "headers": metadatos.get("headers_extraidos", [])[:5],
            },
        )
        for chunk in chunks
    ]

    bm25_retriever = BM25Retriever.from_documents(documentos_bm25)
    bm25_retriever.k = 3

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )


def main():
    texto, metadatos = txt_y_metadatos(PDF_PATH)
    for h in metadatos["headers_extraidos"][:5]:
        print(f"     • {h}")

    chunks = chunk_text(texto)
    vectores = generar_embeddings(chunks)
    if vectores is None:
        print("  ❌ Error generando embeddings. Abortando.")
        return

    vectores_np = np.array(vectores)

    vector_store = guardar_en_faiss(chunks, metadatos, FAISS_PATH)
    if vector_store is None:
        print("  ❌ Error guardando en FAISS.")
        return
    print(f"  ✅ FAISS guardado en: {FAISS_PATH}")

    ensemble = construir_ensemble(chunks, metadatos, vector_store)

    return ensemble


if __name__ == "__main__":
    main()