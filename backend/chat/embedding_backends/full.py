from pathlib import Path

from langchain_core.embeddings import Embeddings
import sent2vec

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "BioSentVec_model" / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin")


def _load_model(path: str) -> sent2vec.Sent2vecModel:
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(path)
        print("✅ BioSentVec cargado correctamente.")
    except Exception as exc:
        print(f"❌ Error cargando BioSentVec: {exc}")
        raise
    return model


class BioSentVecWrapper(Embeddings):
    def __init__(self, model: sent2vec.Sent2vecModel):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_sentences(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_sentences([text])[0].tolist()


def create_embeddings() -> BioSentVecWrapper:
    return BioSentVecWrapper(_load_model(MODEL_PATH))