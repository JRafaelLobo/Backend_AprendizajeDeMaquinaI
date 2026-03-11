from langchain_core.embeddings import Embeddings
import sent2vec
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR / "BioSentVec_model" / "BioSentVec_PubMed_MIMICIII-bigram_d700.bin")


def _load_model(path: str) -> sent2vec.Sent2vecModel:
    model = sent2vec.Sent2vecModel() 
    #model = sent2vec.SentEmbedder()  # API de Linux
    try:
        model.load_model(path)
        print("✅ BioSentVec cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando BioSentVec: {e}")
        raise
    return model


class BioSentVecWrapper(Embeddings):
    def __init__(self, model: sent2vec.Sent2vecModel):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_sentences(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_sentences([text])[0].tolist()

_model = _load_model(MODEL_PATH)
bio_wrapper = BioSentVecWrapper(_model)