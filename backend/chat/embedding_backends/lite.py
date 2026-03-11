import importlib

from langchain_core.embeddings import Embeddings

MODEL_ID = "bioformers/bioformer-8L"


def check_memory(required_gb: float = 0.4) -> bool:
    psutil = importlib.import_module("psutil")
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if available_gb < required_gb:
        print(f"⚠️ Advertencia: Memoria disponible baja ({available_gb:.2f} GB).")
        return False
    return True


def _load_model(model_id: str):
    if not check_memory():
        raise MemoryError("❌ Memoria insuficiente para cargar Bioformer-8L.")

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "El modo lite requiere 'transformers' y 'torch'. Instala las dependencias antes de usar EMBEDDINGS_MODE=lite."
        ) from exc

    print("⏳ Cargando Bioformer-8L...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModel.from_pretrained(model_id)
    print("✅ Bioformer-8L cargado correctamente.")
    return tokenizer, model


class BioformerEmbeddings(Embeddings):
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def _ensure_model_loaded(self):
        if self.model is None:
            self.tokenizer, self.model = _load_model(self.model_id)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model_loaded()

        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError(
                "El modo lite requiere 'torch'. Instala las dependencias antes de usar EMBEDDINGS_MODE=lite."
            ) from exc

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def create_embeddings() -> BioformerEmbeddings:
    return BioformerEmbeddings(MODEL_ID)