import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

EmbeddingBuilder = Callable[[], Embeddings]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_EMBEDDINGS_MODE = "full"
SUPPORTED_EMBEDDINGS_MODES = ("full", "lite")
EMBEDDINGS_MODE_ENV = "EMBEDDINGS_MODE"
FAISS_INDEX_FILENAME = "index.faiss"


def _load_project_env() -> None:
    for env_path in (PROJECT_DIR / ".env", PROJECT_DIR / "backend" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return


_load_project_env()


def resolve_embeddings_mode(mode: str | None = None) -> str:
    selected_mode = (mode or os.getenv(EMBEDDINGS_MODE_ENV, DEFAULT_EMBEDDINGS_MODE)).strip().lower()
    if selected_mode not in SUPPORTED_EMBEDDINGS_MODES:
        supported = ", ".join(SUPPORTED_EMBEDDINGS_MODES)
        raise ValueError(f"Valor inválido para {EMBEDDINGS_MODE_ENV}: {selected_mode}. Valores soportados: {supported}.")
    return selected_mode


def get_faiss_index_path(mode: str | None = None) -> str:
    selected_mode = resolve_embeddings_mode(mode)
    index_name = "faiss_index_renal" if selected_mode == "full" else f"faiss_index_renal_{selected_mode}"
    return str(BASE_DIR / "FAISS" / index_name)


def get_faiss_index_file_path(mode: str | None = None) -> str:
    return str(Path(get_faiss_index_path(mode)) / FAISS_INDEX_FILENAME)


class FaissIndexNotFoundError(FileNotFoundError):
    def __init__(self, mode: str, index_file_path: str):
        message = (
            f"No existe un índice FAISS para EMBEDDINGS_MODE={mode} en {index_file_path}. "
            f"Genera el índice con el mismo modo de embeddings o cambia {EMBEDDINGS_MODE_ENV}=full."
        )
        super().__init__(message)
        self.mode = mode
        self.index_file_path = index_file_path


def ensure_faiss_index_exists(mode: str | None = None) -> str:
    selected_mode = resolve_embeddings_mode(mode)
    index_path = get_faiss_index_path(selected_mode)
    index_file_path = get_faiss_index_file_path(selected_mode)
    if not Path(index_file_path).exists():
        raise FaissIndexNotFoundError(selected_mode, index_file_path)
    return index_path


def is_faiss_index_ready(mode: str | None = None) -> bool:
    return Path(get_faiss_index_file_path(mode)).exists()


def _build_full_embeddings() -> Embeddings:
    from .embedding_backends.full import create_embeddings

    return create_embeddings()


def _build_lite_embeddings() -> Embeddings:
    from .embedding_backends.lite import create_embeddings

    return create_embeddings()


EMBEDDING_BUILDERS: dict[str, EmbeddingBuilder] = {
    "full": _build_full_embeddings,
    "lite": _build_lite_embeddings,
}


class LazyEmbeddings(Embeddings):
    def __init__(self, mode: str | None = None, builders: dict[str, EmbeddingBuilder] | None = None):
        self.mode = resolve_embeddings_mode(mode)
        self._builders = builders or EMBEDDING_BUILDERS
        self._instance: Embeddings | None = None

    def _get_instance(self) -> Embeddings:
        if self._instance is None:
            self._instance = self._builders[self.mode]()
        return self._instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_instance().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_instance().embed_query(text)


def get_embeddings(mode: str | None = None) -> LazyEmbeddings:
    return LazyEmbeddings(mode=mode)


bio_wrapper = get_embeddings()