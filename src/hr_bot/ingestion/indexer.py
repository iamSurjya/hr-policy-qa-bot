import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction # type: ignore
import structlog

from hr_bot.config import settings
from hr_bot.ingestion.chunker import Chunk
from chromadb.api import ClientAPI

log = structlog.get_logger()

COLLECTION_NAME = "hr_policies"


def get_chroma_client() -> ClientAPI:
    """Returns a ChromaDB client that persists to disk.

    PersistentClient writes to CHROMA_PATH in .env.
    Data survives process restarts — unlike FAISS which
    required reloading the index file every time.
    """
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_or_create_collection(
    client: ClientAPI,
) -> chromadb.Collection:
    """Get existing collection or create a new one.

    The embedding function is attached to the collection —
    ChromaDB uses it automatically when you query, ensuring
    queries are embedded with the exact same model as the chunks.
    This is critical: mismatched embedding models produce
    meaningless similarity scores.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )


def build_index(chunks: list[Chunk], reset: bool = False) -> None:
    """Embed all chunks and store in ChromaDB.

    Args:
        chunks: list of Chunk objects from chunker.py
        reset: if True, delete existing collection first.
                Use this when policy documents have changed
                and you want a clean rebuild.
    """
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("collection_reset", collection=COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet, that's fine

    collection = get_or_create_collection(client)

    # Check if already indexed to avoid duplicate embeddings
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        log.warning(
            "collection_already_has_data",
            existing_chunks=existing_count,
            hint="Run with reset=True to rebuild the index",
        )
        return

    # Prepare data for ChromaDB bulk insert
    # ChromaDB expects parallel lists: ids, documents, metadatas
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # Unique ID per chunk: source + index
        chunk_id = f"{chunk.source}_{chunk.chunk_index}"
        ids.append(chunk_id)
        documents.append(chunk.content)
        metadatas.append({
            "source": chunk.source,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
        })

    log.info("embedding_started", total_chunks=len(chunks))

    # ChromaDB handles embedding in batches internally
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    log.info(
        "index_built",
        total_chunks=len(chunks),
        collection=COLLECTION_NAME,
        path=settings.chroma_path,
    )


def get_index_stats() -> dict:
    """Returns stats about the current index — useful for health checks."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    count = collection.count()
    return {
        "collection": COLLECTION_NAME,
        "total_chunks": count,
        "chroma_path": settings.chroma_path,
    }