from dataclasses import dataclass
import chromadb
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction # type: ignore
import structlog

from hr_bot.config import settings
from hr_bot.ingestion.indexer import COLLECTION_NAME

log = structlog.get_logger()


@dataclass
class SearchResult:
    """A single retrieved chunk with its similarity score and metadata.

    distance: ChromaDB cosine distance — lower means more similar.
    0.0 = identical, 2.0 = completely opposite.
    We convert to a similarity score (1 - distance) for readability.
    """
    content: str
    source: str
    title: str
    chunk_index: int
    similarity: float


def get_collection() -> chromadb.Collection:
    """Get the ChromaDB collection for querying.

    We attach the same embedding function used during indexing —
    ChromaDB uses it to embed the query before searching.
    """
    client: ClientAPI = chromadb.PersistentClient(path=settings.chroma_path)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def search(query: str, k: int = 5) -> list[SearchResult]:
    """Search ChromaDB for the k most relevant chunks.

    Args:
        query: the employee's question
        k: number of chunks to retrieve (default 5)
           We retrieve 5 and later rerank — better to retrieve
           more and filter down than miss the right chunk.

    Returns:
        List of SearchResult sorted by similarity (highest first)
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"], # type: ignore
    )

    search_results = []

    # ChromaDB returns nested lists (one per query) — we sent one query
    raw_documents = results.get("documents") or []
    raw_metadatas = results.get("metadatas") or []
    raw_distances = results.get("distances") or []

    if not raw_documents:
        log.warning("no_results_found", query=query[:50])
        return []

    documents = raw_documents[0]
    metadatas = raw_metadatas[0]
    distances = raw_distances[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        similarity = round(1 - distance, 4)
        result = SearchResult(
            content=doc,
            source=str(meta["source"]),
            title=str(meta["title"]),
            chunk_index=int(meta["chunk_index"]),
            similarity=similarity,
        )
        search_results.append(result)

    log.info(
        "search_complete",
        query=query[:50],
        results_returned=len(search_results),
        top_similarity=search_results[0].similarity if search_results else 0,
    )

    return search_results
