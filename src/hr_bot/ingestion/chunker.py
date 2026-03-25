from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import structlog

from hr_bot.ingestion.loader import Document

log = structlog.get_logger()

# 512 tokens * ~4 chars/token = ~2048 chars
# 50 token overlap * ~4 chars/token = ~200 chars
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    """A single chunk of text ready for embedding.

    Carries full metadata from its parent document so ChromaDB
    can store and filter by source, title, and chunk index.
    chunk_index tells us the position of this chunk within
    its source document — useful for debugging retrieval.
    """
    content: str
    source: str       # e.g. "benefits-and-perks.md"
    title: str        # e.g. "Benefits and Perks"
    chunk_index: int  # position within the source document


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Split documents into overlapping chunks with metadata preserved.

    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Paragraphs (double newline) first
    2. Then sentences
    3. Then words
    4. Then characters as last resort

    This hierarchy means we never split a paragraph arbitrarily —
    we always try to find a natural boundary first.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Chunk] = []

    for doc in documents:
        # Split the document text into raw string chunks
        raw_chunks = splitter.split_text(doc.content)

        # Wrap each raw string in a Chunk dataclass with metadata
        for i, raw_chunk in enumerate(raw_chunks):
            # Skip chunks that are too short to be meaningful
            if len(raw_chunk.strip()) < 50:
                continue

            chunk = Chunk(
                content=raw_chunk.strip(),
                source=doc.source,
                title=doc.title,
                chunk_index=i,
            )
            all_chunks.append(chunk)

        log.info(
            "document_chunked",
            source=doc.source,
            total_chunks=len(raw_chunks),
        )

    log.info("chunking_complete", total_chunks=len(all_chunks))
    return all_chunks