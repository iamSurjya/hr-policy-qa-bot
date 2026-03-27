import argparse
import sys

import structlog

from hr_bot.config import settings
from hr_bot.ingestion.chunker import chunk_documents
from hr_bot.ingestion.indexer import build_index
from hr_bot.ingestion.loader import load_documents

logger = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HR policy documents into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and rebuild the index from scratch.",
    )
    args = parser.parse_args()

    logger.info("ingest_started", docs_path=settings.policy_docs_path, reset=args.reset)

    try:
        docs = load_documents(settings.policy_docs_path)
        logger.info("documents_loaded", count=len(docs))

        chunks = chunk_documents(docs)
        logger.info("documents_chunked", count=len(chunks))

        build_index(chunks, reset=args.reset)
        logger.info("ingest_complete", chunks_indexed=len(chunks))

    except Exception as e:
        logger.error("ingest_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()