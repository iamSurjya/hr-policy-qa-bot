import os
from dataclasses import dataclass
import structlog

log = structlog.get_logger()


@dataclass
class Document:
    """A loaded policy document with its content and metadata.

    Metadata is critical — it travels with every chunk so when
    the retriever finds a chunk, we know exactly which file and
    section it came from. This is what lets us tell the user
    'this answer comes from benefits-and-perks.md'.
    """
    content: str
    source: str        # filename e.g. "benefits-and-perks.md"
    title: str         # human readable e.g. "Benefits and Perks"


def filename_to_title(filename: str) -> str:
    """Convert 'benefits-and-perks.md' to 'Benefits and Perks'."""
    name = filename.replace(".md", "").replace(".txt", "")
    return name.replace("-", " ").replace("_", " ").title()


def load_documents(docs_path: str) -> list[Document]:
    """Load all .md and .txt files from the policy docs folder.

    Each file becomes one Document object with its metadata attached.
    We load the full file here — chunking happens separately in
    chunker.py. Single responsibility: this function only loads.
    """
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Policy docs folder not found: {docs_path}\n"
            f"Check POLICY_DOCS_PATH in your .env file."
        )

    documents = []
    skipped = []

    for filename in sorted(os.listdir(docs_path)):
        if not (filename.endswith(".md") or filename.endswith(".txt")):
            skipped.append(filename)
            continue

        filepath = os.path.join(docs_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            log.warning("empty_document_skipped", filename=filename)
            continue

        doc = Document(
            content=content,
            source=filename,
            title=filename_to_title(filename),
        )
        documents.append(doc)
        log.info("document_loaded", filename=filename, chars=len(content))

    if skipped:
        log.debug("non_document_files_skipped", files=skipped)

    log.info("loading_complete", total_documents=len(documents))
    return documents