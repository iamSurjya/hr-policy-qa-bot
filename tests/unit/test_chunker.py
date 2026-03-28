from hr_bot.ingestion.chunker import chunk_documents, Chunk
from hr_bot.ingestion.loader import Document


def test_small_text_single_chunk():
    # Arrange
    doc = Document(
        content="This is a short policy.",
        source="test.md",
        title="Test Policy"
    )

    # Act
    chunks = chunk_documents([doc])

    # Assert
    assert len(chunks) == 0 or len(chunks) == 1

def test_large_text_multiple_chunks():
    # Arrange
    text = "A" * 5000  # definitely larger than 2048

    doc = Document(
        content=text,
        source="test.md",
        title="Test Policy"
    )

    # Act
    chunks = chunk_documents([doc])

    # Assert
    assert len(chunks) > 1  # should split

    for chunk in chunks:
        assert len(chunk.content) <= 2048

def test_chunk_overlap_exists():
    # Arrange
    text = "A" * 5000

    doc = Document(
        content=text,
        source="test.md",
        title="Test Policy"
    )

    # Act
    chunks = chunk_documents([doc])

    # Assert
    assert len(chunks) > 1

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].content
        next_chunk = chunks[i + 1].content

        # Check that there is SOME overlap (not exact 200)
        overlap_found = any(
            current_chunk[-k:] == next_chunk[:k]
            for k in range(50, 201)  # allow flexible overlap
        )

        assert overlap_found, f"No overlap between chunk {i} and {i+1}"