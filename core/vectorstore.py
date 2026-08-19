import os
import shutil
import time
import gc
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_PERSIST_DIR  = "doculegal_chroma_db"
COLLECTION_NAME     = "doculegal_chunks"
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the embedding model."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        encode_kwargs = {"normalize_embeddings": True}
    )
    print("Embedding model ready.")
    return embeddings


def _safe_delete_chroma_dir(path: str) -> None:
    """
    Safely delete Chroma directory on Windows.
    Windows holds file locks so we retry with a short delay.
    Falls back to a fresh collection name if deletion keeps failing.
    """
    if not os.path.exists(path):
        return

    gc.collect()

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(path)
            print("Cleared existing vector store.")
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                print(f"Waiting for file lock release... "
                      f"(attempt {attempt + 1}/{max_attempts})")
                time.sleep(1)
            else:
                print("Could not delete old store. Switching to a fresh collection.")
                _use_fresh_collection()


def _use_fresh_collection() -> None:
    """
    Fallback when the old Chroma directory cannot be deleted.
    Changes the global collection name so a new collection is
    created inside the existing directory without touching locked files.
    """
    global COLLECTION_NAME
    import uuid
    COLLECTION_NAME = f"doculegal_{uuid.uuid4().hex[:8]}"
    print(f"Using new collection: {COLLECTION_NAME}")


def build_vectorstore(
    child_chunks  : List[Document],
    parent_chunks : List[Document],
    embeddings    : HuggingFaceEmbeddings,
    reset         : bool = False
) -> Tuple[Chroma, InMemoryStore]:
    """
    Build the Chroma vector store from child chunks
    and the InMemoryStore from parent chunks.

    child_chunks  → indexed in Chroma (searched by similarity)
    parent_chunks → stored in InMemoryStore (retrieved by parent_id)
    """

    if reset:
        _safe_delete_chroma_dir(CHROMA_PERSIST_DIR)

    print(f"Building vector store with {len(child_chunks)} child chunks...")
    vectorstore = Chroma.from_documents(
        documents         = child_chunks,
        embedding         = embeddings,
        persist_directory = CHROMA_PERSIST_DIR,
        collection_name   = COLLECTION_NAME
    )
    print(f"Vector store built. {vectorstore._collection.count()} vectors stored.")

    print(f"Building document store with {len(parent_chunks)} parent chunks...")
    docstore    = InMemoryStore()
    parent_dict = {
        chunk.metadata["parent_id"]: chunk
        for chunk in parent_chunks
    }
    docstore.mset(list(parent_dict.items()))
    print(f"Document store built. {len(parent_dict)} parent chunks stored.")

    return vectorstore, docstore


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings
) -> Optional[Chroma]:
    """
    Load an existing Chroma vector store from disk.
    Returns None if no store exists or if it is empty.
    """

    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    vectorstore = Chroma(
        persist_directory  = CHROMA_PERSIST_DIR,
        embedding_function = embeddings,
        collection_name    = COLLECTION_NAME
    )

    count = vectorstore._collection.count()
    if count == 0:
        return None

    print(f"Loaded existing vector store ({count} vectors).")
    return vectorstore


def close_vectorstore(vectorstore: Optional[Chroma]) -> None:
    """
    Explicitly close the Chroma client to release Windows file locks.
    Always call this before rebuilding the store.
    """
    if vectorstore is None:
        return
    try:
        vectorstore._client.close()
        print("Vector store connection closed.")
    except Exception:
        pass
    finally:
        gc.collect()
        time.sleep(0.5)


def add_documents_to_store(
    vectorstore   : Chroma,
    docstore      : InMemoryStore,
    child_chunks  : List[Document],
    parent_chunks : List[Document]
) -> None:
    """
    Add new document chunks to existing stores.
    Used when a user uploads additional documents.
    """

    vectorstore.add_documents(child_chunks)

    parent_dict = {
        chunk.metadata["parent_id"]: chunk
        for chunk in parent_chunks
    }
    docstore.mset(list(parent_dict.items()))

    print(f"Added {len(child_chunks)} child chunks to vector store.")
    print(f"Added {len(parent_dict)} parent chunks to document store.")


def get_store_stats(
    vectorstore : Chroma,
    docstore    : InMemoryStore
) -> Dict:
    """Return basic statistics about the current stores."""

    child_count  = vectorstore._collection.count()
    all_keys     = list(docstore.yield_keys())
    parent_count = len(all_keys)

    filenames = set()
    results   = vectorstore._collection.get(include=["metadatas"])
    for metadata in results["metadatas"]:
        if "filename" in metadata:
            filenames.add(metadata["filename"])

    return {
        "child_chunks"  : child_count,
        "parent_chunks" : parent_count,
        "documents"     : list(filenames),
        "doc_count"     : len(filenames)
    }