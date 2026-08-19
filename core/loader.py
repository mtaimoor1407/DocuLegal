import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from models.schemas import DocumentInfo


def load_pdf(file_path: str) -> Tuple[List[Document], DocumentInfo]:
    """
    Load a single PDF file and return its documents and metadata.
    Returns a tuple of (documents, document_info).
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"File must be a PDF: {file_path}")

    filename = os.path.basename(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No text could be extracted from: {filename}")

    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["file_path"] = file_path

    doc_info = DocumentInfo(
        filename=filename,
        page_count=len(documents),
        chunk_count=0,
        doc_type="legal"
    )

    print(f"Loaded: {filename} ({len(documents)} pages)")
    return documents, doc_info


def load_multiple_pdfs(
    file_paths: List[str]
) -> Tuple[List[Document], List[DocumentInfo]]:
    """
    Load multiple PDF files.
    Returns all documents combined and a list of DocumentInfo objects.
    """

    all_documents   : List[Document]     = []
    all_doc_infos   : List[DocumentInfo] = []

    for file_path in file_paths:
        try:
            documents, doc_info = load_pdf(file_path)
            all_documents.extend(documents)
            all_doc_infos.append(doc_info)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not all_documents:
        raise ValueError("No documents could be loaded from the provided files.")

    print(f"\nTotal pages loaded: {sum(d.page_count for d in all_doc_infos)}")
    print(f"Total documents   : {len(all_doc_infos)}")

    return all_documents, all_doc_infos


def save_uploaded_file(uploaded_file, save_dir: str = "uploaded_docs") -> str:
    """
    Save a Streamlit uploaded file object to disk.
    Returns the path where the file was saved.
    """

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    print(f"Saved uploaded file to: {file_path}")
    return file_path