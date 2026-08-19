from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.schemas import DocumentInfo


LEGAL_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\nSection",
    "\nClause",
    "\nArticle",
    "\nPart ",
    r"\n\d+\.",
    "\n",
    " ",
    ""
]


def create_parent_child_chunks(
    documents       : List[Document],
    parent_size     : int = 1500,
    child_size      : int = 300,
    parent_overlap  : int = 100,
    child_overlap   : int = 30,
) -> Tuple[List[Document], List[Document]]:
    """
    Split documents into parent and child chunks.
    
    Parent chunks: large, full-context sections fed to LLM
    Child chunks:  small, focused pieces indexed in vector store
    
    Returns (parent_chunks, child_chunks)
    """

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size      = parent_size,
        chunk_overlap   = parent_overlap,
        separators      = LEGAL_SEPARATORS,
        length_function = len,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size      = child_size,
        chunk_overlap   = child_overlap,
        separators      = LEGAL_SEPARATORS,
        length_function = len,
    )

    parent_chunks : List[Document] = []
    child_chunks  : List[Document] = []

    for parent_doc in parent_splitter.split_documents(documents):

        parent_id = f"{parent_doc.metadata.get('filename', 'unknown')}_" \
                    f"p{parent_doc.metadata.get('page', 0)}_" \
                    f"c{len(parent_chunks)}"

        parent_doc.metadata["parent_id"]     = parent_id
        parent_doc.metadata["chunk_type"]    = "parent"
        parent_doc.metadata["char_count"]    = len(parent_doc.page_content)

        parent_chunks.append(parent_doc)

        children = child_splitter.split_documents([parent_doc])

        for i, child in enumerate(children):
            child.metadata["parent_id"]   = parent_id
            child.metadata["child_index"] = i
            child.metadata["chunk_type"]  = "child"
            child.metadata["char_count"]  = len(child.page_content)
            child_chunks.append(child)

    print(f"Parent chunks created : {len(parent_chunks)}")
    print(f"Child chunks created  : {len(child_chunks)}")
    print(f"Avg children/parent   : {len(child_chunks)/max(len(parent_chunks),1):.1f}")

    return parent_chunks, child_chunks


def update_doc_info_chunk_count(
    doc_infos   : List[DocumentInfo],
    child_chunks: List[Document]
) -> List[DocumentInfo]:
    """
    Update each DocumentInfo with the actual chunk count
    for that document.
    """

    for doc_info in doc_infos:
        count = sum(
            1 for chunk in child_chunks
            if chunk.metadata.get("filename") == doc_info.filename
        )
        doc_info.chunk_count = count

    return doc_infos