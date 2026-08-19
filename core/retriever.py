from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from pydantic import Field


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_FETCH_K     = 12
DEFAULT_FINAL_K     = 4


class DocuLegalRetriever(BaseRetriever):
    """
    Custom retriever combining:
    1. Hybrid search   (BM25 + Vector similarity)
    2. Cross-encoder   reranking
    3. Parent document retrieval
    """

    vectorstore     : Chroma        = Field()
    docstore        : InMemoryStore = Field()
    all_documents   : List[Document]= Field()
    fetch_k         : int           = Field(default=DEFAULT_FETCH_K)
    final_k         : int           = Field(default=DEFAULT_FINAL_K)
    use_reranking   : bool          = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:

        candidates = self._hybrid_search(query)

        if self.use_reranking and len(candidates) > self.final_k:
            candidates = self._rerank(query, candidates)

        parent_docs = self._fetch_parents(candidates)

        return parent_docs


    def _hybrid_search(self, query: str) -> List[Document]:
        """Combine BM25 and vector search results."""

        dense_retriever = self.vectorstore.as_retriever(
            search_type     = "mmr",
            search_kwargs   = {
                "k"             : self.fetch_k,
                "fetch_k"       : self.fetch_k * 2,
                "lambda_mult"   : 0.6
            }
        )

        sparse_retriever = BM25Retriever.from_documents(self.all_documents)
        sparse_retriever.k = self.fetch_k

        hybrid = EnsembleRetriever(
            retrievers  = [sparse_retriever, dense_retriever],
            weights     = [0.4, 0.6]
        )

        results = hybrid.invoke(query)
        return results


    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder for higher precision."""

        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        pairs = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)

        scored_docs = sorted(
            zip(docs, scores),
            key     = lambda x: x[1],
            reverse = True
        )

        return [doc for doc, _ in scored_docs[:self.final_k]]


    def _fetch_parents(self, child_docs: List[Document]) -> List[Document]:
        """
        For each retrieved child chunk, fetch its parent document.
        Deduplicate — multiple children from the same parent
        return that parent only once.
        """

        seen_parent_ids     : set           = set()
        parent_documents    : List[Document]= []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")

            if parent_id is None:
                parent_documents.append(child)
                continue

            if parent_id in seen_parent_ids:
                continue

            parent = self.docstore.mget([parent_id])[0]

            if parent is not None:
                seen_parent_ids.add(parent_id)
                parent_documents.append(parent)
            else:
                parent_documents.append(child)

        return parent_documents


def build_retriever(
    vectorstore     : Chroma,
    docstore        : InMemoryStore,
    all_documents   : List[Document],
    fetch_k         : int   = DEFAULT_FETCH_K,
    final_k         : int   = DEFAULT_FINAL_K,
    use_reranking   : bool  = True
) -> DocuLegalRetriever:
    """Build and return the DocuLegal retriever."""

    retriever = DocuLegalRetriever(
        vectorstore     = vectorstore,
        docstore        = docstore,
        all_documents   = all_documents,
        fetch_k         = fetch_k,
        final_k         = final_k,
        use_reranking   = use_reranking
    )

    print(f"Retriever built — fetch_k:{fetch_k}, final_k:{final_k}, "
          f"reranking:{'on' if use_reranking else 'off'}")

    return retriever