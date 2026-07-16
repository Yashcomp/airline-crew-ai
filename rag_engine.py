from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence


DGCA_RULES_DIR = Path("dgca_rules")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
AZURE_API_VERSION_DEFAULT = "2024-02-15-preview"


class DummyLLM:
    def synthesize(self, query: str, context: str) -> str:
        return (
            "Mock RAG Response: According to Clause 7.4, the retrieved DGCA guidance indicates the "
            "requested assignment must remain within legal duty limits and rest requirements.\n\n"
            f"Retrieved context:\n{context}"
        )


@dataclass(frozen=True)
class RetrievalResult:
    answer: str
    context: str
    sources: List[str]
    mode: str


def _normalize_folder(pdf_folder: str | Path) -> Path:
    folder = Path(pdf_folder)
    if folder.is_file():
        return folder.parent
    return folder


def _has_azure_credentials() -> bool:
    return bool(
        os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        and os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    )


def _import_document_loader():
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader

        return PyPDFDirectoryLoader
    except ImportError:  # pragma: no cover - import guard for restricted PCs
        return None


def _import_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter
        except ImportError:  # pragma: no cover - import guard for restricted PCs
            return None


def _import_faiss():
    try:
        from langchain_community.vectorstores import FAISS

        return FAISS
    except ImportError:  # pragma: no cover - import guard for restricted PCs
        return None


def _import_fake_embeddings():
    try:
        from langchain_core.embeddings import FakeEmbeddings

        return FakeEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import FakeEmbeddings

            return FakeEmbeddings
        except ImportError:  # pragma: no cover - import guard for restricted PCs
            return None


def _import_azure_embeddings():
    try:
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings
    except ImportError:  # pragma: no cover - import guard for restricted PCs
        return None


def _import_azure_chat_model():
    try:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI
    except ImportError:  # pragma: no cover - import guard for restricted PCs
        return None


def load_documents(pdf_folder: str | Path = DGCA_RULES_DIR):
    folder = _normalize_folder(pdf_folder)
    if not folder.exists():
        return []
    loader_cls = _import_document_loader()
    if loader_cls is None:
        return []
    loader = loader_cls(str(folder))
    return loader.load()


def split_documents(documents: Sequence) -> List:
    if not documents:
        return []
    splitter_cls = _import_splitter()
    if splitter_cls is None:
        return list(documents)
    splitter = splitter_cls(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(list(documents))


def _build_fake_embeddings():
    FakeEmbeddings = _import_fake_embeddings()
    if FakeEmbeddings is None:
        return None
    return FakeEmbeddings(size=1536)


def _build_azure_embeddings():
    AzureOpenAIEmbeddings = _import_azure_embeddings()
    if AzureOpenAIEmbeddings is None:
        return None
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION_DEFAULT),
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
    )


def _build_llm():
    if not _has_azure_credentials():
        return DummyLLM(), "offline"
    AzureChatOpenAI = _import_azure_chat_model()
    if AzureChatOpenAI is None:
        return DummyLLM(), "offline"
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION_DEFAULT),
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )
    return model, "azure"


def _document_page_label(document) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    source = Path(str(metadata.get("source", "DGCA PDF"))).name
    page = metadata.get("page")
    if page is None:
        return source
    return f"{source} p.{int(page) + 1}"


def _format_context(documents: Iterable) -> tuple[str, List[str]]:
    chunks: List[str] = []
    sources: List[str] = []
    for index, document in enumerate(documents, start=1):
        label = _document_page_label(document)
        sources.append(label)
        text = getattr(document, "page_content", "").strip()
        if text:
            chunks.append(f"[{index}] {label}\n{text}")
    return "\n\n".join(chunks), sources


@lru_cache(maxsize=8)
def _build_vectorstore_cached(folder_str: str, mode: str, folder_mtime_ns: int):
    FAISS = _import_faiss()
    if FAISS is None:
        return None
    documents = load_documents(folder_str)
    if not documents:
        return None
    chunks = split_documents(documents)
    if not chunks:
        return None
    embeddings = _build_azure_embeddings() if mode == "azure" else _build_fake_embeddings()
    if embeddings is None:
        return None
    return FAISS.from_documents(chunks, embeddings)


def build_vectorstore(pdf_folder: str | Path = DGCA_RULES_DIR):
    folder = _normalize_folder(pdf_folder)
    folder_mtime_ns = 0
    if folder.exists():
        folder_mtime_ns = max((path.stat().st_mtime_ns for path in folder.glob("**/*.pdf")), default=0)
    mode = "azure" if _has_azure_credentials() else "offline"
    return _build_vectorstore_cached(str(folder), mode, folder_mtime_ns)


def retrieve_legal_guidance(query: str, pdf_folder: str | Path = DGCA_RULES_DIR, top_k: int = DEFAULT_TOP_K) -> str:
    folder = _normalize_folder(pdf_folder)
    if not folder.exists():
        return DummyLLM().synthesize(query, "DGCA rules folder not found.")

    vectorstore = build_vectorstore(folder)
    if vectorstore is None:
        return DummyLLM().synthesize(query, "No DGCA PDF pages were loaded or the local retrieval stack is unavailable.")

    try:
        relevant_docs = vectorstore.similarity_search(query, k=top_k)
    except Exception:
        relevant_docs = []

    context, sources = _format_context(relevant_docs)
    if not context:
        return DummyLLM().synthesize(query, "No relevant DGCA clauses were retrieved.")

    llm, mode = _build_llm()
    if mode == "offline":
        return llm.synthesize(query, context)

    prompt = (
        "You are a DGCA compliance assistant. Use only the provided context to answer in plain English. "
        "Cite the most relevant clause references when possible.\n\n"
        f"User scenario: {query}\n\n"
        f"Context:\n{context}"
    )

    try:
        response = llm.invoke(prompt)
        answer = getattr(response, "content", None) or str(response)
    except Exception:
        answer = DummyLLM().synthesize(query, context)

    return answer if answer.strip() else DummyLLM().synthesize(query, context)


def retrieve_legal_guidance_with_sources(
    query: str, pdf_folder: str | Path = DGCA_RULES_DIR, top_k: int = DEFAULT_TOP_K
) -> RetrievalResult:
    answer = retrieve_legal_guidance(query, pdf_folder=pdf_folder, top_k=top_k)
    folder = _normalize_folder(pdf_folder)
    vectorstore = build_vectorstore(folder)
    sources: List[str] = []
    context = ""
    mode = "offline"
    if vectorstore is not None:
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
            context, sources = _format_context(docs)
            mode = "azure" if _has_azure_credentials() else "offline"
        except Exception:
            pass
    return RetrievalResult(answer=answer, context=context, sources=sources, mode=mode)