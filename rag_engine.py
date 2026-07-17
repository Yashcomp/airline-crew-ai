from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

KB_ROOT = Path("knowledge_base")
DGCA_RULES_DIR = Path("dgca_rules")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
AZURE_API_VERSION_DEFAULT = "2024-02-15-preview"

RULE_KEYWORDS = ("regulation", "dgca", "fdtl", "rule", "clause", "legal", "compliance")
SOP_KEYWORDS = ("sop", "procedure", "protocol", "standard operating", "guideline")
UNION_KEYWORDS = ("union", "agreement", "contract", "collective", "leave policy")
AIRPORT_KEYWORDS = ("airport", "curfew", "slot", "gate", "terminal")
TRAINING_KEYWORDS = ("training", "recurrency", "type rating", "check ride", "proficiency")


def _categorize_query(query: str) -> Optional[str]:
    lowered = query.lower()
    if any(kw in lowered for kw in RULE_KEYWORDS):
        return "regulations"
    if any(kw in lowered for kw in SOP_KEYWORDS):
        return "sops"
    if any(kw in lowered for kw in UNION_KEYWORDS):
        return "union"
    if any(kw in lowered for kw in AIRPORT_KEYWORDS):
        return "airports"
    if any(kw in lowered for kw in TRAINING_KEYWORDS):
        return "training"
    return None


class DummyLLM:
    def synthesize(self, query: str, context: str) -> str:
        return (
            "Mock RAG Response: According to the retrieved DGCA guidance, the "
            "requested assignment must remain within legal duty limits and rest requirements.\n\n"
            f"Retrieved context:\n{context}"
        )


@dataclass(frozen=True)
class RetrievalResult:
    answer: str
    context: str
    sources: List[str]
    mode: str
    category: Optional[str] = None


def _get_pdf_dirs() -> List[Path]:
    dirs = []
    if DGCA_RULES_DIR.exists():
        dirs.append(DGCA_RULES_DIR)
    if KB_ROOT.exists():
        for sub in sorted(KB_ROOT.iterdir()):
            if sub.is_dir() and any(sub.glob("*.pdf")):
                dirs.append(sub)
    return dirs


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
    except ImportError:
        return None


def _import_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter
    except ImportError:
        return None


def _import_faiss():
    try:
        from langchain_community.vectorstores import FAISS
        return FAISS
    except ImportError:
        return None


def _build_embeddings(mode: str):
    if mode == "azure":
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            return AzureOpenAIEmbeddings(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION_DEFAULT),
                azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
            )
        except (ImportError, KeyError):
            pass
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    except ImportError:
        try:
            from langchain_core.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=1536)
        except ImportError:
            return None


def _build_llm():
    if _has_azure_credentials():
        try:
            from langchain_openai import AzureChatOpenAI
            model = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION_DEFAULT),
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                temperature=0,
            )
            return model, "azure"
        except (ImportError, KeyError):
            pass
    try:
        from langchain_ollama import ChatOllama
        model = ChatOllama(model="qwen2.5:3b", temperature=0.0)
        return model, "ollama"
    except ImportError:
        return DummyLLM(), "dummy"


def load_documents(pdf_dirs: Optional[List[Path]] = None) -> list:
    loader_cls = _import_document_loader()
    if loader_cls is None:
        return []
    dirs = pdf_dirs or _get_pdf_dirs()
    all_docs = []
    for d in dirs:
        if d.exists():
            all_docs.extend(loader_cls(str(d)).load())
    return all_docs


def split_documents(documents: Sequence) -> list:
    if not documents:
        return []
    splitter_cls = _import_splitter()
    if splitter_cls is None:
        return list(documents)
    splitter = splitter_cls(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(list(documents))


def _document_page_label(document) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    source = Path(str(metadata.get("source", "PDF"))).name
    page = metadata.get("page")
    if page is None:
        return source
    return f"{source} p.{int(page) + 1}"


def _format_context(documents: Iterable) -> tuple[str, List[str]]:
    chunks: List[str] = []
    sources: List[str] = []
    for idx, document in enumerate(documents, start=1):
        label = _document_page_label(document)
        sources.append(label)
        text = getattr(document, "page_content", "").strip()
        if text:
            chunks.append(f"[{idx}] {label}\n{text}")
    return "\n\n".join(chunks), sources


@lru_cache(maxsize=12)
def _build_vectorstore_cached(folder_str: str, mode: str, folder_mtime_ns: int):
    FAISS = _import_faiss()
    if FAISS is None:
        return None
    documents = load_documents([Path(folder_str)])
    if not documents:
        return None
    chunks = split_documents(documents)
    if not chunks:
        return None
    embeddings = _build_embeddings(mode)
    if embeddings is None:
        return None
    return FAISS.from_documents(chunks, embeddings)


def build_vectorstore(pdf_folder: Optional[Path] = None):
    folder = pdf_folder or DGCA_RULES_DIR
    if not folder.exists():
        folder = DGCA_RULES_DIR
    folder_mtime_ns = 0
    if folder.exists():
        folder_mtime_ns = max(
            (p.stat().st_mtime_ns for p in folder.glob("**/*.pdf")),
            default=0,
        )
    mode = "azure" if _has_azure_credentials() else "ollama"
    return _build_vectorstore_cached(str(folder), mode, folder_mtime_ns)


def retrieve_legal_guidance(
    query: str,
    pdf_folder: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str] = None,
) -> str:
    folder = pdf_folder or DGCA_RULES_DIR
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
    if mode == "dummy":
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
    query: str,
    pdf_folder: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:
    auto_category = _categorize_query(query)
    answer = retrieve_legal_guidance(query, pdf_folder=pdf_folder, top_k=top_k, category=auto_category)
    folder = pdf_folder or DGCA_RULES_DIR
    vectorstore = build_vectorstore(folder)
    sources: List[str] = []
    context = ""
    mode = "ollama"
    if vectorstore is not None:
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
            context, sources = _format_context(docs)
            mode = "azure" if _has_azure_credentials() else "ollama"
        except Exception:
            pass
    return RetrievalResult(
        answer=answer, context=context, sources=sources,
        mode=mode, category=auto_category,
    )
