from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

KB_ROOT = Path("knowledge_base")
DGCA_RULES_DIR = Path("dgca_rules")
RULE_SOURCE_NAMES = ("guide.pdf", "guide2.pdf", "guide3.pdf", "guide4.pdf")
GUIDE_PDF_NAME = "guide.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 6
RETRIEVAL_THRESHOLD = 0.9
AZURE_API_VERSION_DEFAULT = "2024-02-15-preview"

RULE_KEYWORDS = ("regulation", "dgca", "fdtl", "rule", "clause", "legal", "compliance", "reporting time", "report time", "reporting", "duty time", "duty period", "rest requirement", "minimum rest", "flying hour", "flight time limit")
SOP_KEYWORDS = ("sop", "procedure", "protocol", "standard operating", "guideline", "report for duty", "sign on", "sign-on", "duty start", "check in time")
UNION_KEYWORDS = ("union", "agreement", "contract", "collective", "leave policy")
AIRPORT_KEYWORDS = ("airport", "curfew", "slot", "gate", "terminal")
TRAINING_KEYWORDS = ("training", "recurrency", "type rating", "check ride", "proficiency")

DEFINITION_ALIASES = {
    "report time": "reporting time",
    "report for duty": "reporting time",
    "rest": "rest period",
    "rest period": "rest period",
    "flight duty period": "flight duty period",
    "duty period": "duty period",
    "bunk": "bunk",
}

GUIDE_HEADING_HINTS = {
    "bunk": [r"^\s*3\.16\.1\s+bunk\b"],
    "reporting time": [r"^\s*3\.14\s+reporting time\b"],
    "rest period": [r"^\s*3\.15\s+rest period\b"],
    "duty period": [r"^\s*3\.5\.1\s+duty period\b"],
    "flight duty period": [r"^\s*3\.5\.2\s+flight duty period\b"],
}


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
        if not context.strip():
            return "The retrieved context does not contain sufficient information to answer this question."
        return context.strip()


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


def _import_pdf_loader():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader
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


def _normalize_text_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def _extract_definition_subject(query: str) -> Optional[str]:
    lowered = query.lower().strip()
    lowered = lowered.rstrip("?.!")
    lowered = re.sub(r"^(what is|what are|define|definition of|meaning of)\s+", "", lowered)
    lowered = re.sub(r"\s+(according to|as per|in the guide|in guide|from the guide|in dgca).*", "", lowered)
    lowered = re.sub(r"\s+(dgca|guide|pdf)$", "", lowered).strip()
    lowered = lowered.replace("what's ", "")
    lowered = lowered.replace(",", " ")
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered or None


def _guide_pdf_docs() -> list:
    guide = Path(GUIDE_PDF_NAME)
    if not guide.exists():
        return []
    return load_documents([guide])


def _pick_definition_block(page_text: str, subject: str) -> Optional[str]:
    lines = [line.rstrip() for line in page_text.splitlines()]
    subject_lower = subject.lower().strip()
    alias = DEFINITION_ALIASES.get(subject_lower, subject_lower)
    target_terms = [alias]
    if alias != subject_lower:
        target_terms.append(subject_lower)

    for term in (alias, subject_lower):
        for pattern in GUIDE_HEADING_HINTS.get(term, []):
            regex = re.compile(pattern, re.IGNORECASE)
            for index, line in enumerate(lines):
                if not regex.search(line):
                    continue

                block = [line.strip()]
                for follow in lines[index + 1 : index + 8]:
                    follow_text = follow.strip()
                    if not follow_text:
                        continue
                    if re.match(r"^\s*\d+(?:\.\d+)*\s+\S+", follow_text) and not regex.search(follow_text):
                        break
                    block.append(follow_text)

                snippet = " ".join(block).strip()
                if snippet:
                    return snippet

    for index, line in enumerate(lines):
        line_lower = line.lower()
        if not any(term in line_lower for term in target_terms):
            continue

        if not re.search(r"\b(means|is|shall be|commences|refers to)\b", line_lower):
            if index > 0 and not re.search(r"^\s*\d+(?:\.\d+)*\s+", line):
                continue

        block = [line.strip()]
        for follow in lines[index + 1 : index + 7]:
            follow_text = follow.strip()
            if not follow_text:
                continue
            if re.match(r"^\s*\d+(?:\.\d+)*\s+\S+", follow_text) and not any(term in follow_text.lower() for term in target_terms):
                break
            block.append(follow_text)

        snippet = " ".join(block).strip()
        if snippet:
            return snippet

    return None


def _pick_exact_definition_block(page_text: str, subject: str) -> Optional[str]:
    lines = [line.rstrip() for line in page_text.splitlines()]
    subject_lower = subject.lower().strip()
    alias = DEFINITION_ALIASES.get(subject_lower, subject_lower)

    for term in (alias, subject_lower):
        for pattern in GUIDE_HEADING_HINTS.get(term, []):
            regex = re.compile(pattern, re.IGNORECASE)
            for index, line in enumerate(lines):
                if not regex.search(line):
                    continue

                block = [line.strip()]
                for follow in lines[index + 1 : index + 8]:
                    follow_text = follow.strip()
                    if not follow_text:
                        continue
                    if re.match(r"^\s*\d+(?:\.\d+)*\s+\S+", follow_text) and not regex.search(follow_text):
                        break
                    block.append(follow_text)

                snippet = " ".join(block).strip()
                if snippet:
                    return snippet
    return None


def _guide_definition_lookup(query: str) -> Optional[str]:
    subject = _extract_definition_subject(query)
    if not subject:
        return None

    docs = _guide_pdf_docs()
    if not docs:
        return None

    best_snippet = None
    best_score = 0
    subject_terms = _normalize_text_terms(subject)

    for document in docs:
        page_text = getattr(document, "page_content", "") or ""
        exact_snippet = _pick_exact_definition_block(page_text, subject)
        if exact_snippet:
            if len(exact_snippet) > 1200:
                exact_snippet = exact_snippet[:1200].rstrip() + "..."
            return exact_snippet

    for document in docs:
        page_text = getattr(document, "page_content", "") or ""
        page_lower = page_text.lower()
        alias = DEFINITION_ALIASES.get(subject, subject)

        score = 0
        if alias in page_lower:
            score += 8
        if subject in page_lower:
            score += 5
        score += len(subject_terms & _normalize_text_terms(page_text))

        if score == 0:
            continue

        snippet = _pick_definition_block(page_text, subject)
        if not snippet:
            continue

        if score > best_score:
            best_score = score
            best_snippet = snippet

    if not best_snippet:
        return None

    if len(best_snippet) > 1200:
        best_snippet = best_snippet[:1200].rstrip() + "..."
    return best_snippet


def _resolve_rule_sources(pdf_folder: Optional[Path] = None) -> List[Path]:
    if pdf_folder is not None:
        if pdf_folder.is_file():
            return [pdf_folder] if pdf_folder.suffix.lower() == ".pdf" else []
        if pdf_folder.exists():
            return sorted(pdf_folder.glob("*.pdf"))
        return []

    sources: dict[str, Path] = {}
    for name in RULE_SOURCE_NAMES:
        path = Path(name)
        if path.exists() and path.suffix.lower() == ".pdf":
            sources[path.name.lower()] = path

    if DGCA_RULES_DIR.exists():
        for path in sorted(DGCA_RULES_DIR.glob("*.pdf")):
            sources.setdefault(path.name.lower(), path)

    return list(sources.values())


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


def load_documents(pdf_sources: Optional[List[Path]] = None) -> list:
    pdf_loader_cls = _import_pdf_loader()
    directory_loader_cls = _import_document_loader()
    if pdf_loader_cls is None and directory_loader_cls is None:
        return []

    sources = pdf_sources or _resolve_rule_sources()
    all_docs = []
    for source in sources:
        if not source.exists():
            continue
        if source.is_file():
            if pdf_loader_cls is None:
                continue
            all_docs.extend(pdf_loader_cls(str(source)).load())
        elif source.is_dir() and directory_loader_cls is not None:
            all_docs.extend(directory_loader_cls(str(source)).load())
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


def _score_document(query_terms: set[str], query_text: str, document_text: str) -> int:
    if not document_text.strip():
        return 0

    text_terms = _normalize_text_terms(document_text)
    score = len(query_terms & text_terms)

    lowered_query = query_text.lower().strip()
    lowered_text = document_text.lower()
    if lowered_query and lowered_query in lowered_text:
        score += 12

    phrase_hits = 0
    for phrase in re.findall(r"[a-z0-9]+(?:\s+[a-z0-9]+){1,4}", lowered_query):
        if phrase in lowered_text:
            phrase_length = len(phrase.split())
            phrase_hits += max(1, phrase_length - 1)
    score += min(phrase_hits, 12)

    digit_hits = 0
    for digit in re.findall(r"\d+(?:\.\d+)?", query_text):
        if digit in document_text:
            digit_hits += 1
    score += digit_hits * 2

    if any(marker in lowered_text for marker in (" shall ", " means ", " must ", " is defined ", " is a period ")):
        score += 2

    if any(token in lowered_query for token in ("definition", "define", "defined")) and any(
        marker in lowered_text for marker in (" means ", " is defined ", " is a period ", " is an ")
    ):
        score += 4

    if "definition" in lowered_query and "flight duty period" in lowered_text and any(
        marker in lowered_text for marker in (" is a period", "commences when", "means ")
    ):
        score += 6

    if "rest" in lowered_query and any(marker in lowered_text for marker in ("minimum rest", "rest before", "rest shall", "shall be given")):
        score += 5

    if "duty period" in lowered_query and any(marker in lowered_text for marker in ("flight duty period", "fdp", "duty period")):
        score += 3

    return score


def _rank_documents(query: str, documents: Sequence) -> list:
    query_terms = _normalize_text_terms(query)
    ranked = []
    for index, document in enumerate(documents):
        text = getattr(document, "page_content", "") or ""
        score = _score_document(query_terms, query, text)
        if score > 0:
            ranked.append((score, index, document))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in ranked]


def _score_sentence(query_terms: set[str], query_text: str, sentence: str) -> int:
    return _score_document(query_terms, query_text, sentence)


def _best_excerpt(context: str, query: str) -> str:
    best_label = ""
    best_sentence = ""
    best_score = 0
    lowered_query = query.lower()

    if "definition" in lowered_query and "flight duty period" in lowered_query:
        for block in [part.strip() for part in context.split("\n\n") if part.strip()]:
            lines = block.splitlines()
            if len(lines) < 2:
                continue
            label = lines[0].strip()
            body = "\n".join(line.rstrip() for line in lines[1:] if line.strip())
            lowered_body = body.lower()
            for marker in (
                "3.5.2 flight duty period (fdp)",
                "flight duty period (fdp)",
                "flight duty period",
            ):
                start = lowered_body.find(marker)
                if start == -1:
                    continue
                end_candidates = [
                    lowered_body.find("3.5.3", start),
                    lowered_body.find("3.6", start),
                    lowered_body.find("3.5.4", start),
                ]
                end_candidates = [idx for idx in end_candidates if idx != -1]
                end = min(end_candidates) if end_candidates else min(len(body), start + 1100)
                snippet = body[start:end].strip()
                if len(snippet) > 900:
                    snippet = snippet[:900].rstrip() + "..."
                return f"{snippet}\n\nSource: {label}"

    for block in [part.strip() for part in context.split("\n\n") if part.strip()]:
        lines = block.splitlines()
        if not lines:
            continue
        label = lines[0].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        if not body:
            continue
        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", body) if piece.strip()]
        if not sentences:
            sentences = [body.strip()]
        for sentence in sentences:
            score = _score_sentence(_normalize_text_terms(query), query, sentence)
            if score > best_score:
                best_score = score
                best_label = label
                best_sentence = sentence

    if not best_sentence:
        first_block = context.strip().split("\n\n")[0].strip()
        return first_block[:1400].rstrip() + ("..." if len(first_block) > 1400 else "")

    if len(best_sentence) > 900:
        best_sentence = best_sentence[:900].rstrip() + "..."

    if best_label:
        return f"{best_sentence}\n\nSource: {best_label}"
    return best_sentence


def _extractive_answer(query: str, context: str, sources: List[str]) -> str:
    if not context.strip():
        return "The retrieved context does not contain sufficient information to answer this question."

    answer = _best_excerpt(context, query)

    if sources and "Source:" not in answer:
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)
        source_text = ", ".join(unique_sources[:3])
        return f"{answer}\n\nSource: {source_text}"

    return answer


@lru_cache(maxsize=12)
def _build_vectorstore_cached(source_key: tuple[str, ...], mode: str, folder_mtime_ns: int):
    FAISS = _import_faiss()
    if FAISS is None:
        return None
    documents = load_documents([Path(item) for item in source_key])
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
    sources = _resolve_rule_sources(pdf_folder)
    source_key = tuple(str(path.resolve()) for path in sources)
    folder_mtime_ns = max((path.stat().st_mtime_ns for path in sources), default=0)
    mode = "azure" if _has_azure_credentials() else "ollama"
    return _build_vectorstore_cached(source_key, mode, folder_mtime_ns)


def clear_vectorstore_cache():
    _build_vectorstore_cached.cache_clear()


def retrieve_legal_guidance(
    query: str,
    pdf_folder: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str] = None,
) -> str:
    guide_snippet = _guide_definition_lookup(query)
    if guide_snippet:
        return guide_snippet

    sources = _resolve_rule_sources(pdf_folder)
    if not sources:
        return DummyLLM().synthesize(query, "")

    documents = split_documents(load_documents(sources))
    if not documents:
        return DummyLLM().synthesize(query, "")

    relevant_docs = _rank_documents(query, documents)[:top_k]

    if not relevant_docs:
        vectorstore = build_vectorstore(pdf_folder)
        if vectorstore is not None:
            try:
                relevant_docs = [doc for doc, _ in vectorstore.similarity_search_with_score(query, k=top_k)]
            except Exception:
                relevant_docs = []

    context, sources = _format_context(relevant_docs)
    if not context:
        return DummyLLM().synthesize(query, "")

    return _extractive_answer(query, context, sources)


def retrieve_legal_guidance_with_sources(
    query: str,
    pdf_folder: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:
    auto_category = _categorize_query(query)
    answer = retrieve_legal_guidance(query, pdf_folder=pdf_folder, top_k=top_k, category=auto_category)
    vectorstore = build_vectorstore(pdf_folder)
    sources: List[str] = []
    context = ""
    mode = "extractive"
    if vectorstore is not None:
        try:
            raw_docs = vectorstore.similarity_search_with_score(query, k=top_k)
            filtered = [doc for doc, _ in raw_docs]
            context, sources = _format_context(filtered)
            mode = "azure" if _has_azure_credentials() else "ollama"
        except Exception:
            pass
    return RetrievalResult(
        answer=answer, context=context, sources=sources,
        mode=mode, category=auto_category,
    )
