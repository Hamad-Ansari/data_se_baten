"""Retrieval memory for the agent.

Two kinds of knowledge are indexed:

1. *run knowledge* - facts derived from the run artifacts (profile, quality,
   insights, metrics, feature importance, report); and
2. *domain knowledge* - documents the user drops into ``data/knowledge``
   (markdown, text, CSV), so the agent can answer questions with the project's
   own vocabulary and business rules.

Retrieval prefers a local embedding model (sentence-transformers + FAISS) when
it is installed and falls back to a TF-IDF/BM25-style keyword index, so the
feature works offline and without optional dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.persistence import RunStore
from utils.optional_deps import is_available
from utils.serialization import to_jsonable

logger = get_logger(__name__)

TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml"}
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120


@dataclass
class KnowledgeDocument:
    """One retrievable chunk of knowledge."""

    doc_id: str
    text: str
    source: str
    kind: str = "document"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    cleaned = "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind("\n\n", start, end)
            if boundary == -1 or boundary < start + size * 0.5:
                boundary = cleaned.rfind(". ", start, end)
            if boundary > start + size * 0.4:
                end = boundary + 1
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


class KnowledgeBase:
    """Small hybrid (embedding -> TF-IDF) retriever."""

    def __init__(self, documents: Optional[Iterable[KnowledgeDocument]] = None) -> None:
        self.documents: List[KnowledgeDocument] = list(documents or [])
        self._matrix = None
        self._vectorizer = None
        self._embeddings = None
        self._embedder = None
        self._backend = "tfidf"

    # ------------------------------------------------------------------ build
    @classmethod
    def from_run(cls, store: RunStore, *, include_domain_docs: bool = True) -> "KnowledgeBase":
        """Index everything known about a run."""
        kb = cls()
        kb.add_document(store.run_id, f"Run {store.run_id} analysis knowledge.", source="run", kind="run")
        narrative = store.load_json("narratives.json", default=None)
        for name, label in (
            ("profile.json", "Dataset profile"),
            ("quality_report.json", "Data quality"),
            ("cleaning_log.json", "Cleaning"),
            ("eda.json", "Exploratory analysis"),
            ("problem.json", "Detected task"),
            ("selection.json", "Algorithm selection"),
            ("feature_plan.json", "Features"),
            ("split.json", "Validation strategy"),
            ("evaluation.json", "Evaluation"),
            ("explanation.json", "Explainability"),
            ("error_analysis.json", "Error analysis"),
            ("quality_gate.json", "Quality gate"),
            ("deployment.json", "Deployment"),
            ("monitoring_baseline.json", "Monitoring"),
        ):
            payload = store.load_json(name, default=None)
            if payload:
                kb.add_document(label, json.dumps(to_jsonable(payload), default=str)[:12000],
                                source=name, kind="artifact")
        if narrative:
            kb.add_document("Narratives", json.dumps(to_jsonable(narrative), default=str)[:8000],
                            source="narratives.json", kind="narrative")
        report_path = store.path("report.md", "reports")
        if report_path.exists():
            kb.add_document("Report", report_path.read_text(encoding="utf-8", errors="ignore")[:20000],
                            source="reports/report.md", kind="report")
        if include_domain_docs:
            kb.add_directory(get_settings().knowledge_dir)
        kb.build_index()
        return kb

    def add_document(self, title: str, text: str, *, source: str = "", kind: str = "document",
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        for index, chunk in enumerate(chunk_text(text)):
            self.documents.append(
                KnowledgeDocument(
                    doc_id=f"{source or title}#{index}",
                    text=f"{title}\n{chunk}",
                    source=source or title,
                    kind=kind,
                    metadata={"title": title, **(metadata or {})},
                )
            )

    def add_directory(self, directory: Path | str) -> int:
        """Index every supported document in a directory (recursively)."""
        path = Path(directory)
        if not path.exists():
            return 0
        added = 0
        for file in sorted(path.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # pragma: no cover
                logger.debug("Could not read knowledge file %s: %s", file, exc)
                continue
            before = len(self.documents)
            self.add_document(file.stem.replace("_", " ").title(), text, source=str(file.relative_to(path)),
                              kind="domain")
            added += len(self.documents) - before
        return added

    # ------------------------------------------------------------------ index
    def build_index(self) -> None:
        """Fit the retrieval index (embeddings when available, TF-IDF otherwise)."""
        texts = [document.text for document in self.documents]
        if not texts:
            return
        if is_available("sentence_transformers") and is_available("faiss"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                from config.settings import get_settings as _settings

                model_name = str(getattr(_settings(), "embedding_model", "all-MiniLM-L6-v2"))
                self._embedder = SentenceTransformer(model_name)
                matrix = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                self._embeddings = matrix
                self._backend = f"embeddings:{model_name}"
                return
            except Exception as exc:  # pragma: no cover - optional path
                logger.info("Embedding backend unavailable (%s); using TF-IDF.", exc)
                self._embedder = None
                self._embeddings = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=20000
            )
            self._matrix = self._vectorizer.fit_transform(texts)
            self._backend = "tfidf"
        except Exception as exc:  # pragma: no cover
            logger.debug("TF-IDF index failed: %s", exc)

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> List[Tuple[KnowledgeDocument, float]]:
        """Return the ``k`` most relevant chunks with scores."""
        if not self.documents:
            return []
        if self._embeddings is not None and self._embedder is not None:
            try:
                import numpy as np

                query_vector = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
                scores = (np.asarray(self._embeddings) @ np.asarray(query_vector).T).ravel()
                order = scores.argsort()[::-1][:k]
                return [(self.documents[i], float(scores[i])) for i in order]
            except Exception as exc:  # pragma: no cover
                logger.debug("Embedding search failed (%s); falling back to keywords.", exc)
        if self._matrix is None or self._vectorizer is None:
            return self._keyword_search(query, k)
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        results = [(self.documents[i], float(scores[i])) for i in order if scores[i] > 0]
        return results or self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int) -> List[Tuple[KnowledgeDocument, float]]:
        terms = {token for token in str(query).lower().split() if len(token) > 2}
        scored: List[Tuple[KnowledgeDocument, float]] = []
        for document in self.documents:
            text = document.text.lower()
            hits = sum(text.count(term) for term in terms)
            if hits:
                scored.append((document, float(hits)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def context_for(self, question: str, k: int = 4, max_chars: int = 4000) -> str:
        """Render retrieved knowledge as a prompt-ready block ('' when empty)."""
        hits = self.search(question, k=k)
        if not hits:
            return ""
        parts: List[str] = []
        used = 0
        for document, score in hits:
            snippet = document.text.strip()
            if used + len(snippet) > max_chars:
                snippet = snippet[: max(max_chars - used, 200)]
            if not snippet:
                continue
            parts.append(f"[{document.kind}:{document.source} | relevance {score:.3f}]\n{snippet}")
            used += len(snippet)
            if used >= max_chars:
                break
        return "\n\n".join(parts)

    # -------------------------------------------------------------- persistence
    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"backend": self._backend, "documents": [document.to_dict() for document in self.documents]}
        target.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str, *, build_index: bool = True) -> "KnowledgeBase":
        target = Path(path)
        if not target.exists():
            return cls()
        payload = json.loads(target.read_text(encoding="utf-8"))
        kb = cls(
            KnowledgeDocument(
                doc_id=item.get("doc_id", ""),
                text=item.get("text", ""),
                source=item.get("source", ""),
                kind=item.get("kind", "document"),
                metadata=item.get("metadata") or {},
            )
            for item in payload.get("documents", [])
        )
        if build_index:
            kb.build_index()
        return kb

    def summary(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for document in self.documents:
            kinds[document.kind] = kinds.get(document.kind, 0) + 1
        return {"backend": self._backend, "documents": len(self.documents), "by_kind": kinds,
                "sources": sorted({document.source for document in self.documents})[:20]}


def search_knowledge(question: str, *, run_id: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
    """Convenience API used by the chat endpoint."""
    kb = KnowledgeBase()
    settings = get_settings()
    kb.add_directory(settings.knowledge_dir)
    if run_id and RunStore.exists(run_id):
        run_kb = KnowledgeBase.from_run(RunStore.load(run_id), include_domain_docs=False)
        kb.documents.extend(run_kb.documents)
    kb.build_index()
    return [{"text": document.text, "source": document.source, "kind": document.kind, "score": score}
            for document, score in kb.search(question, k=k)]


__all__ = ["KnowledgeBase", "KnowledgeDocument", "chunk_text", "search_knowledge"]
