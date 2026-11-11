"""NSMC 리뷰를 TF-IDF 코사인 유사도로 검색한다."""

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from konlpy.tag import Okt

from numpy_tfidf import NumpyTfidfVectorizer
from preprocess_nsmc import DATA_PATH, preprocess_text


DEFAULT_CORPUS_SIZE = 2_000
RANDOM_STATE = 42


@dataclass(frozen=True)
class SearchDocument:
    doc_id: str
    text: str
    tokens: tuple[str, ...]


def load_documents(path: Path, corpus_size: int, tagger: Okt) -> list[SearchDocument]:
    """원문이 같은 리뷰는 하나만 남기고 고정 시드로 검색 문서를 고른다."""
    unique_texts: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != ["id", "document", "label"]:
            raise ValueError(f"예상한 열과 다릅니다: {reader.fieldnames}")
        for row in reader:
            text = row["document"] or ""
            if text.strip():
                unique_texts.setdefault(text, row["id"])

    candidates = list(unique_texts.items())
    random.Random(RANDOM_STATE).shuffle(candidates)
    documents = []
    for text, doc_id in candidates:
        _, tokens, _ = preprocess_text(text, tagger)
        if tokens:
            documents.append(SearchDocument(doc_id, text, tokens))
            if len(documents) == corpus_size:
                break

    if len(documents) < corpus_size:
        raise ValueError(f"토큰이 있는 리뷰가 {corpus_size}개보다 적습니다.")
    return documents


class ReviewSearch:
    def __init__(self, documents: list[SearchDocument], tagger: Okt) -> None:
        self.documents = documents
        self.tagger = tagger
        self.vectorizer = NumpyTfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(
            [document.tokens for document in documents]
        )
        self.document_norms = np.linalg.norm(self.document_vectors, axis=1)

    def search(self, query: str, topk: int = 5) -> tuple[tuple[str, ...], list[tuple[float, str, str]]]:
        """쿼리와 문서 벡터의 코사인 유사도가 높은 순서로 반환한다."""
        if topk < 1:
            raise ValueError("topk는 1 이상이어야 합니다.")
        _, query_tokens, _ = preprocess_text(query, self.tagger)
        if not query_tokens:
            return query_tokens, []

        query_vector = self.vectorizer.transform([query_tokens])[0]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return query_tokens, []

        # cosine(q, d) = (d · q) / (||d|| × ||q||)
        numerators = self.document_vectors @ query_vector
        denominators = self.document_norms * query_norm
        scores = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators),
            where=denominators > 0,
        )

        results = []
        for index in np.argsort(-scores, kind="stable"):
            if scores[index] <= 0 or len(results) == topk:
                break
            document = self.documents[index]
            snippet = re.sub(r"\s+", " ", document.text).strip()
            if len(snippet) > 100:
                snippet = snippet[:100] + "…"
            results.append((float(scores[index]), document.doc_id, snippet))
        return query_tokens, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="찾고 싶은 리뷰 내용")
    parser.add_argument("--topk", type=int, default=5, help="반환할 최대 문서 수")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("topk는 1 이상이어야 합니다.")
    if not DATA_PATH.exists():
        parser.error("NSMC 데이터가 없습니다. explore_nsmc.py --download를 먼저 실행하세요.")

    tagger = Okt()
    documents = load_documents(DATA_PATH, DEFAULT_CORPUS_SIZE, tagger)
    engine = ReviewSearch(documents, tagger)
    tokens, results = engine.search(args.query, args.topk)

    print(f"검색 문서: {len(documents):,}개 / 어휘: {len(engine.vectorizer.vocabulary_):,}개")
    print(f"쿼리: {args.query}")
    print(f"쿼리 토큰: {list(tokens)}")
    if not results:
        print("검색어와 겹치는 토큰이 있는 문서가 없습니다.")
    for rank, (score, doc_id, snippet) in enumerate(results, start=1):
        print(f"{rank}. [{score:.6f}] id={doc_id} {snippet}")


if __name__ == "__main__":
    main()
