"""NumPy만 사용한 TF-IDF 벡터화 구현."""

import numpy as np


class NumpyTfidfVectorizer:
    """원시 빈도, smoothed IDF, L2 정규화를 직접 계산한다."""

    def __init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _count_terms(self, documents: list[tuple[str, ...]]) -> np.ndarray:
        counts = np.zeros((len(documents), len(self.vocabulary_)), dtype=np.float64)
        for row, tokens in enumerate(documents):
            columns = [self.vocabulary_[token] for token in tokens if token in self.vocabulary_]
            np.add.at(counts[row], columns, 1.0)
        return counts

    def fit(self, documents: list[tuple[str, ...]]) -> "NumpyTfidfVectorizer":
        if not documents:
            raise ValueError("TF-IDF를 학습할 문서가 없습니다.")
        terms = sorted({token for tokens in documents for token in tokens})
        if not terms:
            raise ValueError("어휘에 포함할 토큰이 없습니다.")

        self.vocabulary_ = {term: index for index, term in enumerate(terms)}
        counts = self._count_terms(documents)
        document_frequency = np.count_nonzero(counts, axis=0)
        self.idf_ = np.log((1 + len(documents)) / (1 + document_frequency)) + 1
        return self

    def transform(self, documents: list[tuple[str, ...]]) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("fit()으로 어휘와 IDF를 먼저 계산하세요.")
        term_frequency = self._count_terms(documents)
        tfidf = term_frequency * self.idf_
        lengths = np.linalg.norm(tfidf, axis=1, keepdims=True)
        np.divide(tfidf, lengths, out=tfidf, where=lengths != 0)
        return tfidf

    def fit_transform(self, documents: list[tuple[str, ...]]) -> np.ndarray:
        return self.fit(documents).transform(documents)
