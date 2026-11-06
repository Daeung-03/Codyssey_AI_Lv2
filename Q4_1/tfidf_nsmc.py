"""NumPy로 TF-IDF를 계산하고 scikit-learn 결과와 비교한다."""

import hashlib
import random
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess_nsmc import DATA_PATH, prepare_reviews


LOG_PATH = Path(__file__).parent / "docs" / "tfidf_validation.log"
SAMPLE_PER_LABEL = 500
RANDOM_STATE = 42
TOLERANCE = 1e-6


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


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("NSMC 데이터가 없습니다. explore_nsmc.py --download를 먼저 실행하세요.")

    reviews, stats = prepare_reviews(DATA_PATH)
    rng = random.Random(RANDOM_STATE)
    negative = [review for review in reviews if review.label == 0]
    positive = [review for review in reviews if review.label == 1]
    if min(len(negative), len(positive)) < SAMPLE_PER_LABEL:
        raise SystemExit("레이블별 검증 표본 500건을 뽑을 수 없습니다.")

    sample = rng.sample(negative, SAMPLE_PER_LABEL) + rng.sample(
        positive, SAMPLE_PER_LABEL
    )
    rng.shuffle(sample)
    documents = [review.tokens for review in sample]

    custom = NumpyTfidfVectorizer()
    custom_matrix = custom.fit_transform(documents)

    # 전처리된 형태소를 그대로 사용해 양쪽 구현의 토큰·어휘 순서를 맞춘다.
    reference = TfidfVectorizer(
        analyzer=str.split,
        vocabulary=custom.vocabulary_,
        lowercase=False,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        norm="l2",
        dtype=np.float64,
    )
    reference_matrix = reference.fit_transform(
        [" ".join(tokens) for tokens in documents]
    ).toarray()

    matrix_error = np.abs(custom_matrix - reference_matrix)
    max_error = float(np.max(matrix_error))
    mean_error = float(np.mean(matrix_error))
    idf_error = float(np.max(np.abs(custom.idf_ - reference.idf_)))
    passed = max_error <= TOLERANCE and idf_error <= TOLERANCE

    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    log = "\n".join(
        [
            "=== NSMC TF-IDF 구현 검증 ===",
            f"데이터: {DATA_PATH.name} (SHA-256: {data_hash})",
            f"정제 후 리뷰: {stats['prepared']:,}개",
            f"검증 표본: 부정 {SAMPLE_PER_LABEL}개 + 긍정 {SAMPLE_PER_LABEL}개, random_state={RANDOM_STATE}",
            "토큰화: KoNLPy Okt.pos(norm=True, stem=True), 조사 불용어 제거",
            "TF: 문서 안의 원시 출현 횟수",
            "IDF: log((1 + N) / (1 + df)) + 1",
            "정규화: 각 문서 벡터의 L2 길이로 나눔",
            "scikit-learn 설정: analyzer=str.split, vocabulary=직접 구현 어휘, "
            "lowercase=False, use_idf=True, smooth_idf=True, "
            "sublinear_tf=False, norm='l2', dtype=float64",
            f"버전: NumPy {np.__version__}, scikit-learn {sklearn.__version__}",
            f"TF-IDF 행렬 shape: {custom_matrix.shape}",
            f"IDF 최대 오차: {idf_error:.12e}",
            f"행렬 최대 오차: {max_error:.12e}",
            f"행렬 평균 오차: {mean_error:.12e}",
            f"결과: {'PASS' if passed else 'FAIL'} (허용 오차 {TOLERANCE:.0e})",
        ]
    )
    LOG_PATH.write_text(log + "\n", encoding="utf-8")
    print(log)
    print(f"검증 로그 저장: {LOG_PATH}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
