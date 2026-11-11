"""NSMC 리뷰를 직접 구현한 TF-IDF와 Logistic Regression으로 분류한다."""

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from konlpy.tag import Okt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from numpy_tfidf import NumpyTfidfVectorizer
from preprocess_nsmc import DATA_PATH, PreparedReview, preprocess_text


RANDOM_STATE = 42
SAMPLE_PER_LABEL = 1_000
REPORT_DIR = Path(__file__).parent / "docs"
LOG_PATH = REPORT_DIR / "classification_validation.log"
FIGURE_PATH = REPORT_DIR / "confusion_matrix.svg"


def load_sample(path: Path) -> tuple[list[PreparedReview], int]:
    """원문 중복과 충돌 레이블을 정리한 뒤 레이블별 리뷰를 추출한다."""
    first_row_by_text: dict[str, tuple[str, int]] = {}
    conflicting_texts: set[str] = set()
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != ["id", "document", "label"]:
            raise ValueError(f"예상한 열과 다릅니다: {reader.fieldnames}")
        for row in reader:
            text = row["document"] or ""
            if not text.strip():
                continue
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"예상하지 못한 레이블: {label}")
            if text in first_row_by_text:
                if first_row_by_text[text][1] != label:
                    conflicting_texts.add(text)
            else:
                first_row_by_text[text] = (row["id"], label)

    candidates: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
    for text, (doc_id, label) in first_row_by_text.items():
        if text not in conflicting_texts:
            candidates[label].append((doc_id, text))

    rng = random.Random(RANDOM_STATE)
    tagger = Okt()
    sample: list[PreparedReview] = []
    for label in (0, 1):
        rng.shuffle(candidates[label])
        selected = 0
        for doc_id, text in candidates[label]:
            cleaned, tokens, _ = preprocess_text(text, tagger)
            if not tokens:
                continue
            sample.append(PreparedReview(doc_id, text, cleaned, label, tokens))
            selected += 1
            if selected == SAMPLE_PER_LABEL:
                break
        if selected < SAMPLE_PER_LABEL:
            raise ValueError(f"레이블 {label}의 유효한 리뷰가 {SAMPLE_PER_LABEL}개보다 적습니다.")

    rng.shuffle(sample)
    return sample, len(conflicting_texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="NSMC TSV 파일 경로")
    args = parser.parse_args()
    if not args.data.exists():
        parser.error("NSMC 데이터가 없습니다. explore_nsmc.py --download를 먼저 실행하세요.")

    sample, conflict_count = load_sample(args.data)
    train, test = train_test_split(
        sample,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=[review.label for review in sample],
    )

    vectorizer = NumpyTfidfVectorizer()
    vectorizer.fit([review.tokens for review in train])
    train_vectors = vectorizer.transform([review.tokens for review in train])
    test_vectors = vectorizer.transform([review.tokens for review in test])

    model = LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)
    model.fit(train_vectors, [review.label for review in train])
    actual = [review.label for review in test]
    predicted = model.predict(test_vectors)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    display = ConfusionMatrixDisplay(matrix, display_labels=["negative (0)", "positive (1)"])
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    display.ax_.set_title("NSMC test set confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(FIGURE_PATH)
    plt.close(display.figure_)

    errors = [
        (index, review, int(guess))
        for index, (review, guess) in enumerate(zip(test, predicted))
        if review.label != guess
    ]
    vocabulary = vectorizer.vocabulary_
    negation_tokens = {"안", "못", "아니다", "않다", "없다"}
    indicators = {
        "학습 어휘 밖 토큰 포함": lambda review: bool(set(review.tokens) - vocabulary.keys()),
        "토큰 3개 이하": lambda review: len(review.tokens) <= 3,
        "부정 표현 토큰 포함": lambda review: bool(set(review.tokens) & negation_tokens),
    }
    error_reviews = [review for _, review, _ in errors]
    correct_reviews = [
        review for review, guess in zip(test, predicted) if review.label == guess
    ]
    log_lines = [
        "=== NSMC TF-IDF 감성 분류 실험 ===",
        f"데이터: {args.data.name}",
        f"원문 레이블 충돌 제외: {conflict_count}개 그룹; 동일 원문·레이블 중복은 첫 행만 유지",
        f"표본: 부정 {SAMPLE_PER_LABEL}개 + 긍정 {SAMPLE_PER_LABEL}개",
        f"분할: Train {len(train)}개 / Test {len(test)}개, stratify=label, random_state={RANDOM_STATE}",
        "전처리: Okt.pos(norm=True, stem=True), 선택한 조사 불용어 제거",
        "TF-IDF: 직접 구현; 어휘·IDF는 Train에서만 fit, Test는 transform만 수행",
        f"Train 어휘 수: {len(vectorizer.vocabulary_)}개",
        "모델: LogisticRegression(max_iter=1000, random_state=42)",
        f"정확도: {accuracy_score(actual, predicted):.6f}",
        f"Precision (macro): {precision_score(actual, predicted, average='macro'):.6f}",
        f"Recall (macro): {recall_score(actual, predicted, average='macro'):.6f}",
        f"F1-Score (macro): {f1_score(actual, predicted, average='macro'):.6f}",
        "혼동 행렬 (행=실제, 열=예측; 순서=0,1):",
        f"  {matrix[0].tolist()}",
        f"  {matrix[1].tolist()}",
        f"오분류: {len(errors)}개",
        "오류 관련 신호 (서로 중복 가능; 원인으로 확정하지 않음):",
    ]
    for name, condition in indicators.items():
        error_count = sum(condition(review) for review in error_reviews)
        correct_count = sum(condition(review) for review in correct_reviews)
        error_rate = error_count / len(error_reviews) if error_reviews else 0.0
        correct_rate = correct_count / len(correct_reviews) if correct_reviews else 0.0
        log_lines.append(
            f"  {name}: 오분류 {error_count}/{len(error_reviews)} "
            f"({error_rate:.1%}), "
            f"정분류 {correct_count}/{len(correct_reviews)} "
            f"({correct_rate:.1%})"
        )
    log_lines.append("오분류 예시 5개:")
    terms_by_index = {index: term for term, index in vectorizer.vocabulary_.items()}
    for number, (index, review, guess) in enumerate(errors[:5], start=1):
        contributions = test_vectors[index] * model.coef_[0]
        strongest = np.argsort(-np.abs(contributions))[:4]
        evidence = [
            f"{terms_by_index[column]}:{contributions[column]:+.3f}"
            for column in strongest
            if contributions[column] != 0
        ]
        unseen = sorted(set(review.tokens).difference(vectorizer.vocabulary_))
        log_lines.append(
            f"  {number}. id={review.doc_id} 실제={review.label} 예측={guess} "
            f"원문={review.text!r} 토큰={list(review.tokens)!r} "
            f"기여(+=긍정/-=부정)={evidence!r} 미등록={unseen!r}"
        )
    log_lines.append(f"혼동 행렬 그림: {FIGURE_PATH.name}")
    log = "\n".join(log_lines) + "\n"
    LOG_PATH.write_text(log, encoding="utf-8")
    print(log, end="")


if __name__ == "__main__":
    main()
