"""NSMC 리뷰 정제, Okt 형태소 분석, 불용어 제거를 살펴보는 실습."""

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from konlpy.tag import Okt


DATA_PATH = Path(__file__).parent / "data" / "ratings_train.txt"

# stopwords-iso 한국어 목록에서 조사 일부만 선택했다.
# 같은 글자라도 명사·동사 등으로 분석되면 제거하지 않는다.
JOSA_STOPWORDS = frozenset(
    {"은", "는", "이", "가", "을", "를", "의", "에", "에서", "에게", "와", "과"}
)


@dataclass(frozen=True)
class PreparedReview:
    doc_id: str
    text: str
    cleaned: str
    label: int
    tokens: tuple[str, ...]


def clean_text(text: str) -> str:
    """영문을 소문자로 만들고, 구두점 등을 공백으로 바꾼다."""
    lowered = text.lower()
    kept = re.sub(r"[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", lowered)
    return " ".join(kept.split())


def tokenize(cleaned_text: str, tagger: Okt) -> list[tuple[str, str]]:
    """Okt로 형태소와 품사를 얻고 동사·형용사를 기본형으로 만든다."""
    return tagger.pos(cleaned_text, norm=True, stem=True)


def remove_stopwords(tagged: list[tuple[str, str]]) -> list[str]:
    return [
        token
        for token, pos in tagged
        if not (pos == "Josa" and token in JOSA_STOPWORDS)
    ]


def preprocess_text(text: str, tagger: Okt) -> tuple[str, tuple[str, ...], int]:
    """검색어에도 똑같이 적용할 전처리 함수."""
    cleaned = clean_text(text)
    tagged = tokenize(cleaned, tagger)
    tokens = tuple(remove_stopwords(tagged))
    return cleaned, tokens, len(tagged) - len(tokens)


def prepare_reviews(path: Path) -> tuple[list[PreparedReview], Counter[str]]:
    """빈 리뷰와 중복을 정리한 뒤 각 리뷰를 토큰으로 변환한다."""
    stats: Counter[str] = Counter()
    label_counts_by_text: dict[str, Counter[int]] = {}
    first_id_by_text: dict[str, str] = {}

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != ["id", "document", "label"]:
            raise ValueError(f"예상한 열과 다릅니다: {reader.fieldnames}")

        for row in reader:
            stats["original"] += 1
            text = row["document"] or ""
            if not text.strip():
                stats["empty"] += 1
                continue

            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"예상하지 못한 레이블: {label}")
            if text not in label_counts_by_text:
                label_counts_by_text[text] = Counter()
                first_id_by_text[text] = row["id"]

            counts = label_counts_by_text[text]
            counts[label] += 1

    tagger = Okt()
    prepared = []
    for text, counts in label_counts_by_text.items():
        if len(counts) > 1:
            stats["conflicting_texts"] += 1
            stats["conflicting_rows"] += sum(counts.values())
            continue

        stats["same_label_duplicates_removed"] += sum(counts.values()) - 1
        cleaned, tokens, removed_count = preprocess_text(text, tagger)
        stats["removed_stopword_tokens"] += removed_count
        if not tokens:
            stats["empty_after_preprocessing"] += 1
            continue

        label = next(iter(counts))
        prepared.append(
            PreparedReview(first_id_by_text[text], text, cleaned, label, tokens)
        )
        stats[f"label_{label}"] += 1

    stats["prepared"] = len(prepared)
    return prepared, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="NSMC TSV 파일 경로")
    parser.add_argument("--samples", type=int, default=3, help="표시할 변환 예시 수")
    parser.add_argument("--text", help="한 문장만 형태소 분석하여 확인")
    args = parser.parse_args()
    if args.text is not None:
        tagger = Okt()
        cleaned, tokens, removed_count = preprocess_text(args.text, tagger)
        print(f"정제: {cleaned}")
        print(f"형태소·품사: {tokenize(cleaned, tagger)}")
        print(f"불용어 제거 후: {list(tokens)} ({removed_count}개 제거)")
        return

    if not args.data.exists():
        parser.error("데이터가 없습니다. 먼저 explore_nsmc.py --download를 실행하세요.")

    reviews, stats = prepare_reviews(args.data)
    print(f"원본 리뷰: {stats['original']:,}")
    print(f"빈 리뷰 제외: {stats['empty']:,}")
    print(f"동일 텍스트·레이블 중복 제외: {stats['same_label_duplicates_removed']:,}")
    print(
        "레이블 충돌 제외: "
        f"{stats['conflicting_texts']:,}개 텍스트 / {stats['conflicting_rows']:,}개 행"
    )
    print(f"전처리 후 토큰이 없는 리뷰 제외: {stats['empty_after_preprocessing']:,}")
    print(f"제거된 불용어 토큰: {stats['removed_stopword_tokens']:,}")
    print(f"최종 리뷰: {stats['prepared']:,}")
    print(f"레이블 분포: 0={stats['label_0']:,}, 1={stats['label_1']:,}")

    print("\n변환 예시:")
    for review in reviews[: max(args.samples, 0)]:
        print(f"[{review.label}] {review.text}")
        print(f"    정제: {review.cleaned}")
        print(f"    토큰: {list(review.tokens)}")


if __name__ == "__main__":
    main()
