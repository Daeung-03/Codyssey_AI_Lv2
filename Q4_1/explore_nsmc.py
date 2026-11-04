"""NSMC 학습 데이터의 구조와 레이블을 살펴보는 첫 실습."""

from pathlib import Path
from urllib.request import urlretrieve
import argparse
import csv
from collections import Counter


DATA_PATH = Path(__file__).parent / "data" / "ratings_train.txt"
DATA_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="원본 데이터가 없으면 다운로드")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        if not args.download:
            parser.error("데이터가 없습니다. --download 옵션으로 원본을 받아 주세요.")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"NSMC 다운로드: {DATA_URL}")
        urlretrieve(DATA_URL, DATA_PATH)

    total = empty = same_label_duplicates = conflicting_label_reviews = 0
    labels = Counter()
    seen_text_labels: dict[str, set[int]] = {}
    conflicting_texts = set()
    examples = {0: [], 1: []}

    with DATA_PATH.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != ["id", "document", "label"]:
            raise ValueError(f"예상한 열과 다릅니다: {reader.fieldnames}")
        for row in reader:
            total += 1
            label = int(row["label"])
            labels[label] += 1
            review = row["document"] or ""
            if not review.strip():
                empty += 1
                continue
            previous_labels = seen_text_labels.setdefault(review, set())
            if label in previous_labels:
                same_label_duplicates += 1
            elif previous_labels:
                conflicting_label_reviews += 1
                conflicting_texts.add(review)
            previous_labels.add(label)
            if label in examples and len(examples[label]) < 3:
                examples[label].append(review)

    print(f"전체 리뷰 수: {total:,}")
    print("열: ['id', 'document', 'label']")
    print(f"빈 리뷰 수: {empty:,}")
    print(f"텍스트·레이블이 모두 같은 중복: {same_label_duplicates:,}")
    print(f"텍스트는 같고 레이블은 다른 추가 리뷰: {conflicting_label_reviews:,}")
    print(f"서로 다른 레이블이 붙은 고유 텍스트: {len(conflicting_texts):,}")
    print("\n레이블 분포 (0=부정, 1=긍정):")
    for label in sorted(labels):
        print(f"  {label}: {labels[label]:,}")

    print("\n레이블별 리뷰 예시:")
    for label, name in [(0, "부정"), (1, "긍정")]:
        print(f"{name} ({label}):")
        for text in examples[label]:
            print(f"  - {text}")


if __name__ == "__main__":
    main()
