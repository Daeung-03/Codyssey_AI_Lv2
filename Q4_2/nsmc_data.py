"""NSMC(네이버 영화 리뷰) 다운로드·정제·표본 추출.

데이터 사용 구분 (누수 방지)
- 후보 풀:  train에서 뽑은 20,000개. 감성 사전 후보 단어 통계에만 쓴다.
- 개발셋:   train에서 후보 풀과 겹치지 않게 뽑은 1,000개. 규칙·사전 조정과 0점 처리 기준에 쓴다.
- 평가셋:   test에서 뽑은 1,000개. 최종 수치에만 쓰고 조정에는 쓰지 않는다.
모두 긍정·부정 같은 수로 뽑는다.
"""

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


DATA_DIR = Path(__file__).parent / "data"
TRAIN_PATH = DATA_DIR / "ratings_train.txt"
TEST_PATH = DATA_DIR / "ratings_test.txt"
BASE_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/"
RANDOM_STATE = 42

CANDIDATE_POOL_SIZE = 20_000
DEV_SIZE = 1_000
EVAL_SIZE = 1_000


@dataclass(frozen=True)
class Review:
    doc_id: str
    text: str
    label: int  # 0 부정, 1 긍정


def download(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_PATH, TEST_PATH):
        if force or not path.exists():
            print(f"다운로드: {BASE_URL + path.name}")
            urlretrieve(BASE_URL + path.name, path)


def load_reviews(path: Path) -> list[Review]:
    """빈 리뷰를 빼고, 같은 원문은 하나만 남긴다. 원문이 같은데 레이블이 다르면 모두 뺀다 (Q4_1과 같은 기준)."""
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. python nsmc_data.py --download 로 받아 주세요.")
    first: dict[str, Review] = {}
    labels: dict[str, set[int]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ["id", "document", "label"]:
            raise ValueError(f"예상한 열과 다릅니다: {reader.fieldnames}")
        for row in reader:
            text = (row["document"] or "").strip()
            if not text:
                continue
            label = int(row["label"])
            first.setdefault(text, Review(row["id"], text, label))
            labels.setdefault(text, set()).add(label)
    return [review for text, review in first.items() if len(labels[text]) == 1]


def sample_balanced(reviews: list[Review], size: int, seed: int = RANDOM_STATE) -> list[Review]:
    """긍정·부정을 size/2개씩 고정 시드로 뽑는다."""
    rng = random.Random(seed)
    sample = []
    for label in (0, 1):
        pool = [review for review in reviews if review.label == label]
        sample += rng.sample(pool, size // 2)
    rng.shuffle(sample)
    return sample


def split_train() -> tuple[list[Review], list[Review]]:
    """train을 (사전 후보 풀, 개발셋)으로 겹치지 않게 나눈다."""
    reviews = load_reviews(TRAIN_PATH)
    pool = sample_balanced(reviews, CANDIDATE_POOL_SIZE)
    used = {review.text for review in pool}
    dev = sample_balanced([review for review in reviews if review.text not in used], DEV_SIZE)
    return pool, dev


def eval_set() -> list[Review]:
    """test에서 train과 원문이 겹치는 리뷰를 뺀 뒤 평가셋을 뽑는다."""
    train_texts = {review.text for review in load_reviews(TRAIN_PATH)}
    test = [review for review in load_reviews(TEST_PATH) if review.text not in train_texts]
    return sample_balanced(test, EVAL_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="NSMC를 받고 표본 구성을 확인한다.")
    parser.add_argument("--download", action="store_true", help="원본이 없으면 다운로드")
    args = parser.parse_args()
    if args.download:
        download()

    pool, dev = split_train()
    evaluation = eval_set()
    for name, reviews in (("사전 후보 풀", pool), ("개발셋", dev), ("평가셋", evaluation)):
        labels = Counter(review.label for review in reviews)
        print(f"{name}: {len(reviews)}개 (부정 {labels[0]}, 긍정 {labels[1]})")


if __name__ == "__main__":
    main()
