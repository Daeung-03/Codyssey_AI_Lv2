"""감성 사전과 부정어·강조어 목록.

사전 파일(TSV): word, score(-2~+2), pos, note
- data/sentiment_lexicon.tsv: 범용 감성어
- data/domain_lexicon.tsv:    영화 리뷰 도메인어 (꿀잼, 노잼, 발연기 …)

표제어는 Okt `pos(norm=True, stem=True)`가 내놓는 형태로 적는다 (예: 맛있다, 지루하다, 감동).
그래야 토큰과 그대로 대조할 수 있다. `python lexicon.py --check`로 확인한다.

후보 단어는 train 후보 풀에서만 뽑는다: `python lexicon.py --candidates`
"""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from konlpy.tag import Okt


DATA_DIR = Path(__file__).parent / "data"
GENERAL_LEXICON_PATH = DATA_DIR / "sentiment_lexicon.tsv"
DOMAIN_LEXICON_PATH = DATA_DIR / "domain_lexicon.tsv"
LEXICON_PATHS = (GENERAL_LEXICON_PATH, DOMAIN_LEXICON_PATH)
CANDIDATES_PATH = Path(__file__).parent / "docs" / "lexicon_candidates.tsv"

# 부정어 (Okt 원형 기준)
# 앞 부정어: 감성어 앞에서 부정한다. '안 좋다', '못 웃기다'
PRE_NEGATORS = frozenset({"안", "못"})
# 뒤 부정어: '-지 않다', '-지 못하다', '재미가 없다', '좋은 영화는 아니다'처럼 감성어 뒤에서 부정한다.
POST_NEGATORS = frozenset({"않다", "없다", "아니다", "못하다"})
# 이중부정에서 두 부정어 사이에 올 수 있는 말: '-지 않을 수 없다', '-지 않은 것은 아니다'
DOUBLE_NEGATION_FILLERS = frozenset({"수", "것", "건", "게", "리"})
# 앞 부정어가 '명사 + 안/못 + 하다/되다'로 쓰이면 앞 명사를 부정한다: '공감 못 하다', '감동 안 되다'
LIGHT_VERBS = frozenset({"하다", "되다"})
# 부정 결합어: 늘 부정어와 함께 쓰이고 사전 점수가 이미 그 결합의 뜻이라 반전하지 않는다.
# '별로 없다', '그닥 재밌지 않다', '어이 없다', '도저히 못 보다', '개연성 없다'
NEGATION_IMMUNE = frozenset({"별로", "그닥", "어이", "도저히", "개연"})

# 강조어: 감성어 앞에 오면 점수에 배율을 곱한다.
INTENSIFIERS = {
    "정말": 1.5, "진짜": 1.5, "너무": 1.5, "매우": 1.5, "완전": 1.5, "엄청": 1.5,
    "아주": 1.5, "무척": 1.5, "되게": 1.5, "굉장히": 1.5, "참": 1.5, "넘": 1.5,
    "정말로": 1.5, "진심": 1.5, "제일": 1.5, "가장": 1.5, "최고로": 1.5, "존나": 1.5,
    "조금": 0.5, "좀": 0.5, "약간": 0.5, "다소": 0.5, "살짝": 0.5,
}

# 후보 통계에서 제외할 품사
SKIP_POS = frozenset({"Josa", "Punctuation", "Eomi", "PreEomi", "Number", "Foreign", "Alpha", "URL"})

MIN_LEXICON_SIZE = 200
MIN_DOMAIN_SIZE = 30


def load_lexicon(paths=LEXICON_PATHS) -> dict[str, int]:
    """사전 파일들을 합친다. 같은 단어가 다른 점수로 두 번 나오면 오류."""
    lexicon: dict[str, int] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                word, score = row["word"].strip(), int(row["score"])
                if not -2 <= score <= 2 or score == 0:
                    raise ValueError(f"{path.name}: 점수는 -2~+2(0 제외)여야 합니다: {word} {score}")
                if word in lexicon and lexicon[word] != score:
                    raise ValueError(f"{path.name}: {word} 점수 충돌 {lexicon[word]} vs {score}")
                lexicon[word] = score
    return lexicon


def check_entries(tagger: Okt, paths=LEXICON_PATHS) -> list[str]:
    """표제어가 Okt 토큰으로 나오는지 확인한다.

    단어만 따로 분석하면 다르게 나오지만 문장 속에서는 원형으로 나오는 단어(웃기다 등)는
    note에 '예문: …'을 적어 두고 그 예문으로 확인한다.
    """
    problems = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                word, note = row["word"].strip(), row["note"] or ""
                sample = note.split("예문: ", 1)[1] if "예문: " in note else word
                tokens = [token for token, _ in tagger.pos(sample, norm=True, stem=True)]
                if word not in tokens:
                    problems.append(f"{path.name}: {word} → {tokens}")
    return problems


def build_candidates(tagger: Okt, min_df: int = 30) -> list[tuple[str, str, int, int, float]]:
    """train 후보 풀에서 (토큰, 품사, 부정 문서 수, 긍정 문서 수, 로그 오즈)를 구한다.

    로그 오즈 = log((긍정 문서 수 + 1) / (부정 문서 수 + 1)). 양수면 긍정 리뷰에, 음수면 부정 리뷰에 치우친 단어다.
    """
    from nsmc_data import split_train

    pool, _ = split_train()
    doc_freq: dict[tuple[str, str], Counter] = {}
    for review in pool:
        seen = {(t, p) for t, p in tagger.pos(review.text.lower(), norm=True, stem=True) if p not in SKIP_POS}
        for key in seen:
            doc_freq.setdefault(key, Counter())[review.label] += 1

    rows = []
    for (token, pos), counts in doc_freq.items():
        if counts[0] + counts[1] >= min_df:
            rows.append((token, pos, counts[0], counts[1], math.log((counts[1] + 1) / (counts[0] + 1))))
    return sorted(rows, key=lambda row: row[4])


def main() -> None:
    parser = argparse.ArgumentParser(description="감성 사전 확인·후보 추출")
    parser.add_argument("--check", action="store_true", help="사전 표제어가 Okt 토큰과 맞는지 확인")
    parser.add_argument("--candidates", action="store_true", help="train 후보 풀에서 후보 단어 통계 저장")
    args = parser.parse_args()
    tagger = Okt()

    if args.candidates:
        rows = build_candidates(tagger)
        CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CANDIDATES_PATH.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["token", "pos", "neg_df", "pos_df", "log_odds"])
            writer.writerows((t, p, n, q, f"{lo:.3f}") for t, p, n, q, lo in rows)
        print(f"후보 {len(rows)}개 저장: {CANDIDATES_PATH}")

    if args.check or not args.candidates:
        lexicon = load_lexicon()
        domain = load_lexicon((DOMAIN_LEXICON_PATH,))
        scores = Counter(lexicon.values())
        print(f"사전 {len(lexicon)}개 (도메인 {len(domain)}개), 점수 분포: {dict(sorted(scores.items()))}")
        if len(lexicon) < MIN_LEXICON_SIZE or len(domain) < MIN_DOMAIN_SIZE:
            print(f"경고: 사전 {MIN_LEXICON_SIZE}개·도메인 {MIN_DOMAIN_SIZE}개 이상이 필요합니다.")
        problems = check_entries(tagger)
        print("Okt 토큰 불일치: " + ("없음" if not problems else f"{len(problems)}개"))
        for problem in problems:
            print("  " + problem)


if __name__ == "__main__":
    main()
