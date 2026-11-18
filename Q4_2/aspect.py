"""[보너스] 측면(Aspect)별 감성 분석.

"음식은 맛있는데 가격이 비싸요" → 음식(긍정), 가격(부정)

처리 순서
1. 절 나누기: 원문을 문장부호와 연결어미(-는데, -지만, -고, -으나 …) 뒤에서 자른다.
   Okt 원형화(stem)는 연결어미를 지워 버리므로 토큰화 전에 원문에서 자른다.
2. 측면 찾기: 절의 토큰 중 측면 키워드(맛·가격·직원·배송·배우·결말 …)를 찾는다.
   키워드가 없으면 서술어로 측면을 추정한다 (맛있다 → 음식, 비싸다 → 가격).
3. 구간 나누기: 측면 키워드에서 다음 키워드 직전까지를 그 측면의 구간으로 본다.
   첫 키워드 앞의 말('안 맛있다'의 '안' 등)은 첫 구간에 붙인다.
4. 점수: 구간마다 sentiment.SentimentAnalyzer로 점수를 낸다. 측면에 따라 극성이 다른 말
   (배송이 '빠르다'는 긍정, 가격이 '싸다'는 긍정)은 측면별 사전으로 덮어쓴다.
   키워드 자체는 감성어로 세지 않는다 (예: 영화 사전의 '알바'는 부정이지만 식당 리뷰의 직원 '알바'는 아님).
5. 같은 측면이 여러 번 나오면 점수를 더한다. 점수 0이면 판정하지 않는다.

평가: python aspect.py --evaluate (data/aspect_testset.jsonl, 패턴을 보지 않은 에이전트가 작성)
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from konlpy.tag import Okt

from lexicon import load_lexicon
from sentiment import LABEL_NAMES, SentimentAnalyzer


TESTSET_PATH = Path(__file__).parent / "data" / "aspect_testset.jsonl"
LOG_PATH = Path(__file__).parent / "docs" / "aspect_eval.log"

# 측면 키워드 (Okt 원형 기준)
ASPECT_KEYWORDS = {
    "음식": {"음식", "맛", "메뉴", "요리", "반찬", "국물", "고기", "양", "식감", "간", "디저트", "커피"},
    "가격": {"가격", "값", "가성", "가성비", "비용", "요금", "금액"},
    "서비스": {"서비스", "직원", "사장", "사장님", "알바", "점원", "종업원", "응대", "상담", "고객센터", "친절"},
    "배송": {"배송", "택배", "포장", "배달", "도착"},
    "분위기": {"분위기", "인테리어", "매장", "공간", "좌석", "자리", "조명"},
    "연기": {"연기", "연기력", "배우", "주연", "조연", "캐스팅"},
    "스토리": {"스토리", "내용", "줄거리", "결말", "전개", "각본", "시나리오", "반전", "플롯", "개연"},
    "영상": {"영상", "영상미", "상미", "화면", "cg", "그래픽", "촬영", "색감", "액션"},
    "음악": {"음악", "ost", "노래", "사운드", "배경음악", "브금"},
}
KEYWORD_TO_ASPECT = {word: aspect for aspect, words in ASPECT_KEYWORDS.items() for word in words}

# 키워드가 없는 절에서 측면을 알려 주는 서술어
IMPLICIT_ASPECTS = {
    "맛있다": "음식", "맛없다": "음식", "맛나다": "음식",
    "비싸다": "가격", "싸다": "가격", "저렴하다": "가격",
    "친절하다": "서비스", "불친절하다": "서비스",
    "발연기": "연기", "명연기": "연기", "열연": "연기", "호연": "연기",
}

# 측면에 따라 극성이 정해지는 말. 기본 사전 위에 덮어쓴다.
ASPECT_LEXICON = {
    "음식": {"맛나다": 2, "맛없다": -2, "짜다": -1, "싱겁다": -1, "느끼하다": -1, "질기다": -1, "담백하다": 1,
             "부드럽다": 1, "바삭하다": 1, "많다": 1, "푸짐하다": 1},
    "가격": {"싸다": 1, "저렴하다": 1, "합리적": 1, "착하다": 1, "비싸다": -1},
    "서비스": {"불친절하다": -2, "빠르다": 1, "빨르다": 1, "느리다": -1},
    "배송": {"빠르다": 1, "빨르다": 1, "신속하다": 1, "늦다": -1, "느리다": -1, "파손": -2, "꼼꼼하다": 1},
    "분위기": {"조용하다": 1, "넓다": 1, "좁다": -1, "시끄럽다": -1},
    "연기": {"미치다": 2},
    "스토리": {},
    "영상": {},
    "음악": {},
}

# 절을 나누는 위치: 문장부호, 또는 연결어미 + 공백
CLAUSE_BREAK = re.compile(r"[.!?,;~\n]+|(?:는데|은데|인데|던데|지만|으나|구요|고요|고|며|면서|어서|아서)(?=\s)")
ASPECT_TOKEN = ("<측면>", "Noun")  # 키워드 자리를 비워 위치(부정어·강조어 거리)는 유지한다


@dataclass
class AspectScore:
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def label(self) -> int | None:
        return None if self.score == 0 else int(self.score > 0)


def split_clauses(text: str) -> list[str]:
    clauses, start = [], 0
    for match in CLAUSE_BREAK.finditer(text):
        clauses.append(text[start:match.end()])
        start = match.end()
    clauses.append(text[start:])
    return [clause.strip() for clause in clauses if clause.strip()]


class AspectAnalyzer:
    def __init__(self, lexicon: dict[str, int] | None = None, tagger: Okt | None = None) -> None:
        base = lexicon if lexicon is not None else load_lexicon()
        self.tagger = tagger or Okt()
        self.analyzers = {
            aspect: SentimentAnalyzer({**base, **ASPECT_LEXICON[aspect]}, tagger=self.tagger)
            for aspect in ASPECT_KEYWORDS
        }

    def analyze(self, text: str) -> dict[str, AspectScore]:
        results: dict[str, AspectScore] = {}
        tokenize = next(iter(self.analyzers.values())).tokenize
        for clause in split_clauses(text):
            tokens = tokenize(clause)
            anchors = [(i, KEYWORD_TO_ASPECT[t]) for i, (t, _) in enumerate(tokens) if t in KEYWORD_TO_ASPECT]
            implicit = not anchors
            if implicit:
                anchors = [(i, IMPLICIT_ASPECTS[t]) for i, (t, _) in enumerate(tokens) if t in IMPLICIT_ASPECTS]
            for n, (start, aspect) in enumerate(anchors):
                begin = 0 if n == 0 else start
                end = anchors[n + 1][0] if n + 1 < len(anchors) else len(tokens)
                segment = tuple(
                    ASPECT_TOKEN if (not implicit and t in KEYWORD_TO_ASPECT) else (t, p)
                    for t, p in tokens[begin:end]
                )
                result = self.analyzers[aspect].score_tokens(clause, segment)
                if result.score:
                    entry = results.setdefault(aspect, AspectScore())
                    entry.score += result.score
                    entry.evidence += [f"{h.word}{'(반전)' if h.negator else ''}={h.score:+g}" for h in result.hits]
        return {aspect: entry for aspect, entry in results.items() if entry.label is not None}


def format_aspects(text: str, aspects: dict[str, AspectScore]) -> str:
    lines = ["=== 측면별 감성 ===", f'문장: "{text}"', f"절: {split_clauses(text)}"]
    if not aspects:
        lines.append("- (판정할 측면 없음)")
    for aspect, entry in aspects.items():
        lines.append(f"- {aspect}: {LABEL_NAMES[entry.label]} ({entry.score:+g}; {', '.join(entry.evidence)})")
    return "\n".join(lines)


def evaluate(analyzer: AspectAnalyzer, path: Path = TESTSET_PATH) -> str:
    """측면 탐지 P/R과, 측면을 맞힌 경우의 극성 정확도, (측면, 극성) 쌍 P/R을 낸다."""
    label_ids = {"긍정": 1, "부정": 0}
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter()
    by_domain: dict[str, Counter] = {}
    failures = []
    for case in cases:
        gold = {aspect: label_ids[label] for aspect, label in case["aspects"].items()}
        pred = {aspect: entry.label for aspect, entry in analyzer.analyze(case["text"]).items()}
        domain = by_domain.setdefault(case["domain"], Counter())
        found = set(gold) & set(pred)
        polarity_ok = sum(gold[a] == pred[a] for a in found)
        for counter in (counts, domain):
            counter["gold"] += len(gold)
            counter["pred"] += len(pred)
            counter["found"] += len(found)
            counter["pair_ok"] += polarity_ok
            counter["exact"] += gold == pred
            counter["n"] += 1
        if gold != pred:
            failures.append((case, gold, pred))

    def row(name: str, c: Counter) -> str:
        detect_p = c["found"] / c["pred"] if c["pred"] else 0
        detect_r = c["found"] / c["gold"]
        polarity = c["pair_ok"] / c["found"] if c["found"] else 0
        pair_p = c["pair_ok"] / c["pred"] if c["pred"] else 0
        pair_r = c["pair_ok"] / c["gold"]
        return (f"{name:<12}{c['n']:>4}{c['gold']:>5}{detect_p:>8.3f}{detect_r:>8.3f}"
                f"{polarity:>8.3f}{pair_p:>8.3f}{pair_r:>8.3f}{c['exact'] / c['n']:>8.3f}")

    names = {1: "긍정", 0: "부정"}
    lines = [
        f"=== Aspect 감성 평가 ({len(cases)}문장) ===",
        "탐지P/R: 측면을 찾았는지, 극성정확도: 찾은 측면 중 극성이 맞은 비율, 쌍P/R: (측면, 극성)이 모두 맞은 비율",
        f"{'분야':<12}{'문장':>4}{'정답':>5}{'탐지P':>8}{'탐지R':>8}{'극성정확':>8}{'쌍P':>8}{'쌍R':>8}{'문장일치':>8}",
    ]
    lines += [row(domain, c) for domain, c in sorted(by_domain.items())]
    lines.append(row("전체", counts))
    lines.append(f"\n--- 불일치 {len(failures)}문장 ---")
    for case, gold, pred in failures:
        lines.append(f"\n[{case['id']}] {case['text']}")
        lines.append(f"  절: {split_clauses(case['text'])}")
        lines.append(f"  정답: {', '.join(f'{a}({names[l]})' for a, l in gold.items())}")
        lines.append(f"  예측: {', '.join(f'{a}({names[l]})' for a, l in pred.items()) or '(없음)'}")
        if case.get("note"):
            lines.append(f"  메모: {case['note']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="측면별 감성 분석")
    parser.add_argument("--text", help="분석할 문장")
    parser.add_argument("--evaluate", action="store_true", help="aspect_testset.jsonl로 평가하고 로그 저장")
    args = parser.parse_args()
    analyzer = AspectAnalyzer()
    if args.text:
        print(format_aspects(args.text, analyzer.analyze(args.text)))
    if args.evaluate:
        output = evaluate(analyzer)
        print(output)
        LOG_PATH.write_text(output + "\n", encoding="utf-8")
        print(f"\n저장: {LOG_PATH}")
    if not (args.text or args.evaluate):
        parser.error("--text 또는 --evaluate가 필요합니다.")


if __name__ == "__main__":
    main()
