"""감성 사전 기반 감성 분석기.

처리 순서
1. 토큰화: Okt pos(norm=True, stem=True). '맛있지 않았어요' → 맛있다, 않다
2. 감성어 탐지: 사전에 있는 토큰의 점수(-2~+2)
3. 이중부정: 부정어 두 개가 '수/것' 등만 사이에 두고 이어지면 서로 상쇄 ('감동하지 않을 수 없다')
4. 부정어: 남은 부정어가 가까운 감성어의 극성을 한 번 뒤집는다
   - 뒤 부정어(않다·없다·아니다·못하다)는 앞쪽 감성어를 찾는다
   - 앞 부정어(안·못)는 바로 다음 말만 부정한다. '공감 못 하다'처럼 다음 말이 하다/되다면 앞 명사를 부정한다
   - 부정 결합어(별로·어이 등)는 뜻에 이미 부정이 들어 있어 뒤집지 않는다
5. 강조어: 감성어 앞의 강조어 배율을 곱한다 (정말 ×1.5, 조금 ×0.5)
6. 판정: 점수 합 > 0 긍정, < 0 부정, = 0 이면 ZERO_LABEL

찾는 거리는 조사를 빼고 센 토큰 수(WINDOW)로 제한하고, 문장부호를 넘지 않는다.
"""

import argparse
from dataclasses import dataclass

from konlpy.tag import Okt

from lexicon import (
    DOUBLE_NEGATION_FILLERS,
    INTENSIFIERS,
    LIGHT_VERBS,
    NEGATION_IMMUNE,
    POST_NEGATORS,
    PRE_NEGATORS,
    load_lexicon,
)


WINDOW = 2
# 사전에 걸리는 감성어가 없는 리뷰(0점)의 판정. 개발셋 0점 리뷰의 다수 레이블로 정했다 (evaluate_sentiment.py 참고).
ZERO_LABEL = 0
LABEL_NAMES = {0: "부정", 1: "긍정"}


@dataclass(frozen=True)
class RuleConfig:
    negation: bool = True
    intensifier: bool = True
    double_negation: bool = True


@dataclass(frozen=True)
class Hit:
    index: int
    word: str
    base: int
    negator: str | None = None  # 극성을 뒤집은 부정어
    intensifier: str | None = None
    multiplier: float = 1.0

    @property
    def score(self) -> float:
        return self.base * (-1 if self.negator else 1) * self.multiplier


@dataclass(frozen=True)
class SentimentResult:
    text: str
    tokens: tuple[tuple[str, str], ...]
    hits: tuple[Hit, ...]
    double_negations: tuple[str, ...]
    score: float
    label: int

    @property
    def zero(self) -> bool:
        return self.score == 0


def _content_positions(tokens, start: int, step: int, limit: int):
    """start부터 step 방향으로 조사를 건너뛰며 최대 limit개 위치를 낸다. 문장부호를 만나면 멈춘다."""
    i, seen = start, 0
    while 0 <= i < len(tokens) and seen < limit:
        token, pos = tokens[i]
        if pos == "Punctuation":
            return
        if pos != "Josa":
            seen += 1
            yield i
        i += step


class SentimentAnalyzer:
    def __init__(self, lexicon: dict[str, int] | None = None, rules: RuleConfig = RuleConfig(),
                 zero_label: int = ZERO_LABEL, tagger: Okt | None = None) -> None:
        self.lexicon = lexicon if lexicon is not None else load_lexicon()
        self.rules = rules
        self.zero_label = zero_label
        self.tagger = tagger or Okt()

    def tokenize(self, text: str) -> tuple[tuple[str, str], ...]:
        return tuple(self.tagger.pos(text.lower(), norm=True, stem=True))

    def analyze(self, text: str) -> SentimentResult:
        return self.score_tokens(text, self.tokenize(text))

    def score_tokens(self, text: str, tokens: tuple[tuple[str, str], ...]) -> SentimentResult:
        """토큰화가 끝난 문장에 규칙을 적용한다. 평가에서 설정만 바꿔 여러 번 채점할 때 쓴다."""
        hits = {i: Hit(i, token, self.lexicon[token]) for i, (token, _) in enumerate(tokens) if token in self.lexicon}

        cancelled, double_negations = self._double_negations(tokens)

        if self.rules.negation:
            for i, (token, _) in enumerate(tokens):
                if i in cancelled:
                    continue
                if token in POST_NEGATORS:
                    target = self._nearest_hit(tokens, hits, i - 1, -1)
                elif token in PRE_NEGATORS:
                    target = self._pre_negation_target(tokens, hits, i)
                else:
                    continue
                if target is not None and hits[target].negator is None:
                    hits[target] = _replace(hits[target], negator=token)

        if self.rules.intensifier:
            for i, hit in list(hits.items()):
                for j in _content_positions(tokens, i - 1, -1, WINDOW):
                    if tokens[j][0] in INTENSIFIERS:
                        hits[i] = _replace(hit, intensifier=tokens[j][0], multiplier=INTENSIFIERS[tokens[j][0]])
                        break

        ordered = tuple(hits[i] for i in sorted(hits))
        score = round(sum(hit.score for hit in ordered), 6)
        label = 1 if score > 0 else 0 if score < 0 else self.zero_label
        return SentimentResult(text, tokens, ordered, double_negations, score, label)

    def _double_negations(self, tokens) -> tuple[set[int], tuple[str, ...]]:
        """'않다 (수/것+조사) 없다/아니다', '없다 않다' 같은 부정어 쌍의 위치를 찾는다."""
        if not (self.rules.negation and self.rules.double_negation):
            return set(), ()
        cancelled: set[int] = set()
        found = []
        for i, (token, _) in enumerate(tokens):
            if token not in POST_NEGATORS or i in cancelled:
                continue
            for j in _content_positions(tokens, i + 1, +1, 3):
                other = tokens[j][0]
                if other in POST_NEGATORS:
                    cancelled |= {i, j}
                    found.append(" ".join(t for t, _ in tokens[i:j + 1]))
                    break
                if other not in DOUBLE_NEGATION_FILLERS:
                    break
        return cancelled, tuple(found)

    def _nearest_hit(self, tokens, hits, start: int, step: int, limit: int = WINDOW) -> int | None:
        """부정어가 뒤집을 감성어. 부정 결합어(별로 등)는 대상에서 뺀다."""
        for j in _content_positions(tokens, start, step, limit):
            if j in hits and hits[j].word not in NEGATION_IMMUNE:
                return j
        return None

    def _pre_negation_target(self, tokens, hits, i: int) -> int | None:
        """'안/못'은 바로 다음 말만 부정한다. 다음 말이 하다/되다면 앞 명사를 부정한다 ('공감 못 하다')."""
        following = next(_content_positions(tokens, i + 1, +1, 1), None)
        if following is None:
            return None
        if tokens[following][0] in LIGHT_VERBS:
            return self._nearest_hit(tokens, hits, i - 1, -1, limit=1)
        return self._nearest_hit(tokens, hits, following, +1, limit=1)


def _replace(hit: Hit, **changes) -> Hit:
    return Hit(**{**hit.__dict__, **changes})


def format_trace(result: SentimentResult) -> str:
    """PROBLEM.md의 '감성 분석 과정' 예시 형식."""
    def signed(value: float) -> str:
        return f"{value:+g}"

    words = ", ".join(f"{h.word}({signed(h.base)})" for h in result.hits) or "없음"
    negations = ", ".join(f"'{h.negator}' → '{h.word}' 반전" for h in result.hits if h.negator) or "없음"
    intensities = ", ".join(f"'{h.intensifier}' → '{h.word}' ×{h.multiplier:g}" for h in result.hits if h.intensifier) or "없음"
    doubles = ", ".join(f"'{d}' → 상쇄" for d in result.double_negations) or "없음"

    terms = []
    for h in result.hits:
        parts = [f"({signed(h.base)})"]
        if h.negator:
            parts.append("(-1)")
        if h.intensifier:
            parts.append(f"{h.multiplier:g}")
        terms.append(" × ".join(parts))
    formula = " + ".join(terms) or "0"
    label = LABEL_NAMES[result.label] + (" (감성어 없음: 기본값)" if result.zero else "")

    return "\n".join([
        "=== 감성 분석 과정 ===",
        f'문장: "{result.text}"',
        "",
        f"1. 토큰화: [{', '.join(t for t, _ in result.tokens)}]",
        f"2. 감성 단어 탐지: {words}",
        f"3. 이중부정 탐지: {doubles}",
        f"4. 부정어 탐지: {negations}",
        f"5. 강조어 탐지: {intensities}",
        "",
        f"계산: {formula} = {signed(result.score)}",
        f"최종 판정: {label}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="감성 사전으로 문장의 감성 점수를 계산한다.")
    parser.add_argument("--text", required=True, help="분석할 문장")
    args = parser.parse_args()
    print(format_trace(SentimentAnalyzer().analyze(args.text)))


if __name__ == "__main__":
    main()
