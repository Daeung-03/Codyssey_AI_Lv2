"""감성 분석 평가: 규칙 조합별 Accuracy·F1, 규칙이 적용되는 리뷰만 본 비교, 오분류 사례.

개발셋(train에서 추출)으로 0점 처리 기준과 규칙을 조정하고, 최종 수치는 평가셋(test)에서 낸다.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from lexicon import INTENSIFIERS, POST_NEGATORS, PRE_NEGATORS, load_lexicon
from nsmc_data import Review, eval_set, split_train
from sentiment import LABEL_NAMES, RuleConfig, SentimentAnalyzer, SentimentResult


LOG_PATH = Path(__file__).parent / "docs" / "sentiment_eval.log"
MISCLASSIFIED_SAMPLE = 20

CONFIGS = [
    ("사전만", RuleConfig(negation=False, intensifier=False, double_negation=False)),
    ("+부정어", RuleConfig(negation=True, intensifier=False, double_negation=False)),
    ("+강조어", RuleConfig(negation=False, intensifier=True, double_negation=False)),
    ("+부정어+강조어", RuleConfig(negation=True, intensifier=True, double_negation=False)),
    ("+부정어+강조어+이중부정", RuleConfig(negation=True, intensifier=True, double_negation=True)),
]
FINAL = CONFIGS[-1][0]


def score_all(reviews: list[Review], analyzer: SentimentAnalyzer) -> dict[str, list[SentimentResult]]:
    """토큰화는 한 번만 하고 규칙 설정마다 다시 채점한다."""
    tokenized = [(review.text, analyzer.tokenize(review.text)) for review in reviews]
    results = {}
    for name, rules in CONFIGS:
        scorer = SentimentAnalyzer(analyzer.lexicon, rules, analyzer.zero_label, analyzer.tagger)
        results[name] = [scorer.score_tokens(text, tokens) for text, tokens in tokenized]
    return results


def metrics(reviews: list[Review], results: list[SentimentResult]) -> tuple[float, float]:
    y_true = [review.label for review in reviews]
    y_pred = [result.label for result in results]
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def config_table(reviews: list[Review], results: dict[str, list[SentimentResult]]) -> str:
    lines = [f"{'설정':<24}{'Accuracy':>10}{'Macro F1':>10}"]
    for name, _ in CONFIGS:
        accuracy, f1 = metrics(reviews, results[name])
        lines.append(f"{name:<24}{accuracy:>10.3f}{f1:>10.3f}")
    return "\n".join(lines)


def subset_table(reviews: list[Review], results: dict[str, list[SentimentResult]]) -> str:
    """규칙이 실제로 작동할 수 있는 리뷰만 골라 규칙 전후 정확도를 비교한다."""
    final = results[FINAL]
    subsets = {
        "부정어 포함": lambda r: any(t in PRE_NEGATORS | POST_NEGATORS for t, _ in r.tokens),
        "부정어가 감성어를 반전": lambda r: any(h.negator for h in r.hits),
        "강조어가 감성어에 적용": lambda r: any(h.intensifier for h in r.hits),
        "이중부정 탐지": lambda r: bool(r.double_negations),
    }
    names = [name for name, _ in CONFIGS]
    lines = [f"{'부분집합':<22}{'n':>5}" + "".join(f"{i + 1:>8}" for i in range(len(names)))]
    for subset, rule in subsets.items():
        idx = [i for i, result in enumerate(final) if rule(result)]
        row = f"{subset:<22}{len(idx):>5}"
        for name in names:
            correct = sum(results[name][i].label == reviews[i].label for i in idx)
            row += f"{correct / len(idx):>8.3f}" if idx else f"{'-':>8}"
        lines.append(row)
    lines.append("열 번호: " + ", ".join(f"{i + 1}={name}" for i, name in enumerate(names)))
    return "\n".join(lines)


def coverage(results: list[SentimentResult]) -> str:
    no_hit = sum(not result.hits for result in results)
    zero = sum(result.zero for result in results)
    return (f"사전 감성어가 하나도 없는 리뷰: {no_hit}/{len(results)} ({no_hit / len(results):.1%}), "
            f"점수 0 리뷰(기본값 판정): {zero}")


# NSMC에는 이중부정이 거의 없어(개발셋 1건) 규칙 동작을 직접 쓴 문장으로도 확인한다. (문장, 정답)
RULE_EXAMPLES = [
    ("이 식당은 정말 맛있지 않았어요", 0),
    ("좋은 영화는 아니다", 0),
    ("안 좋아요", 0),
    ("공감은 못하겠다", 0),
    ("긴장감도 별로 없고", 0),
    ("감동하지 않을 수 없다", 1),
    ("재미가 없지 않았다", 1),
    ("지루하지 않은 것은 아니다", 0),
    ("조금 지루했지만 결말은 정말 감동적이었다", 1),
    ("별 기대 안 하고 봤는데 꿀잼", 1),
]


def rule_examples(analyzer: SentimentAnalyzer) -> str:
    names = [name for name, _ in CONFIGS]
    lines = [f"{'정답':<4}" + "".join(f"{i + 1:>6}" for i in range(len(names))) + "  문장"]
    for text, label in RULE_EXAMPLES:
        tokens = analyzer.tokenize(text)
        row = f"{LABEL_NAMES[label]:<4}"
        for _, rules in CONFIGS:
            result = SentimentAnalyzer(analyzer.lexicon, rules, analyzer.zero_label, analyzer.tagger).score_tokens(text, tokens)
            mark = "O" if result.label == label else "X"
            row += f"{result.score:>+5g}{mark}"
        lines.append(f"{row}  {text}")
    lines.append("열: 규칙 설정 번호별 점수와 정답 여부(O/X)")
    return "\n".join(lines)


def zero_label_decision(reviews: list[Review], results: list[SentimentResult]) -> str:
    labels = Counter(review.label for review, result in zip(reviews, results) if result.zero)
    return f"0점 리뷰의 실제 레이블: 부정 {labels[0]}, 긍정 {labels[1]} → 다수 레이블 {LABEL_NAMES[labels.most_common(1)[0][0]]}"


def misclassified(reviews: list[Review], results: list[SentimentResult], k: int, seed: int = 42) -> str:
    wrong = [(review, result) for review, result in zip(reviews, results) if review.label != result.label]
    zero_wrong = sum(result.zero for _, result in wrong)
    lines = [f"오분류 {len(wrong)}건 (0점으로 기본값 판정 {zero_wrong}건, 점수 부호가 반대 {len(wrong) - zero_wrong}건)"]
    # 0점 오분류는 원인이 '사전 미적중' 하나라 분석 표본에서는 점수가 있는 오분류를 고른다.
    scored = [(review, result) for review, result in wrong if not result.zero]
    for review, result in random.Random(seed).sample(scored, min(k, len(scored))):
        hits = ", ".join(
            f"{h.word}{'(반전:' + h.negator + ')' if h.negator else ''}"
            f"{'(×' + format(h.multiplier, 'g') + ')' if h.intensifier else ''}={h.score:+g}"
            for h in result.hits
        )
        lines.append(f"\n[{review.doc_id}] 정답 {LABEL_NAMES[review.label]} / 예측 {LABEL_NAMES[result.label]} ({result.score:+g})")
        lines.append(f"  리뷰: {review.text}")
        lines.append(f"  토큰: {' '.join(t for t, _ in result.tokens)}")
        lines.append(f"  감성어: {hits}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="감성 분석 성능 평가")
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    args = parser.parse_args()

    analyzer = SentimentAnalyzer(load_lexicon())
    _, dev = split_train()
    test = eval_set()
    dev_results = score_all(dev, analyzer)
    test_results = score_all(test, analyzer)

    output = "\n\n".join([
        f"=== 개발셋 (train {len(dev)}개) ===",
        zero_label_decision(dev, dev_results[FINAL]),
        coverage(dev_results[FINAL]),
        config_table(dev, dev_results),
        f"=== 평가셋 (test {len(test)}개) ===",
        coverage(test_results[FINAL]),
        config_table(test, test_results),
        "--- 규칙이 작동하는 리뷰만 본 정확도 (평가셋) ---\n" + subset_table(test, test_results),
        "--- 규칙 확인 문장 (직접 작성) ---\n" + rule_examples(analyzer),
        "--- 오분류 사례 (평가셋, 최종 설정) ---\n" + misclassified(test, test_results[FINAL], MISCLASSIFIED_SAMPLE),
    ])
    print(output)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(output + "\n", encoding="utf-8")
    print(f"\n저장: {args.log}")


if __name__ == "__main__":
    main()
