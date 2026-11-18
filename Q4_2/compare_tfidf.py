"""[보너스] 규칙 기반 감성 분석과 TF-IDF + 로지스틱 회귀를 같은 평가셋에서 비교한다.

- 평가셋: evaluate_sentiment.py와 같은 test 1,000개
- TF-IDF 학습 데이터: train 후보 풀(20,000개)에서 크기별로 긍정·부정 같은 수를 뽑는다.
  크기 10·100·1,000(과제 지정)과 10,000(추세 확인용). 작은 표본은 운에 따라 결과가 크게 흔들려서
  시드 5개의 평균·표준편차를 낸다.
- 두 방법 모두 같은 Okt 토큰을 쓴다. 규칙 기반은 학습하지 않으므로 크기와 무관한 한 값이다.
"""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from lexicon import load_lexicon
from nsmc_data import eval_set, sample_balanced, split_train
from sentiment import RuleConfig, SentimentAnalyzer


DOCS_DIR = Path(__file__).parent / "docs"
LOG_PATH = DOCS_DIR / "rule_vs_tfidf.log"
PLOT_PATH = DOCS_DIR / "rule_vs_tfidf.svg"
TRAIN_SIZES = (10, 100, 1_000, 10_000)
SEEDS = range(5)

# 차트 색 (dataviz 기본 팔레트 1·2번, 밝은 배경에서 색각 이상 검사 통과)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b68", "#e6e6e3"
TFIDF_COLOR, RULE_COLOR = "#2a78d6", "#eb6834"


def identity(tokens):
    return tokens


def train_and_predict(train_tokens, train_labels, test_tokens):
    vectorizer = TfidfVectorizer(analyzer=identity)
    model = LogisticRegression(max_iter=1_000)
    model.fit(vectorizer.fit_transform(train_tokens), train_labels)
    return model.predict(vectorizer.transform(test_tokens))


def subset_accuracy(labels, preds, mask) -> float:
    picked = [(y, p) for y, p, m in zip(labels, preds, mask) if m]
    return sum(y == p for y, p in picked) / len(picked)


def plot(rows, rule_acc: float, path: Path) -> None:
    plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    sizes = [row["size"] for row in rows]
    means = [row["acc_mean"] for row in rows]
    stds = [row["acc_std"] for row in rows]

    ax.axhline(rule_acc, color=RULE_COLOR, linewidth=2, linestyle=(0, (6, 3)), label="규칙 기반 (학습 없음)")
    ax.errorbar(sizes, means, yerr=stds, color=TFIDF_COLOR, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, capsize=4, label="TF-IDF + 로지스틱 회귀 (시드 5개 평균 ± 표준편차)")
    for size, mean in zip(sizes, means):
        ax.annotate(f"{mean:.3f}", (size, mean), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=9, color=INK)
    ax.annotate(f"규칙 기반 {rule_acc:.3f}", (sizes[0], rule_acc), textcoords="offset points", xytext=(0, 7),
                fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("TF-IDF 학습 리뷰 수 (긍정·부정 같은 수)", color=MUTED)
    ax.set_ylabel("평가셋 정확도", color=MUTED)
    ax.set_ylim(0.4, 0.9)
    ax.set_title("학습 데이터 크기별 감성 분류 정확도 (NSMC 평가셋 1,000개)", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="규칙 기반 vs TF-IDF 감성 분류 비교")
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    parser.add_argument("--plot", type=Path, default=PLOT_PATH)
    args = parser.parse_args()

    analyzer = SentimentAnalyzer(load_lexicon(), RuleConfig())
    pool, _ = split_train()
    test = eval_set()
    print("토큰화 중…")
    tokens = {review.text: list(t for t, _ in analyzer.tokenize(review.text)) for review in pool + test}

    test_tokens = [tokens[r.text] for r in test]
    test_labels = [r.label for r in test]
    rule_results = [analyzer.score_tokens(r.text, analyzer.tokenize(r.text)) for r in test]
    rule_preds = [result.label for result in rule_results]
    rule_acc = accuracy_score(test_labels, rule_preds)
    rule_f1 = f1_score(test_labels, rule_preds, average="macro")
    has_hit = [bool(result.hits) for result in rule_results]
    no_hit = [not h for h in has_hit]

    rows = []
    for size in TRAIN_SIZES:
        accs, f1s, hit_accs, nohit_accs = [], [], [], []
        for seed in SEEDS:
            train = sample_balanced(pool, size, seed)
            preds = train_and_predict([tokens[r.text] for r in train], [r.label for r in train], test_tokens)
            accs.append(accuracy_score(test_labels, preds))
            f1s.append(f1_score(test_labels, preds, average="macro"))
            hit_accs.append(subset_accuracy(test_labels, preds, has_hit))
            nohit_accs.append(subset_accuracy(test_labels, preds, no_hit))
        rows.append({
            "size": size,
            "acc_mean": statistics.mean(accs), "acc_std": statistics.stdev(accs),
            "f1_mean": statistics.mean(f1s), "f1_std": statistics.stdev(f1s),
            "hit_acc": statistics.mean(hit_accs), "nohit_acc": statistics.mean(nohit_accs),
        })

    lines = [
        f"평가셋 test {len(test)}개, TF-IDF는 크기별 시드 {len(SEEDS)}개 평균 ± 표준편차",
        "",
        f"{'방법':<22}{'Accuracy':>16}{'Macro F1':>16}",
        f"{'규칙 기반 (학습 없음)':<22}{rule_acc:>16.3f}{rule_f1:>16.3f}",
    ]
    for row in rows:
        lines.append(f"{'TF-IDF n=' + format(row['size'], ','):<22}"
                     f"{row['acc_mean']:>10.3f} ±{row['acc_std']:.3f}{row['f1_mean']:>10.3f} ±{row['f1_std']:.3f}")

    n_hit = sum(has_hit)
    lines += [
        "",
        f"사전 감성어가 있는 리뷰 {n_hit}개 / 없는 리뷰 {len(test) - n_hit}개로 나눈 정확도",
        f"{'방법':<22}{'감성어 있음':>12}{'감성어 없음':>12}",
        f"{'규칙 기반':<22}{subset_accuracy(test_labels, rule_preds, has_hit):>12.3f}"
        f"{subset_accuracy(test_labels, rule_preds, no_hit):>12.3f}",
    ]
    for row in rows:
        lines.append(f"{'TF-IDF n=' + format(row['size'], ','):<22}{row['hit_acc']:>12.3f}{row['nohit_acc']:>12.3f}")

    output = "\n".join(lines)
    print(output)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(output + "\n", encoding="utf-8")
    plot(rows, rule_acc, args.plot)
    print(f"\n저장: {args.log}, {args.plot}")


if __name__ == "__main__":
    main()
