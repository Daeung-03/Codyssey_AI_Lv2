"""정보 추출 테스트 문장으로 패턴별 정밀도·재현율을 계산하고 실패 사례를 유형별로 분류한다.

채점: 정답과 예측의 유형·정규화 값이 같고 원문 위치가 겹치면 TP.
남은 정답은 FN, 남은 예측은 FP다.

실패 유형
- unsupported:    어떤 규칙도 정답 위치를 매칭하지 않음 (표기 미지원)
- boundary:       같은 유형으로 잡았지만 범위가 달라 값이 틀림 (경계 오류)
- normalization:  범위는 맞지만 정규화 값이 틀림
- overlap:        후보는 있었지만 겹침 정리에서 다른 후보에 밀림
- false_positive: 정답이 없는 곳을 추출 (오탐)
"""

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from entities import CORE_TYPES, KIND_LABELS, Entity
from extraction_testset import OUTPUT_PATH as TESTSET_PATH
from extraction_testset import ExtractionCase, load_cases
from extractor import extract, find_candidates, format_value


HOLDOUT_PATH = Path(__file__).parent / "data" / "extraction_holdout.jsonl"
LOG_PATH = Path(__file__).parent / "docs" / "extraction_eval.log"

FAILURE_LABELS = {
    "unsupported": "표기 미지원",
    "boundary": "경계 오류",
    "normalization": "정규화 오류",
    "overlap": "겹침 정리 탈락",
    "false_positive": "오탐",
}


@dataclass(frozen=True)
class Failure:
    case: ExtractionCase
    type: str
    gold: Entity | None  # 놓친 정답 (오탐이면 None)
    pred: Entity | None  # 잘못된 예측 (놓치기만 했으면 None)


def locate(text: str, entities: tuple[Entity, ...]) -> list[Entity]:
    """정답의 raw를 원문에서 찾아 span을 붙인다. 같은 raw가 여러 번이면 앞에서부터 차례로 쓴다."""
    next_start: Counter[str] = Counter()
    located = []
    for entity in entities:
        start = text.find(entity.raw, next_start[entity.raw])
        if start < 0:
            raise ValueError(f"정답 표기가 원문에 없습니다: {entity.raw!r} in {text!r}")
        next_start[entity.raw] = start + len(entity.raw)
        located.append(replace(entity, span=(start, start + len(entity.raw))))
    return located


def overlaps(a: Entity, b: Entity) -> bool:
    return a.span[0] < b.span[1] and b.span[0] < a.span[1]


def evaluate_case(case: ExtractionCase, kinds=CORE_TYPES) -> tuple[list[Entity], list[Failure]]:
    """한 문장의 TP 목록과 실패 목록을 반환한다."""
    gold = [g for g in locate(case.text, case.entities) if g.kind in kinds]
    preds = extract(case.text, kinds)

    used: set[int] = set()
    true_positives, missed = [], []
    for g in gold:
        hit = next(
            (i for i, p in enumerate(preds) if i not in used and p.key() == g.key() and overlaps(p, g)),
            None,
        )
        if hit is None:
            missed.append(g)
        else:
            used.add(hit)
            true_positives.append(g)

    failures = []
    candidates = None
    for g in missed:
        same_kind = [i for i, p in enumerate(preds) if i not in used and p.kind == g.kind and overlaps(p, g)]
        if same_kind:
            pred = preds[same_kind[0]]
            used.add(same_kind[0])
            failure_type = "normalization" if pred.span == g.span else "boundary"
            failures.append(Failure(case, failure_type, g, pred))
            continue
        if candidates is None:
            candidates = find_candidates(case.text, kinds)
        dropped = any(c.kind == g.kind and overlaps(c, g) for c in candidates)
        failures.append(Failure(case, "overlap" if dropped else "unsupported", g, None))

    failures += [Failure(case, "false_positive", None, p) for i, p in enumerate(preds) if i not in used]
    return true_positives, failures


def evaluate(cases: list[ExtractionCase], kinds=CORE_TYPES) -> tuple[dict[str, Counter], list[Failure]]:
    """유형별 TP·FP·FN과 전체 실패 목록."""
    counts = {kind: Counter() for kind in kinds}
    all_failures = []
    for case in cases:
        true_positives, failures = evaluate_case(case, kinds)
        for g in true_positives:
            counts[g.kind]["tp"] += 1
        for failure in failures:
            if failure.gold is not None:
                counts[failure.gold.kind]["fn"] += 1
            if failure.pred is not None:
                counts[failure.pred.kind]["fp"] += 1
        all_failures += failures
    return counts, all_failures


def prf(count: Counter) -> tuple[float, float, float]:
    tp, fp, fn = count["tp"], count["fp"], count["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_table(counts: dict[str, Counter]) -> str:
    lines = [f"{'패턴':<8}{'TP':>5}{'FP':>5}{'FN':>5}{'Precision':>11}{'Recall':>9}{'F1':>8}"]
    total = Counter()
    for kind, count in counts.items():
        total += count
        lines.append(_score_row(KIND_LABELS[kind], count))
    lines.append(_score_row("전체", total))
    return "\n".join(lines)


def _score_row(name: str, count: Counter) -> str:
    precision, recall, f1 = prf(count)
    return (f"{name:<8}{count['tp']:>5}{count['fp']:>5}{count['fn']:>5}"
            f"{precision:>11.3f}{recall:>9.3f}{f1:>8.3f}")


def _describe(entity: Entity | None) -> str:
    if entity is None:
        return "(없음)"
    return f"{KIND_LABELS[entity.kind]} {entity.raw!r} → {format_value(entity)}"


def failure_report(failures: list[Failure]) -> str:
    by_type = Counter(f.type for f in failures)
    lines = ["유형별 건수: " + ", ".join(f"{FAILURE_LABELS[t]} {n}" for t, n in by_type.most_common())]
    for failure in sorted(failures, key=lambda f: (list(FAILURE_LABELS).index(f.type), f.case.case_id)):
        lines.append(f"\n[{FAILURE_LABELS[failure.type]}] {failure.case.case_id} ({failure.case.variant})")
        lines.append(f"  입력: {failure.case.text!r}")
        lines.append(f"  정답: {_describe(failure.gold)}")
        lines.append(f"  예측: {_describe(failure.pred)}")
        if failure.case.note:
            lines.append(f"  메모: {failure.case.note}")
    return "\n".join(lines)


def report_section(title: str, cases: list[ExtractionCase]) -> str:
    counts, failures = evaluate(cases)
    easy_counts, _ = evaluate([case for case in cases if not case.hard])
    hard_n = sum(case.hard for case in cases)
    parts = [f"=== {title} ({len(cases)}문장) ===", score_table(counts)]
    if hard_n:
        parts.append(f"\n[hard {hard_n}문장 제외]\n" + score_table(easy_counts))
    parts.append(f"\n--- 실패 사례 {len(failures)}건 ---\n" + (failure_report(failures) if failures else "없음"))
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="정보 추출 성능(패턴별 P/R)과 실패 사례를 기록한다.")
    parser.add_argument("--log", type=Path, default=LOG_PATH, help="결과를 저장할 로그 경로")
    args = parser.parse_args()

    sections = [report_section("개발 테스트셋", load_cases(TESTSET_PATH))]
    if HOLDOUT_PATH.exists():
        sections.append(report_section("홀드아웃 (패턴을 보지 않고 작성)", load_cases(HOLDOUT_PATH)))
    output = "\n\n".join(sections)

    print(output)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(output + "\n", encoding="utf-8")
    print(f"\n저장: {args.log}")


if __name__ == "__main__":
    main()
