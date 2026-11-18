"""정규표현식 후보를 정규화·검증하고 겹치는 후보를 정리해 정보를 추출한다.

처리 순서
1. patterns.RULES의 각 규칙으로 후보를 모두 찾는다.
2. 후보를 정규화한다. 정규화할 수 없는 후보(2월 30일, 0원 등)는 버린다.
3. 서로 겹치는 후보 중 긴 것을 남긴다. 길이가 같으면 유형 우선순위로 고른다.
"""

import argparse
import re
from collections.abc import Iterable
from datetime import date

from entities import (
    ADDRESS,
    CARD,
    CORE_TYPES,
    DATE,
    EMAIL,
    KIND_LABELS,
    MONEY,
    PHONE,
    RRN,
    URL,
    Entity,
)
from patterns import RULES, PatternRule


# 길이가 같은 후보가 겹칠 때의 우선순위 (작을수록 우선). 더 구체적인 형식을 앞에 둔다.
KIND_PRIORITY = {URL: 0, EMAIL: 1, RRN: 2, CARD: 3, PHONE: 4, DATE: 5, MONEY: 6, ADDRESS: 7}

MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

SMALL_UNITS = {"십": 10, "백": 100, "천": 1_000}
LARGE_UNITS = {"만": 10**4, "억": 10**8, "조": 10**12}
_AMOUNT_TOKEN = re.compile(r"(?P<number>\d[\d,]*)?\s*(?P<unit>[십백천]?[만억조]|[십백천])?")


# --- 유형별 정규화 -----------------------------------------------------------

def normalize_email(match: re.Match) -> str:
    """전체를 소문자로 통일한다."""
    return match.group(0).lower()


def normalize_phone(match: re.Match) -> str | None:
    """국가번호를 떼고 0을 복원한 뒤 하이픈 3묶음(대표번호는 2묶음)으로 만든다."""
    groups = match.groupdict()
    if groups.get("rep"):
        return f"{groups['rep']}-{groups['last']}"
    if match.group(0).startswith("+"):
        area = "0" + groups["area"]
    else:
        area = groups.get("paren_area") or groups["area"]
    return f"{area}-{groups['mid']}-{groups['last']}"


def normalize_date(match: re.Match) -> str | None:
    """ISO 8601로 바꾼다. 연도 없음은 --MM-DD, 일 없음은 YYYY-MM. 달력에 없는 날짜는 None."""
    groups = match.groupdict()
    if groups.get("month_name"):
        month = MONTH_NAMES[groups["month_name"][:3].lower()]
    else:
        month = int(groups["month"])
    year = int(groups["year"]) if groups.get("year") else None
    day = int(groups["day"]) if groups.get("day") else None

    if day is None:
        if year is None or not 1 <= month <= 12:
            return None
        return f"{year:04d}-{month:02d}"
    try:
        # 연도가 없으면 2월 29일도 허용하도록 윤년(2000)으로 검사한다.
        date(year or 2000, month, day)
    except ValueError:
        return None
    if year is None:
        return f"--{month:02d}-{day:02d}"
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_amount(body: str) -> int:
    """'1억 2천만', '1만 5천', '10,000' 같은 금액 본문을 정수로 바꾼다.

    만·억·조 단위가 나올 때마다 그동안 모은 값(section)에 큰 단위를 곱해 total에 더한다.
    예) 1억 2천만 → (1)×1억 + (2×1000)×1만 = 120,000,000
    """
    total = section = 0
    for token in _AMOUNT_TOKEN.finditer(body):
        number, unit = token.group("number"), token.group("unit")
        if not number and not unit:
            continue
        value = int(number.replace(",", "")) if number else 1
        unit = unit or ""
        if unit[:1] in SMALL_UNITS:
            value *= SMALL_UNITS[unit[0]]
            unit = unit[1:]
        if unit in LARGE_UNITS:
            total += (section + value) * LARGE_UNITS[unit]
            section = 0
        else:
            section += value
    return total + section


def normalize_money(match: re.Match) -> int | None:
    value = parse_amount(match.group("body"))
    return value if value > 0 else None


def normalize_url(match: re.Match) -> str:
    """스킴·호스트만 소문자로 바꾸고, 스킴이 없으면 https://를 붙인다. 경로·쿼리는 그대로 둔다."""
    scheme = (match.group("scheme") or "https").lower()
    rest = match.group("rest")
    host, path = re.match(r"([^/?#]*)(.*)", rest).groups()
    return f"{scheme}://{host.lower()}{path}"


NORMALIZERS = {
    EMAIL: normalize_email,
    PHONE: normalize_phone,
    DATE: normalize_date,
    MONEY: normalize_money,
    URL: normalize_url,
}


# --- 추출 -------------------------------------------------------------------

def to_entity(rule: PatternRule, match: re.Match) -> Entity | None:
    """정규식 매칭 하나를 Entity로 바꾼다. 정규화에 실패하면 None."""
    normalized = NORMALIZERS[rule.kind](match)
    if normalized is None:
        return None
    return Entity(rule.kind, match.group(0), normalized, rule.unit, span=match.span())


def find_candidates(text: str, kinds: Iterable[str] = CORE_TYPES) -> list[Entity]:
    """겹침을 정리하기 전의 모든 후보. 평가에서 규칙별 동작을 확인할 때도 쓴다."""
    wanted = set(kinds)
    candidates = []
    for rule in RULES:
        if rule.kind not in wanted:
            continue
        for match in rule.regex.finditer(text):
            entity = to_entity(rule, match)
            if entity is not None:
                candidates.append(entity)
    return candidates


def resolve_overlaps(candidates: list[Entity]) -> list[Entity]:
    """겹치는 후보 중 긴 것 → 우선순위 높은 것 → 앞에 나온 것 순으로 남긴다."""
    ordered = sorted(
        candidates,
        key=lambda e: (-(e.span[1] - e.span[0]), KIND_PRIORITY[e.kind], e.span[0]),
    )
    chosen: list[Entity] = []
    for entity in ordered:
        start, end = entity.span
        if all(end <= other.span[0] or other.span[1] <= start for other in chosen):
            chosen.append(entity)
    return sorted(chosen, key=lambda e: e.span)


def extract(text: str, kinds: Iterable[str] = CORE_TYPES) -> list[Entity]:
    """텍스트에서 정보를 추출해 등장 순서대로 반환한다."""
    return resolve_overlaps(find_candidates(text, kinds))


def format_value(entity: Entity) -> str:
    if entity.kind == MONEY:
        return f"{entity.normalized} {entity.unit}"
    return str(entity.normalized)


def format_extraction(text: str, entities: list[Entity]) -> str:
    """PROBLEM.md 결과 예시와 같은 형식의 출력 문자열."""
    lines = ["=== 정보 추출 결과 ===", f'입력: "{text}"', "", "추출 결과:"]
    if not entities:
        lines.append("- (없음)")
    for entity in entities:
        value = format_value(entity)
        note = "" if value == entity.raw else f"  (원문: {entity.raw})"
        lines.append(f"- {KIND_LABELS[entity.kind]}: {value}{note}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="정규표현식으로 이메일·전화·날짜·금액·URL을 추출한다.")
    parser.add_argument("--text", required=True, help="분석할 문장")
    args = parser.parse_args()
    print(format_extraction(args.text, extract(args.text)))


if __name__ == "__main__":
    main()
