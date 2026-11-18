"""[보너스] 개인정보를 추출한 뒤 일부를 *로 가린다.

| 유형    | 규칙                                  | 예                         |
| 전화    | 가운데 묶음                           | 010-****-5678              |
| 주민번호| 뒷자리 첫 숫자(성별)만 남김           | 900101-1******             |
| 카드    | 앞 4자리·뒤 4자리만 남김              | 1234-****-****-3456        |
| 이메일  | 아이디 첫 글자만 남김                 | u***@domain.com            |
| 주소    | 시·도와 시·군·구까지만 남김           | 서울특별시 강남구 ***      |

가린 결과는 정규화된 형식으로 쓴다(01012345678 → 010-****-5678).
전국 대표번호(1588-1234)는 개인 번호가 아니므로 가리지 않는다.
"""

import argparse
import re

from entities import ADDRESS, CARD, EMAIL, KIND_LABELS, PHONE, RRN, Entity
from extractor import extract


MASK_KINDS = (RRN, CARD, PHONE, EMAIL, ADDRESS)


def mask_digits(value: str, keep_first: int, keep_last: int) -> str:
    """숫자 중 앞 keep_first개와 뒤 keep_last개만 남기고 *로 바꾼다. 하이픈은 그대로 둔다."""
    total = sum(ch.isdigit() for ch in value)
    seen = 0
    out = []
    for ch in value:
        if ch.isdigit():
            seen += 1
            ch = ch if seen <= keep_first or seen > total - keep_last else "*"
        out.append(ch)
    return "".join(out)


def mask_value(entity: Entity) -> str | None:
    """가린 문자열. 가릴 필요가 없으면 None."""
    value = str(entity.normalized)
    if entity.kind == PHONE:
        parts = value.split("-")
        if len(parts) != 3:  # 대표번호
            return None
        return f"{parts[0]}-{'*' * len(parts[1])}-{parts[2]}"
    if entity.kind == RRN:
        return mask_digits(value, keep_first=7, keep_last=0)
    if entity.kind == CARD:
        return mask_digits(value, keep_first=4, keep_last=4)
    if entity.kind == EMAIL:
        local, domain = value.split("@", 1)
        return f"{local[0]}{'*' * 3}@{domain}"
    if entity.kind == ADDRESS:
        tokens = value.split()
        kept = [tokens[0]]  # 시·도
        for token in tokens[1:3]:
            if re.search(r"(시|군|구)$", token):
                kept.append(token)
            else:
                break
        return " ".join(kept + ["***"])
    return None


def mask(text: str, kinds=MASK_KINDS) -> tuple[str, list[Entity]]:
    """개인정보를 가린 문장과 가린 정보 목록. 뒤쪽부터 바꿔 앞쪽 위치가 밀리지 않게 한다."""
    masked = text
    targets = []
    for entity in sorted(extract(text, kinds), key=lambda e: e.span, reverse=True):
        replacement = mask_value(entity)
        if replacement is None:
            continue
        start, end = entity.span
        masked = masked[:start] + replacement + masked[end:]
        targets.append(entity)
    return masked, targets[::-1]


def format_masking(text: str) -> str:
    masked, targets = mask(text)
    lines = ["=== 개인정보 마스킹 ===", f'입력: "{text}"', f'결과: "{masked}"']
    for entity in targets:
        lines.append(f"- {KIND_LABELS[entity.kind]}: {entity.raw} → {mask_value(entity)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="개인정보를 찾아 가린다.")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(format_masking(args.text))


if __name__ == "__main__":
    main()
