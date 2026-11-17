"""정보 추출 결과와 정답(gold)을 같은 형태로 표현하는 공용 자료형.

추출기(extractor.py)의 예측과 테스트 문장 생성기(extraction_testset.py)의 정답이
모두 `Entity`를 쓰므로, 평가에서는 `Entity.key()`끼리 바로 비교할 수 있다.
"""

from dataclasses import dataclass, field


# 필수 5개 패턴
EMAIL = "email"
PHONE = "phone"
DATE = "date"
MONEY = "money"
URL = "url"
CORE_TYPES = (EMAIL, PHONE, DATE, MONEY, URL)

# 보너스: 개인정보(PII) 패턴
RRN = "rrn"  # 주민등록번호·외국인등록번호
CARD = "card"
ADDRESS = "address"
PII_TYPES = (RRN, CARD, ADDRESS)

ENTITY_TYPES = CORE_TYPES + PII_TYPES
CURRENCIES = ("KRW", "USD")


@dataclass(frozen=True)
class Entity:
    """텍스트에서 찾은(또는 찾아야 하는) 정보 하나.

    kind:       ENTITY_TYPES 중 하나
    raw:        원문에 나타난 표기 그대로
    normalized: 정규화 값. 금액만 정수, 나머지는 문자열
    unit:       금액의 통화(KRW/USD). 금액이 아니면 None
    span:       원문에서의 (시작, 끝) 위치. 추출기만 채우며 비교에는 쓰지 않는다
    """

    kind: str
    raw: str
    normalized: str | int
    unit: str | None = None
    span: tuple[int, int] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in ENTITY_TYPES:
            raise ValueError(f"알 수 없는 정보 유형: {self.kind}")
        if not self.raw:
            raise ValueError("raw는 비어 있을 수 없습니다.")
        if self.kind == MONEY:
            if not isinstance(self.normalized, int) or self.unit not in CURRENCIES:
                raise ValueError(f"금액은 정수 값과 통화가 필요합니다: {self}")
        elif not isinstance(self.normalized, str) or self.unit is not None:
            raise ValueError(f"금액 외 정보는 문자열 값만 가집니다: {self}")

    def key(self) -> tuple[str, str | int, str | None]:
        """평가에서 정답·예측을 맞춰 볼 때 쓰는 값. 표기(raw)가 달라도 정규화 값이 같으면 일치."""
        return self.kind, self.normalized, self.unit

    def to_dict(self) -> dict:
        data = {"kind": self.kind, "raw": self.raw, "normalized": self.normalized}
        if self.unit is not None:
            data["unit"] = self.unit
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(data["kind"], data["raw"], data["normalized"], data.get("unit"))
