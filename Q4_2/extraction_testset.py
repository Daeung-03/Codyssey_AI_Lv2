"""정보 추출 평가용 테스트 문장을 만들고 정답과 함께 JSONL로 저장한다.

문장은 네 묶음으로 구성한다.
1) 단일 패턴 문장: 패턴(이메일·전화·날짜·금액·URL)마다 표기 변형을 직접 작성했다.
   규칙이 놓칠 것으로 예상되는 어려운 변형(hard)도 일부러 넣어 재현율 한계를 드러낸다.
2) 오탐 유도 문장(negative): 정답이 없거나, 패턴과 비슷하지만 정보가 아닌 표현을 넣어 정밀도를 본다.
3) 복합 문의 문장(mixed): 직접 고른 값 목록을 고객 문의 템플릿에 고정 시드로 조합한다.
4) 개인정보 문장(보너스): 주민등록번호·카드번호·주소 패턴과 마스킹 평가에 쓴다.

정답(normalized)은 추출기 코드를 호출하지 않고 사람이 직접 적었다.
정규화 규격은 docs/PLAN.md의 "정규화 규격" 표를 따른다.
"""

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from entities import (
    ADDRESS,
    CARD,
    CORE_TYPES,
    DATE,
    EMAIL,
    MONEY,
    PHONE,
    PII_TYPES,
    RRN,
    URL,
    Entity,
)


OUTPUT_PATH = Path(__file__).parent / "data" / "extraction_testset.jsonl"
RANDOM_STATE = 42
MIXED_CASE_COUNT = 20
MIN_TOTAL_CASES = 50
MIN_CASES_PER_PATTERN = 10
MIN_VARIANTS_PER_PATTERN = 3

NEGATIVE = "negative"
MIXED = "mixed"


@dataclass(frozen=True)
class ExtractionCase:
    case_id: str
    category: str  # 주로 검증하는 패턴, negative, mixed 중 하나
    variant: str  # 표기 변형 이름
    text: str
    entities: tuple[Entity, ...]
    hard: bool = False  # 규칙 기반으로 놓치기 쉬울 것으로 예상한 사례
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.case_id,
            "category": self.category,
            "variant": self.variant,
            "text": self.text,
            "entities": [entity.to_dict() for entity in self.entities],
            "hard": self.hard,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionCase":
        return cls(
            case_id=data["id"],
            category=data["category"],
            variant=data["variant"],
            text=data["text"],
            entities=tuple(Entity.from_dict(item) for item in data["entities"]),
            hard=data.get("hard", False),
            note=data.get("note", ""),
        )


# --- 정답 작성 도우미: normalized를 생략하면 raw가 이미 정규화된 표기라는 뜻이다 ---

def email(raw: str, normalized: str | None = None) -> Entity:
    return Entity(EMAIL, raw, normalized or raw)


def phone(raw: str, normalized: str | None = None) -> Entity:
    return Entity(PHONE, raw, normalized or raw)


def date(raw: str, normalized: str) -> Entity:
    return Entity(DATE, raw, normalized)


def krw(raw: str, value: int) -> Entity:
    return Entity(MONEY, raw, value, "KRW")


def usd(raw: str, value: int) -> Entity:
    return Entity(MONEY, raw, value, "USD")


def url(raw: str, normalized: str | None = None) -> Entity:
    return Entity(URL, raw, normalized or raw)


def rrn(raw: str, normalized: str | None = None) -> Entity:
    return Entity(RRN, raw, normalized or raw)


def card(raw: str, normalized: str | None = None) -> Entity:
    return Entity(CARD, raw, normalized or raw)


def address(raw: str) -> Entity:
    # 주소 정규화: 연속 공백을 한 칸으로 줄이는 것만 한다.
    return Entity(ADDRESS, raw, " ".join(raw.split()))


# (variant, text, 정답 목록, hard, note)
CaseSpec = tuple[str, str, tuple[Entity, ...], bool, str]


def spec(variant: str, text: str, *entities: Entity, hard: bool = False, note: str = "") -> CaseSpec:
    return variant, text, entities, hard, note


EMAIL_SPECS = [
    spec("basic", "문의: user@domain.com", email("user@domain.com")),
    spec(
        "subdomain_ccTLD_josa",
        "이메일은 test.user@sub.domain.co.kr입니다",
        email("test.user@sub.domain.co.kr"),
        note="주소 바로 뒤에 조사가 붙어 경계 처리가 필요하다.",
    ),
    spec("plus_tag", "알림은 alerts+billing@example.com 으로 옵니다", email("alerts+billing@example.com")),
    spec("underscore_digits", "담당자 메일 kim_01@example.org로 보내주세요", email("kim_01@example.org")),
    spec("hyphen_domain", "회신 주소: support@my-company.co.kr", email("support@my-company.co.kr")),
    spec(
        "uppercase",
        "Contact: John.Doe@Example.COM",
        email("John.Doe@Example.COM", "john.doe@example.com"),
        note="정규화 시 전체 소문자로 통일한다.",
    ),
    spec("angle_brackets", "보낸 사람 <noreply@service.example.net>", email("noreply@service.example.net")),
    spec(
        "trailing_period",
        "자세한 내용은 help@example.com.",
        email("help@example.com"),
        note="문장 끝 마침표는 주소에 포함하지 않는다.",
    ),
    spec(
        "two_addresses",
        "주 담당 a.kim@example.com, 부 담당 b.lee@example.com",
        email("a.kim@example.com"),
        email("b.lee@example.com"),
    ),
    spec("parentheses", "메일(cs@shop.example.kr)로 접수해 주세요", email("cs@shop.example.kr")),
    spec(
        "obfuscated",
        "메일은 user at example dot com 입니다",
        email("user at example dot com", "user@example.com"),
        hard=True,
        note="스팸 회피용 난독화 표기. @와 .이 없어 기본 패턴으로는 잡히지 않는다.",
    ),
    spec(
        "fullwidth_at",
        "문의 user＠example.com",
        email("user＠example.com", "user@example.com"),
        hard=True,
        note="전각 ＠ 문자. 유니코드 정규화(NFKC) 없이는 놓친다.",
    ),
]

PHONE_SPECS = [
    spec("mobile_hyphen", "연락처: 010-1234-5678", phone("010-1234-5678")),
    spec(
        "seoul_and_regional",
        "전화번호는 02-123-4567 또는 031-1234-5678",
        phone("02-123-4567"),
        phone("031-1234-5678"),
    ),
    spec("no_separator", "휴대폰 01098765432로 연락 주세요", phone("01098765432", "010-9876-5432")),
    spec("dot_separator", "Tel. 010.2345.6789", phone("010.2345.6789", "010-2345-6789")),
    spec("space_separator", "번호 010 3456 7890 입니다", phone("010 3456 7890", "010-3456-7890")),
    spec("parenthesized_area", "대표번호 (02)987-6543", phone("(02)987-6543", "02-987-6543")),
    spec(
        "international_hyphen",
        "해외에서는 +82-10-1234-5678로 거세요",
        phone("+82-10-1234-5678", "010-1234-5678"),
        note="국가번호 +82를 떼고 맨 앞 0을 복원한다.",
    ),
    spec(
        "international_compact",
        "WhatsApp +821056781234",
        phone("+821056781234", "010-5678-1234"),
    ),
    spec(
        "representative_number",
        "고객센터 1588-1234 (평일 9시~18시)",
        phone("1588-1234"),
        note="전국 대표번호는 8자리 두 묶음이다. 시간 표기는 잡으면 안 된다.",
    ),
    spec("internet_phone", "사무실 070-4321-8765", phone("070-4321-8765")),
    spec("seoul_four_digit", "본사 02-3456-7890", phone("02-3456-7890")),
    spec(
        "korean_numerals",
        "연락처는 공일공-일이삼사-오육칠팔입니다",
        phone("공일공-일이삼사-오육칠팔", "010-1234-5678"),
        hard=True,
        note="숫자를 한글로 읽어 쓴 표기. 숫자 기반 패턴으로는 잡히지 않는다.",
    ),
]

DATE_SPECS = [
    spec("korean_full", "일시: 2024년 3월 15일", date("2024년 3월 15일", "2024-03-15")),
    spec(
        "slash_and_hyphen",
        "기한: 2024/03/15, 2024-03-15",
        date("2024/03/15", "2024-03-15"),
        date("2024-03-15", "2024-03-15"),
    ),
    spec("dot", "공고일 2024.01.05", date("2024.01.05", "2024-01-05")),
    spec("dot_spaced", "접수 마감은 2024. 1. 5.까지", date("2024. 1. 5.", "2024-01-05")),
    spec("korean_unpadded", "2023년 12월 1일에 가입했어요", date("2023년 12월 1일", "2023-12-01")),
    spec("korean_no_space", "2024년3월15일 오후 2시", date("2024년3월15일", "2024-03-15")),
    spec("english_month_first", "Due date: March 5, 2024", date("March 5, 2024", "2024-03-05")),
    spec("english_day_first", "Event on 15 Jan 2024", date("15 Jan 2024", "2024-01-15")),
    spec(
        "month_day_only",
        "3월 15일에 방문 예정입니다",
        date("3월 15일", "--03-15"),
        note="연도가 없으면 ISO 8601의 --MM-DD 형식으로 둔다.",
    ),
    spec(
        "year_month_only",
        "2024년 5월 정기 점검 안내",
        date("2024년 5월", "2024-05"),
        note="일이 없으면 ISO 8601의 YYYY-MM 형식으로 둔다.",
    ),
    spec(
        "two_digit_year",
        "24.03.15 출고 예정",
        date("24.03.15", "2024-03-15"),
        hard=True,
        note="두 자리 연도는 세기를 추정해야 하고 버전 번호와 헷갈린다.",
    ),
    spec(
        "korean_numerals",
        "이천이십사년 삼월 십오일에 개최",
        date("이천이십사년 삼월 십오일", "2024-03-15"),
        hard=True,
        note="숫자를 한글로 읽어 쓴 표기.",
    ),
]

MONEY_SPECS = [
    spec("comma_won", "가격: 10,000원", krw("10,000원", 10_000)),
    spec("korean_units", "총액 1억 2천만원 예정", krw("1억 2천만원", 120_000_000)),
    spec("dollar_sign", "해외 결제 금액 $100", usd("$100", 100)),
    spec("man_won", "참가비는 5만원입니다", krw("5만원", 50_000)),
    spec("man_cheon_won", "1만 5천원 할인 쿠폰", krw("1만 5천원", 15_000)),
    spec("plain_won", "배송비 3000원 별도", krw("3000원", 3_000)),
    spec("won_symbol", "합계 ₩25,000", krw("₩25,000", 25_000)),
    spec("dollar_word", "구독료는 월 9달러", usd("9달러", 9)),
    spec("usd_code", "USD 1,200 송금 완료", usd("USD 1,200", 1_200)),
    spec(
        "spaced_unit",
        "보증금 500 만원",
        krw("500 만원", 5_000_000),
        note="숫자와 단위 사이 공백.",
    ),
    spec("eok_only", "예산 2억원 책정", krw("2억원", 200_000_000)),
    spec(
        "korean_numerals",
        "삼만원만 내시면 됩니다",
        krw("삼만원", 30_000),
        hard=True,
        note="숫자를 한글로 읽어 쓴 표기. 뒤의 '만'은 보조사다.",
    ),
]

URL_SPECS = [
    spec(
        "full_query",
        "자세한 안내: https://www.example.com/path?query=value",
        url("https://www.example.com/path?query=value"),
    ),
    spec(
        "angle_brackets",
        "링크: <https://www.example.com/path?q=1>",
        url("https://www.example.com/path?q=1"),
    ),
    spec("http_simple", "구 홈페이지 http://example.org", url("http://example.org")),
    spec(
        "www_without_scheme",
        "www.example.co.kr 에서 확인하세요",
        url("www.example.co.kr", "https://www.example.co.kr"),
        note="스킴이 없으면 https://를 보충한다.",
    ),
    spec(
        "port_and_fragment",
        "로컬 서버 http://localhost:8080/api/v1#status",
        url("http://localhost:8080/api/v1#status"),
    ),
    spec(
        "korean_josa",
        "https://shop.example.com/item/123에서 구매",
        url("https://shop.example.com/item/123"),
        note="URL 바로 뒤 한글 조사는 URL이 아니다.",
    ),
    spec(
        "trailing_period",
        "공지는 https://example.com/notice.",
        url("https://example.com/notice"),
    ),
    spec(
        "uppercase_host",
        "HTTPS://WWW.EXAMPLE.COM/Docs",
        url("HTTPS://WWW.EXAMPLE.COM/Docs", "https://www.example.com/Docs"),
        note="스킴·호스트만 소문자로 바꾸고 경로 대소문자는 유지한다.",
    ),
    spec(
        "percent_encoded",
        "검색 결과 https://search.example.com/?q=%ED%95%9C%EA%B8%80&page=2",
        url("https://search.example.com/?q=%ED%95%9C%EA%B8%80&page=2"),
    ),
    spec(
        "two_urls_in_parentheses",
        "(참고: https://a.example.com, https://b.example.com/faq)",
        url("https://a.example.com"),
        url("https://b.example.com/faq"),
    ),
    spec(
        "bare_domain",
        "example.com/event 페이지 참고",
        url("example.com/event", "https://example.com/event"),
        hard=True,
        note="스킴도 www도 없어 파일명·버전 표기와 구분하기 어렵다.",
    ),
    spec(
        "korean_path",
        "https://ko.wikipedia.org/wiki/정규_표현식 문서",
        url("https://ko.wikipedia.org/wiki/정규_표현식"),
        hard=True,
        note="경로에 한글이 있어 ASCII 문자 집합만 허용하면 중간에 잘린다.",
    ),
]

NEGATIVE_SPECS = [
    spec("mention", "트위터 @example_user 계정을 팔로우하세요", note="@ 멘션은 이메일이 아니다."),
    spec("missing_tld", "a@b 형식은 올바른 이메일이 아닙니다", note="최상위 도메인이 없다."),
    spec("version_number", "버전 2.10.3으로 업데이트되었습니다", note="점 구분 숫자지만 날짜가 아니다."),
    spec("invalid_date", "2024-02-30은 존재하지 않는 날짜입니다", note="형식은 맞지만 달력에 없는 날짜."),
    spec("order_number", "주문번호 20240315-0001 확인 바랍니다", note="날짜로 시작하는 주문번호."),
    spec("score_fraction", "점수는 95/100점입니다", note="슬래시가 있지만 날짜가 아니다."),
    spec("ip_address", "IP 192.168.0.1 접속 기록", note="점 구분 숫자 네 묶음."),
    spec("count_with_comma", "참석 인원 100명, 좌석 3,000석", note="쉼표 숫자지만 금액이 아니다."),
    spec("year_only", "2024년은 윤년입니다", note="연도만 있는 표현은 날짜로 보지 않는다."),
    spec("count_like_phone", "1588명이 참여했고 1234건이 접수됐다", note="대표번호처럼 생긴 네 자리 숫자."),
    spec("file_name", "파일명 report.final.pdf를 첨부했어요", note="점이 있는 파일명은 URL이 아니다."),
]

RRN_SPECS = [
    spec("hyphen", "주민등록번호: 900101-1234567", rrn("900101-1234567")),
    spec("no_separator", "주민번호 8505152345678 확인", rrn("8505152345678", "850515-2345678")),
    spec("spaced_hyphen", "주민번호 010203 - 3456789", rrn("010203 - 3456789", "010203-3456789")),
    spec("foreigner", "외국인등록번호 920315-5678901", rrn("920315-5678901")),
    spec(
        "with_phone",
        "본인 확인: 881231-1234567, 010-1111-2222",
        rrn("881231-1234567"),
        phone("010-1111-2222"),
    ),
    spec("invalid_month", "번호 991399-1234567 은 형식 오류", note="13월은 생년월일이 될 수 없다."),
    spec("already_masked", "900101-1****** 로 표시됩니다", note="이미 가려진 번호는 다시 추출하지 않는다."),
    spec("birth_date_only", "생년월일 900101 입력", note="앞자리만 있으면 주민등록번호가 아니다."),
]

CARD_SPECS = [
    spec("hyphen", "카드번호 1234-5678-9012-3456", card("1234-5678-9012-3456")),
    spec("space", "카드 4111 1111 1111 1111 결제", card("4111 1111 1111 1111", "4111-1111-1111-1111")),
    spec("no_separator", "5500000000000004로 결제했습니다", card("5500000000000004", "5500-0000-0000-0004")),
    spec("amex_15", "AMEX 3782-822463-10005", card("3782-822463-10005")),
    spec(
        "with_expiry",
        "카드 1234-5678-9012-3456, 유효기간 12/27",
        card("1234-5678-9012-3456"),
        note="유효기간 12/27은 날짜로 추출하지 않는다.",
    ),
    spec("dot", "결제 카드 1234.5678.9012.3456", card("1234.5678.9012.3456", "1234-5678-9012-3456")),
    spec(
        "with_phone",
        "카드 9410-1234-5678-9012 / 연락처 010-5555-6666",
        card("9410-1234-5678-9012"),
        phone("010-5555-6666"),
        note="카드번호 일부가 전화번호로 오탐되면 안 된다.",
    ),
    spec("already_masked", "1234-****-****-3456 카드로 결제", note="이미 가려진 번호."),
]

ADDRESS_SPECS = [
    spec("road", "배송지: 서울특별시 강남구 테헤란로 123", address("서울특별시 강남구 테헤란로 123")),
    spec("short_province", "서울 마포구 월드컵북로 400 으로 보내주세요", address("서울 마포구 월드컵북로 400")),
    spec(
        "with_detail",
        "경기도 성남시 분당구 판교역로 200, 101동 1203호",
        address("경기도 성남시 분당구 판교역로 200, 101동 1203호"),
        note="동·호수 상세 주소까지 주소로 본다.",
    ),
    spec(
        "lot_number",
        "부산광역시 해운대구 우동 1234-5 소재",
        address("부산광역시 해운대구 우동 1234-5"),
        note="지번 1234-5가 전화번호로 오탐되면 안 된다.",
    ),
    spec("gil", "대구 중구 동성로2길 15 매장", address("대구 중구 동성로2길 15")),
    spec("no_gu", "세종특별자치시 한누리대로 2000 방문", address("세종특별자치시 한누리대로 2000")),
    spec("eup", "전라남도 해남군 해남읍 중앙1로 300", address("전라남도 해남군 해남읍 중앙1로 300")),
    spec(
        "road_only",
        "테헤란로 123으로 와주세요",
        address("테헤란로 123"),
        hard=True,
        note="시·도가 없으면 주소 시작점을 정하기 어렵다.",
    ),
]

SINGLE_PATTERN_SPECS = {
    EMAIL: EMAIL_SPECS,
    PHONE: PHONE_SPECS,
    DATE: DATE_SPECS,
    MONEY: MONEY_SPECS,
    URL: URL_SPECS,
    NEGATIVE: NEGATIVE_SPECS,
}
PII_SPECS = {RRN: RRN_SPECS, CARD: CARD_SPECS, ADDRESS: ADDRESS_SPECS}


# --- 복합 문의 문장 템플릿과 값 목록 (직접 선정) ---

MIXED_TEMPLATES = [
    "문의: {email}, 전화 {phone}\n일시: {date}, 참가비: {money}",
    "{date} 주문 건 환불 요청합니다. 결제 금액 {money}, 회신은 {email} 부탁드려요.",
    "예약 확인 - 방문일 {date}, 연락처 {phone}, 예약금 {money}",
    "안내 페이지 {url} 확인 후 궁금한 점은 {email} 으로 보내주세요.",
    "청구 금액 {money} (청구일 {date}) 상세 내역: {url}",
    "고객님 {phone} 번호로 연락드렸으나 부재중이셨습니다. {date}까지 {url} 에서 확인 바랍니다.",
]

MIXED_VALUES: dict[str, list[Entity]] = {
    "email": [
        email("support@company.co.kr"),
        email("cs.team@example.com"),
        email("order-help@shop.example.kr"),
        email("Info@Example.NET", "info@example.net"),
    ],
    "phone": [
        phone("02-1234-5678"),
        phone("010-2222-3333"),
        phone("031-987-6543"),
        phone("01044445555", "010-4444-5555"),
        phone("1577-0000"),
    ],
    "date": [
        date("2024년 3월 15일", "2024-03-15"),
        date("2024/04/01", "2024-04-01"),
        date("2024-12-25", "2024-12-25"),
        date("2025.1.2", "2025-01-02"),
        date("2024년 7월 7일", "2024-07-07"),
    ],
    "money": [
        krw("50,000원", 50_000),
        krw("3만원", 30_000),
        usd("$25", 25),
        krw("1,250,000원", 1_250_000),
        krw("12만 5천원", 125_000),
    ],
    "url": [
        url("https://www.example.com/refund"),
        url("http://shop.example.kr/order?id=77"),
        url("https://help.example.com/faq#pay"),
    ],
}


def _slot_names(template: str) -> list[str]:
    """템플릿에 나오는 슬롯 이름을 등장 순서대로 반환한다."""
    names = []
    for part in template.split("{")[1:]:
        names.append(part.split("}", 1)[0])
    return names


def build_mixed_cases(count: int = MIXED_CASE_COUNT, seed: int = RANDOM_STATE) -> list[ExtractionCase]:
    """템플릿과 값 목록을 고정 시드로 조합해 여러 정보가 섞인 문의 문장을 만든다."""
    rng = random.Random(seed)
    cases: list[ExtractionCase] = []
    seen_texts: set[str] = set()
    attempts = 0
    while len(cases) < count:
        attempts += 1
        if attempts > count * 50:
            raise RuntimeError("서로 다른 복합 문장을 충분히 만들지 못했습니다.")
        template_index = len(cases) % len(MIXED_TEMPLATES)
        template = MIXED_TEMPLATES[template_index]
        chosen = {name: rng.choice(MIXED_VALUES[name]) for name in _slot_names(template)}
        text = template.format(**{name: entity.raw for name, entity in chosen.items()})
        if text in seen_texts:
            continue
        seen_texts.add(text)
        cases.append(
            ExtractionCase(
                case_id=f"{MIXED}-{len(cases) + 1:02d}",
                category=MIXED,
                variant=f"template_{template_index + 1}",
                text=text,
                entities=tuple(chosen[name] for name in _slot_names(template)),
            )
        )
    return cases


def _cases_from_specs(category: str, specs: list[CaseSpec]) -> list[ExtractionCase]:
    return [
        ExtractionCase(f"{category}-{index:02d}", category, variant, text, entities, hard, note)
        for index, (variant, text, entities, hard, note) in enumerate(specs, start=1)
    ]


def build_cases(include_pii: bool = True) -> list[ExtractionCase]:
    """평가에 쓸 전체 테스트 문장을 만들고 검증한 뒤 반환한다."""
    cases: list[ExtractionCase] = []
    for category, specs in SINGLE_PATTERN_SPECS.items():
        cases.extend(_cases_from_specs(category, specs))
    cases.extend(build_mixed_cases())
    if include_pii:
        for category, specs in PII_SPECS.items():
            cases.extend(_cases_from_specs(category, specs))
    validate_cases(cases)
    return cases


def validate_cases(cases: list[ExtractionCase]) -> None:
    """과제 조건(50문장 이상, 패턴별 10문장·3변형 이상)과 정답 표기 일관성을 확인한다."""
    problems = []
    if len(cases) < MIN_TOTAL_CASES:
        problems.append(f"전체 문장 {len(cases)}개 < {MIN_TOTAL_CASES}개")

    ids = Counter(case.case_id for case in cases)
    texts = Counter(case.text for case in cases)
    problems += [f"중복 ID: {case_id}" for case_id, n in ids.items() if n > 1]
    problems += [f"중복 문장: {text!r}" for text, n in texts.items() if n > 1]

    for case in cases:
        # 같은 raw가 두 번 정답이면 원문에도 두 번 이상 나와야 한다.
        for raw, needed in Counter(entity.raw for entity in case.entities).items():
            if case.text.count(raw) < needed:
                problems.append(f"{case.case_id}: 원문에 없는 정답 표기 {raw!r}")
        if case.category in CORE_TYPES + PII_TYPES:
            if not case.entities and case.variant not in {"invalid_month", "already_masked", "birth_date_only"}:
                problems.append(f"{case.case_id}: 단일 패턴 문장인데 정답이 없음")
        if case.category == NEGATIVE and case.entities:
            problems.append(f"{case.case_id}: 오탐 유도 문장에 정답이 있음")

    for kind in CORE_TYPES:
        own = [case for case in cases if case.category == kind]
        variants = {case.variant for case in own}
        if len(own) < MIN_CASES_PER_PATTERN:
            problems.append(f"{kind}: 문장 {len(own)}개 < {MIN_CASES_PER_PATTERN}개")
        if len(variants) < MIN_VARIANTS_PER_PATTERN:
            problems.append(f"{kind}: 변형 {len(variants)}개 < {MIN_VARIANTS_PER_PATTERN}개")

    if problems:
        raise ValueError("테스트 문장 검증 실패:\n- " + "\n- ".join(problems))


def save_cases(cases: list[ExtractionCase], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def load_cases(path: Path = OUTPUT_PATH) -> list[ExtractionCase]:
    """평가 스크립트에서 저장된 테스트 문장을 읽을 때 쓴다."""
    with path.open(encoding="utf-8") as file:
        return [ExtractionCase.from_dict(json.loads(line)) for line in file if line.strip()]


def summarize(cases: list[ExtractionCase]) -> str:
    """분류별 문장 수, 변형 수, 어려운 사례 수, 정답 유형별 개수를 표로 만든다."""
    lines = [f"{'분류':<10}{'문장':>6}{'변형':>6}{'hard':>6}  정답 개체 수"]
    categories = list(SINGLE_PATTERN_SPECS) + [MIXED] + list(PII_SPECS)
    for category in categories:
        own = [case for case in cases if case.category == category]
        if not own:
            continue
        entity_counts = Counter(entity.kind for case in own for entity in case.entities)
        counts = ", ".join(f"{kind} {n}" for kind, n in sorted(entity_counts.items())) or "-"
        lines.append(
            f"{category:<10}{len(own):>6}{len({c.variant for c in own}):>6}"
            f"{sum(c.hard for c in own):>6}  {counts}"
        )

    totals = Counter(entity.kind for case in cases for entity in case.entities)
    lines.append(f"{'합계':<10}{len(cases):>6}{'':>6}{sum(c.hard for c in cases):>6}  "
                 + ", ".join(f"{kind} {totals[kind]}" for kind in CORE_TYPES + PII_TYPES))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="저장할 JSONL 경로")
    parser.add_argument("--no-pii", action="store_true", help="보너스 개인정보 문장을 제외")
    args = parser.parse_args()

    cases = build_cases(include_pii=not args.no_pii)
    save_cases(cases, args.output)
    print(summarize(cases))
    print(f"\n저장: {args.output} ({len(cases)}문장)")


if __name__ == "__main__":
    main()
