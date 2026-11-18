# Q4_2 실행 계획: 규칙 기반 정보 추출 + 감성 분석

[PROBLEM.md](PROBLEM.md)의 필수 요구사항과 보너스 3개(개인정보 마스킹, 규칙 vs 통계 비교, Aspect 감성 분석)를 모두 구현한다.
Q4_1과 같은 방식으로 **평평한 모듈 구조 + 모듈별 CLI + `docs/`에 로그·리포트**를 남긴다.

## 1. 전체 흐름

```text
                     ┌──────────── 정보 추출 ────────────┐
고객 문의 텍스트 ──▶ │ patterns.py → extractor.py (정규화)│──▶ masking.py (보너스: 개인정보 가림)
                     └────────────────────────────────────┘
        │
        └──────────▶ ┌──────────── 감성 분석 ────────────┐
                     │ lexicon.py → sentiment.py          │──▶ aspect.py (보너스: 측면별 감성)
                     │ (토큰화·사전 점수·부정어·강조어)   │
                     └────────────────────────────────────┘
main.py: 두 흐름을 묶은 통합 파이프라인 CLI (`python main.py --text '...'`)

평가: extraction_testset.py ─▶ evaluate_extraction.py (패턴별 P/R, 실패 사례)
      nsmc_data.py ──────────▶ evaluate_sentiment.py  (Accuracy/F1, 규칙 전후 비교, 오분류)
                               compare_tfidf.py       (보너스: 규칙 vs TF-IDF, 10/100/1000개)
```

## 2. 디렉터리 구조

```text
Q4_2/
├── README.md                  # 최종 리포트 (권장 목차 7개 절)
├── requirements.txt
├── main.py                    # 통합 CLI
├── entities.py                # 공용 자료형 Entity
├── patterns.py                # 정규표현식 정의 + 주석
├── extractor.py               # 추출·정규화·겹침 해소
├── masking.py                 # [보너스] 개인정보 마스킹
├── lexicon.py                 # 감성 사전·부정어·강조어 로딩
├── sentiment.py               # 감성 점수 계산기
├── aspect.py                  # [보너스] Aspect 감성 분석
├── nsmc_data.py               # NSMC 다운로드·로딩·표본 추출
├── extraction_testset.py      # 정보 추출 테스트 문장 생성기
├── evaluate_extraction.py
├── evaluate_sentiment.py
├── compare_tfidf.py           # [보너스] 규칙 vs TF-IDF
├── data/
│   ├── extraction_testset.jsonl   # 생성기 출력 (Git 포함)
│   ├── extraction_holdout.jsonl   # 패턴을 보지 않고 작성한 평가 문장 (Git 포함)
│   ├── pii_holdout.jsonl          # [보너스] 개인정보 홀드아웃 (Git 포함)
│   ├── sentiment_lexicon.tsv      # 범용 감성 사전 (Git 포함)
│   ├── domain_lexicon.tsv         # 도메인 특화 단어 30개+ (Git 포함)
│   ├── aspect_testset.jsonl       # [보너스] Aspect 정답 (Git 포함)
│   └── ratings_{train,test}.txt   # NSMC 원본 (Git 제외)
└── docs/
    ├── PROBLEM.md · PLAN.md · REPORT.md
    └── *.log, *.svg               # 평가 로그·그래프
```

## 3. 모듈별 설계

### 3.1 `entities.py` — 공용 자료형 ✅

| 항목 | 내용 |
| --- | --- |
| 책임 | 추출 예측과 정답을 같은 `Entity(kind, raw, normalized, unit, span)`로 표현 |
| 상수 | `CORE_TYPES = (email, phone, date, money, url)`, `PII_TYPES = (rrn, card, address)` |
| 핵심 | `Entity.key() = (kind, normalized, unit)` — 표기가 달라도 정규화 값이 같으면 정답으로 인정 |
| 검증 | 금액은 `int` + 통화(KRW/USD), 나머지는 `str`만 허용. `span`은 비교에서 제외 |

### 3.2 `extraction_testset.py` — 테스트 문장 생성기 ✅

| 항목 | 내용 |
| --- | --- |
| 출력 | `data/extraction_testset.jsonl` — `{id, category, variant, text, entities[], hard, note}` |
| 구성 | 단일 패턴 5종 × 12문장, 오탐 유도 11문장, 복합 문의 20문장(템플릿 6종 × 고정 시드 42), 개인정보 24문장 = **115문장** |
| 원칙 | 정답 정규화 값은 추출기를 호출하지 않고 직접 기입. `hard=True`로 규칙이 놓칠 것으로 예상한 사례(한글 숫자, 난독화, 두 자리 연도 등 9건)를 표시해 실패 사례 분석에 사용 |
| 검증 | `validate_cases()`: 전체 50문장+, 패턴별 10문장·3변형+, 정답 raw가 원문에 존재, ID·문장 중복 없음, negative에는 정답 없음 |
| API | `build_cases(include_pii)`, `save_cases()`, `load_cases()`, `summarize()` |
| 실행 | `python extraction_testset.py [--output PATH] [--no-pii]` |

### 3.3 `patterns.py` — 정규표현식

- 패턴마다 `re.compile(..., re.VERBOSE)`로 작성하고, 줄 단위 주석으로 **의도·매칭 범위·의도적으로 제외한 것**을 적는다.
- 경계 처리: `\b`는 한글 조사와 붙은 경우(`...co.kr입니다`)에 동작하지 않으므로 `(?<![\w.])`, `(?![\w@])` 같은 lookaround를 쓰고 ASCII 범위를 명시한다.
- 그룹 캡처는 이름 있는 그룹(`(?P<year>...)`)으로 정규화 단계에 넘긴다.

| 패턴 | 다룰 변형 (최소 3개 이상) |
| --- | --- |
| 이메일 | 기본, 서브도메인·국가 도메인, `+`태그, `_`·숫자, 대문자, 꺾쇠·괄호·문장부호 경계 |
| 전화 | 휴대폰/서울/지역/070, 하이픈·점·공백·무구분, `(02)`, `+82`, 15xx 대표번호 |
| 날짜 | `YYYY년 M월 D일`, `/`·`-`·`.` 구분, 영문 월 이름(앞/뒤), 연도 없는 `M월 D일`, `YYYY년 M월` |
| 금액 | 쉼표 숫자+원, 억·만·천 단위 조합, `$`·`달러`·`USD`, `₩`, 숫자-단위 사이 공백 |
| URL | http/https, 스킴 없는 `www.`, 포트·프래그먼트·쿼리, 꺾쇠, 끝 문장부호·조사 제외 |
| [보너스] 주민번호 | 하이픈/무구분/공백 하이픈, 외국인(뒷자리 5~8), 월·일 범위 검사 |
| [보너스] 카드 | 4-4-4-4 (하이픈/공백/점/무구분), AMEX 4-6-5, 가려진 번호 제외 |
| [보너스] 주소 | 시·도(약칭 포함) + 시·군·구 + 읍·면·동/도로명(로·길) + 건물번호 + 상세(동·호) |

### 3.4 `extractor.py` — 추출·정규화

```python
extract(text: str, kinds: Iterable[str] = CORE_TYPES) -> list[Entity]   # span 순 정렬
normalize_email / normalize_phone / normalize_date / normalize_money / normalize_url
```

- **겹침 해소**: 모든 패턴의 후보를 모은 뒤 `긴 매칭 우선` → 길이가 같으면 `우선순위(url > email > rrn > card > phone > date > money > address)` 순으로 겹치는 후보를 버린다. (URL 안의 날짜, 카드번호 안의 전화번호 오탐 방지)
- **검증 후 탈락**: 정규식으로 잡은 후 `datetime.date()`로 날짜 유효성을, 전화 자릿수를 다시 확인해 정밀도를 올린다.

#### 정규화 규격 (생성기 정답과 반드시 동일)

| 유형 | 규칙 | 예시 |
| --- | --- | --- |
| email | 전체 소문자 | `John.Doe@Example.COM` → `john.doe@example.com` |
| phone | `+82` → `0`, 하이픈 3묶음(대표번호는 2묶음) | `+821056781234` → `010-5678-1234` |
| date | ISO 8601: `YYYY-MM-DD`, 일 없음 `YYYY-MM`, 연도 없음 `--MM-DD` | `2024. 1. 5.` → `2024-01-05` |
| money | `normalized=int`, `unit=KRW/USD` | `1억 2천만원` → `120000000 KRW` |
| url | 스킴·호스트 소문자, 스킴 없으면 `https://` 보충, 경로 유지, 끝 `.,)>` 제거 | `www.example.co.kr` → `https://www.example.co.kr` |
| rrn | `YYMMDD-NNNNNNN` | `8505152345678` → `850515-2345678` |
| card | 16자리 4-4-4-4, AMEX 4-6-5 | `4111 1111 1111 1111` → `4111-1111-1111-1111` |
| address | 연속 공백을 한 칸으로 | — |

### 3.5 `masking.py` — [보너스] 개인정보 마스킹

```python
mask(text: str, kinds=PII_TYPES + (PHONE, EMAIL)) -> tuple[str, list[Entity]]
```

| 유형 | 마스킹 규칙 |
| --- | --- |
| phone | 가운데 묶음 가림 `010-****-5678` |
| rrn | 뒷자리 첫 숫자(성별)만 남김 `900101-1******` |
| card | 앞 4·뒤 4만 남김 `1234-****-****-3456` |
| email | 로컬 첫 글자만 남김 `u***@domain.com` |
| address | 시·도와 시·군·구까지만 남김 `서울특별시 강남구 ***` |

뒤 위치부터 치환해 span이 밀리지 않게 한다. 평가는 "개인정보 정답 span에 원래 숫자가 남아 있지 않은가"로 확인한다.

### 3.6 `lexicon.py` + `data/*.tsv` — 감성 사전

| 파일 | 형식 | 규모 |
| --- | --- | --- |
| `sentiment_lexicon.tsv` | `word \t score(-2~+2) \t source` | 범용 감성어 200개+ |
| `domain_lexicon.tsv` | 같은 형식 | 영화 리뷰 도메인어 30개+ (꿀잼, 노잼, 졸작, 명작, 발연기, 개연성, 킬링타임 …) |

- **표제어는 Okt `stem=True` 결과 형태**(예: `맛있다`, `지루하다`)로 맞춰 토큰과 바로 대조되게 한다.
- 출처: 직접 구축을 기본으로 하고, 외부 사전(KNU 한국어 감성사전 등)을 참고하면 `source` 열과 README에 출처를 명시한다.
- 누수 방지: 사전 후보는 **NSMC train에서만** 빈도·레이블 비율로 뽑고 사람이 검수한다. 평가용 test는 보지 않는다.
- `NEGATORS`(안, 못, 않다, 없다, 아니다), `INTENSIFIERS`(정말·진짜·너무·매우·완전 ×1.5, 조금·좀 ×0.5)도 이 모듈에 둔다.

### 3.7 `sentiment.py` — 감성 점수 계산

```python
@dataclass(frozen=True)
class RuleConfig: negation: bool = True; intensifier: bool = True; double_negation: bool = True

class SentimentAnalyzer:
    def __init__(self, lexicon, rules: RuleConfig = RuleConfig()) -> None
    def analyze(self, text: str) -> SentimentResult   # score, label, tokens, trace
def format_trace(result) -> str                       # PROBLEM.md 예시 형식 출력
```

처리 순서 (필수 범위):
1. **토큰화**: Q4_1과 같이 Okt `pos(norm=True, stem=True)`
2. **사전 점수**: 감성어 토큰마다 기본 점수
3. **강조어**: 감성어 앞 2토큰 안의 강조어 배율을 곱함
4. **부정어 반전**: 뒤 부정어 `않다/없다/아니다/못하다`는 앞 2토큰 안의 감성어를, 앞 부정어 `안/못`은 바로 다음 말(또는 `명사 + 못 하다`의 명사)을 ×(-1). `별로·어이` 같은 부정 결합어는 반전하지 않음
5. **이중부정 단일 패턴**: `-지 않을 수 없다`, `-지 않은 것은 아니다` 형태를 하나의 패턴으로 인식 → 반전 두 번(원 극성 유지)
6. **판정**: `score > 0` 긍정, `< 0` 부정, `= 0`(사전 미적중)은 train 다수 레이블로 두고 비율을 따로 보고

`trace`에 단계별 탐지 내역을 남겨 해석 가능성을 보여준다.

### 3.8 `aspect.py` — [보너스] Aspect 감성 분석

- `ASPECT_KEYWORDS`: 음식(맛, 음식, 메뉴), 가격(가격, 가성비, 비싸), 서비스(직원, 서비스, 친절), 배송, 분위기 + 영화(연기, 스토리, 영상미, 음악)
- 절 분리: `는데 / 지만 / 근데 / 그런데 / 하지만 / ,` 기준
- 각 절에서 측면 키워드를 찾고 그 절만 `SentimentAnalyzer`로 점수화 → `{aspect: label}`
- 평가: 직접 만든 `data/aspect_testset.jsonl` 30문장+ 의 측면-극성 정확도

### 3.9 `nsmc_data.py` — 감성 데이터

- `ratings_train.txt`, `ratings_test.txt`를 e9t/nsmc에서 내려받는다(`--download`).
- 빈 리뷰·원문 중복·레이블 충돌 제거 후, **test에서 긍·부정 균형 1,000개**(필수 100개 이상 충족)를 고정 시드로 평가셋으로 쓴다.
- train은 사전 후보 탐색, 0점 처리 기준, 보너스 TF-IDF 학습에만 쓴다.

### 3.10 평가 모듈

| 모듈 | 산출물 | 내용 |
| --- | --- | --- |
| `evaluate_extraction.py` | `docs/extraction_eval.log` | 문장별 정답·예측 `key()` 다중집합 비교 → 패턴별 TP/FP/FN, Precision/Recall/F1. 실패 사례를 유형(표기 미지원, 경계 오류, 오탐, 정규화 오류, 겹침)별로 5건 이상 분류. 보너스 PII 점수와 마스킹 누락 확인 포함 |
| `evaluate_sentiment.py` | `docs/sentiment_eval.log` | 설정 5가지(사전만 / +부정어 / +강조어 / +둘 다 / +이중부정)의 Accuracy·macro F1 비교, 오분류 10건 이상을 원인 유형(사전 미등록, 반어, 복합 감성, 부정 범위 오류, 토큰화 오류)별 분석 |
| `compare_tfidf.py` | `docs/rule_vs_tfidf.log`, `.svg` | 같은 test 1,000개에서 규칙 기반 vs TF-IDF+LogisticRegression. 학습 크기 10/100/1000(긍·부정 균형), 작은 표본의 편차를 보려고 시드 5개 평균·표준편차 |

### 3.11 `main.py` — 통합 CLI

```bash
python main.py --text '문의: support@company.co.kr, 전화 02-1234-5678 ...'   # 추출 + 감성
python main.py --text '...' --mask      # 개인정보 마스킹 결과 함께 출력
python main.py --text '...' --aspect    # 측면별 감성 함께 출력
```

출력 형식은 PROBLEM.md 8장 예시(`=== 정보 추출 결과 ===`, `=== 감성 분석 과정 ===`)를 따른다.

## 4. 작업 단계

| # | 작업 | 산출물 | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | 아키텍처 설계 + 테스트 문장 생성기 | `PLAN.md`, `entities.py`, `extraction_testset.py`, `data/extraction_testset.jsonl` | 115문장 생성, 검증 통과 | ✅ |
| 2 | 정규식 패턴 + 추출·정규화 | `patterns.py`, `extractor.py` | 필수 5패턴, 패턴별 3변형+, 모든 패턴에 주석 | ✅ |
| 3 | 정보 추출 평가 | `evaluate_extraction.py`, `data/extraction_holdout.jsonl`, `docs/extraction_eval.log` | 패턴별 P/R, 실패 사례 5건+ 유형 분류 | ✅ |
| 4 | NSMC 데이터 준비 + 감성 사전 구축 | `nsmc_data.py`, `lexicon.py`, `data/*_lexicon.tsv` | 사전 200개+, 도메인어 30개+, 출처 명시 | ✅ |
| 5 | 감성 분석기 | `sentiment.py` | 부정어·강조어·이중부정 단일 패턴, trace 출력 | ✅ |
| 6 | 감성 분석 평가 | `evaluate_sentiment.py`, `docs/sentiment_eval.log` | Acc/F1, 규칙 전후 비교, 오분류 10건+ 분석 | ✅ |
| 7 | [보너스] 개인정보 마스킹 | `patterns.py` 확장, `masking.py`, `data/pii_holdout.jsonl` | 주민번호·카드·주소 추출 + 마스킹, PII 문장 평가 | ✅ |
| 8 | [보너스] 규칙 vs TF-IDF | `compare_tfidf.py`, 로그·그래프 | 10/100/1000 학습 크기별 비교 | ⬜ |
| 9 | [보너스] Aspect 감성 분석 | `aspect.py`, `data/aspect_testset.jsonl` | 측면별 극성 출력 + 정확도 | ⬜ |
| 10 | 통합 CLI + 문서화 | `main.py`, `README.md`, `requirements.txt` | README 권장 목차 7절, 설치·실행 방법, 통계 vs 규칙 비교 서술 | ⬜ |

커밋은 Q4_1처럼 단계마다 `Q4_2: <작업 내용>` 형식으로 남긴다.

## 5. 환경

- Python 3.11 (`Q4_2/.venv`), 의존성: `konlpy`(Okt, Java 필요), `numpy`, `scikit-learn`, `matplotlib` — Q4_1과 같은 버전
- 정규표현식은 표준 `re`만 사용 (제약사항)
- 1~3단계는 표준 라이브러리만으로 동작한다.

## 6. 미리 정해 둘 판단과 위험

| 항목 | 결정 / 대응 |
| --- | --- |
| 정답 기준 | 생성기 정답은 "사람이 정보라고 판단하는 것"이다. 규칙이 못 잡는 hard 사례도 정답에 넣어 재현율 한계를 수치로 드러낸다 |
| 날짜 부분 정보 | 연도·일이 없는 날짜도 정보로 보고 ISO 8601 축약형으로 정규화한다 |
| 감성 0점 | 사전 미적중 비율이 높을 수 있다. 0점 비율을 별도로 보고하고, 사전 확장 전후 커버리지를 함께 기록한다 |
| Okt 원형화 | `않았어요 → 않다`처럼 원형이 바뀌므로 사전·부정어 목록을 원형 기준으로 관리한다 |
| 데이터 누수 | 사전 구축·임계값 결정은 train에서만, 최종 수치는 test 평가셋에서만 낸다 |
