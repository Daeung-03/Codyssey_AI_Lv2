"""고객 문의 통합 파이프라인: 정보 추출 → 감성 분석 (선택: 개인정보 마스킹, 측면별 감성).

예)
python main.py --text '문의: support@company.co.kr, 전화 02-1234-5678 일시: 2024년 3월 15일, 참가비: 50,000원'
python main.py --text '음식은 맛있는데 가격이 비싸요' --aspect
python main.py --text '본인 확인 881231-1234567, 연락처 010-1111-2222' --mask
"""

import argparse

from entities import CORE_TYPES, ENTITY_TYPES
from extractor import extract, format_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="정규식 정보 추출 + 감성 사전 감성 분석")
    parser.add_argument("--text", required=True, help="분석할 문장")
    parser.add_argument("--mask", action="store_true", help="[보너스] 개인정보 추출·마스킹 결과도 출력")
    parser.add_argument("--aspect", action="store_true", help="[보너스] 측면별 감성도 출력")
    args = parser.parse_args()

    # 감성 분석 모듈은 KoNLPy(Java)를 띄우므로 필요할 때 불러온다.
    from sentiment import SentimentAnalyzer, format_trace

    kinds = ENTITY_TYPES if args.mask else CORE_TYPES
    sections = [format_extraction(args.text, extract(args.text, kinds))]
    if args.mask:
        from masking import format_masking

        sections.append(format_masking(args.text))
    analyzer = SentimentAnalyzer()
    sections.append(format_trace(analyzer.analyze(args.text)))
    if args.aspect:
        from aspect import AspectAnalyzer, format_aspects

        aspect_analyzer = AspectAnalyzer(analyzer.lexicon, analyzer.tagger)
        sections.append(format_aspects(args.text, aspect_analyzer.analyze(args.text)))
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
