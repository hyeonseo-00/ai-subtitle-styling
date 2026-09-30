# translate_fill_csv_gemini.py

import os
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai  # pyright: ignore[reportMissingImports]


# =========================
# 0) 환경설정
# =========================
load_dotenv()

# 환경 변수에서 API 키 가져오기 (환경 변수 이름: GOOGLE_API_KEY)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 환경 변수가 없으면 직접 API 키 입력 (보안상 .env 파일 사용 권장)
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = ""

if not GOOGLE_API_KEY:
    raise ValueError("Google API 키가 설정되지 않았습니다. 환경 변수 GOOGLE_API_KEY를 설정하거나 코드에 직접 입력하세요.")

genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 선택 (속도 빠른 버전)
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)


# =========================
# 1) 번역 함수 (영어 → 한국어)
# =========================
def translate_with_gemini(text: str) -> str:
    """
    Google Gemini를 사용해 영어 텍스트를 한국어로 번역.
    - Reddit 스타일의 긴 글도 그대로 번역
    - 요약 금지, 전체 번역
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    prompt = f"""
You are a professional English-to-Korean translator.

- Input is a Reddit-style English post about hearing loss, deafness, or mishearing.
- Translate into NATURAL and fluent Korean.
- Do NOT summarize or shorten. Translate EVERYTHING.
- Preserve paragraph breaks and list formatting as much as possible.
- Do NOT add explanations. Output ONLY the Korean translation.

[ENGLISH ORIGINAL]
{text}
"""

    response = model.generate_content(prompt)
    ko = (response.text or "").strip()
    return ko


# =========================
# 2) content_한글 채우기 메인 로직
# =========================
def fill_korean_column(
    input_csv_path: str,
    output_csv_path: str,
    content_col: str = "content",
    ko_col: str = "content_한글",
    max_rows: int | None = None,
    sleep_sec: float = 1.0,
):
    """
    - input_csv_path: 원본 CSV 경로
    - output_csv_path: 번역 완료 CSV 저장 경로
    - content_col: 영어 원문 컬럼명
    - ko_col: 한국어 번역 컬럼명
    - max_rows: 테스트용으로 일부만 돌리고 싶을 때 개수 지정 (None이면 전체)
    - sleep_sec: API 호출 사이 텀 (요금·Rate limit 완화용)
    """

    print(f"CSV 불러오는 중: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # content_한글 이 비어 있는 행만 대상으로
    mask_empty_ko = df[ko_col].isna() | (df[ko_col].astype(str).str.strip() == "")
    target_idx = df[mask_empty_ko].index

    if max_rows is not None:
        target_idx = target_idx[:max_rows]

    print(f"번역 대상 행 수: {len(target_idx)}")

    for i, idx in enumerate(target_idx, start=1):
        en_text = str(df.at[idx, content_col])

        # 디버깅/안심용: 원문 길이 출력 (셀 전체 기준)
        print(f"\n[{i}/{len(target_idx)}] index={idx}")
        print(f"  영어 길이: {len(en_text)}자")

        try:
            ko_text = translate_with_gemini(en_text)
            df.at[idx, ko_col] = ko_text
            print("  ✅ 번역 완료")
        except Exception as e:
            print(f"  ❌ 번역 실패: {e}")
            # 필요하다면 실패 표시 남기기
            # df.at[idx, ko_col] = f"[번역 실패] {e}"

        time.sleep(sleep_sec)

    # 최종 CSV 저장
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_csv_path}")


# =========================
# 3) 직접 실행 구간
# =========================
if __name__ == "__main__":
    # ⚠ 파일 이름/경로는 네 환경에 맞게 바꿔줘
    input_path = "한글화.csv"
    output_path = "reddit_한글화_번역완료.csv"

    # 먼저 2~3개만 테스트해보고 전체 돌리는 걸 추천
    fill_korean_column(
        input_csv_path=input_path,
        output_csv_path=output_path,
        content_col="content",
        ko_col="content_한글",
        max_rows=None,       # 전체 돌릴 땐 None으로 바꾸기
        sleep_sec=0.2,
    )

