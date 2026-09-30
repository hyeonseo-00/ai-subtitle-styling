import pandas as pd  # pyright: ignore[reportMissingImports]
import time
from tqdm import tqdm
 
# deep_translator가 없으면 googletrans 사용 시도, 둘 다 없으면 번역 기능 비활성화
try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='en', target='ko')
    USE_DEEP_TRANSLATOR = True
except ImportError:
    try:
        from googletrans import Translator
        translator = Translator()
        USE_DEEP_TRANSLATOR = False
        print("deep_translator를 찾을 수 없어 googletrans를 사용합니다.")
    except ImportError:
        translator = None
        print("⚠️ 번역 라이브러리를 찾을 수 없습니다. 번역 기능을 사용할 수 없습니다.")
        print("설치 명령: pip install deep-translator 또는 pip install googletrans==4.0.0rc1")
 
# CSV 파일 읽기
df = pd.read_csv('./data/reddit_deaf_misheard.csv')
 
print(f"원본 데이터: {len(df)}건")
print(f"컬럼: {df.columns.tolist()}\n")
 
if translator is None:
    print("번역 기능을 사용할 수 없어 번역 없이 원본만 저장합니다.")
    df.to_csv('reddit_hardofhearing_subreddit_7시간_한글화.csv', index=False, encoding='utf-8-sig')
    exit(0)
 
# 번역할 컬럼들
translation_columns = ['title', 'content']  # 번역할 컬럼명
 
# 각 번역 컬럼에 대해 번역본 컬럼 추가
for col in translation_columns:
    if col not in df.columns:
        print(f"경고: '{col}' 컬럼이 없습니다. 건너뜁니다.")
        continue
   
    translated_col = f"{col}_한글"
    print(f"'{col}' 컬럼 번역 중...")
   
    translated_values = []
   
    for idx, value in enumerate(tqdm(df[col], desc=f"{col} 번역 중", unit="건")):
        if pd.isna(value) or str(value).strip() == "":
            translated_values.append("")
        else:
            try:
                # 텍스트가 너무 길면 잘라서 번역 (5000자 단위)
                text = str(value)
                if len(text) > 5000:
                    # 긴 텍스트는 분할해서 번역 후 합치기
                    chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                    translated_chunks = []
                    for chunk in chunks:
                        if USE_DEEP_TRANSLATOR:
                            translated_chunk = translator.translate(chunk)
                        else:
                            # googletrans 사용
                            translated_chunk = translator.translate(chunk, dest='ko').text
                        translated_chunks.append(translated_chunk)
                        time.sleep(0.5)  # API 레이트 리밋 방지
                    translated_text = " ".join(translated_chunks)
                else:
                    if USE_DEEP_TRANSLATOR:
                        translated_text = translator.translate(text)
                    else:
                        # googletrans 사용
                        translated_text = translator.translate(text, dest='ko').text
               
                translated_values.append(translated_text)
            except Exception as e:
                print(f"\n  [{idx}] 번역 실패: {str(e)[:100]}")
                translated_values.append("")  # 번역 실패 시 빈 문자열
           
            # API 레이트 리밋 방지
            time.sleep(0.2)
   
    # 번역된 컬럼 추가 (원본 컬럼 바로 옆에)
    col_idx = df.columns.get_loc(col)
    df.insert(col_idx + 1, translated_col, translated_values)
    print(f"  완료: '{translated_col}' 컬럼 추가됨\n")
 
# 결과 저장
output_file = 'reddit_deaf_misheard_한글화.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"번역 완료! 저장 파일: {output_file}")
print(f"최종 데이터: {len(df)}건")
print(f"최종 컬럼: {df.columns.tolist()}")