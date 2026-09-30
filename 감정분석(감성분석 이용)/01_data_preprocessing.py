"""
01_data_preprocessing.py
데이터 전처리 및 형태소 분리

기능:
- 특수문자 및 숫자 제거
- 의미없는 짧은 글 제거
- KoNLPy의 Okt를 사용한 형태소 분리
- 불용어 제거
"""

import pandas as pd
import re
from tqdm import tqdm
from konlpy.tag import Okt

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv("한글 크롤링.csv")
print(f"원본 데이터 개수: {len(df)}")

# ============================================
# 1. 데이터 전처리하기
# ============================================

# 1.1. 특수문자 및 숫자 제거
print("\n1.1 특수문자 및 숫자 제거 중...")

def re_pattern(string):
    """정규표현식을 사용하여 특수문자 및 숫자 제거"""
    pattern = re.compile(r'[^a-zA-Z가-힣\s\.\?\!]')
    string = re.sub(pattern, ' ', string)
    
    pattern2 = re.compile(r'\s+')
    result = re.sub(pattern2, ' ', string)
    return result

df['re_review'] = df['Review'].apply(lambda x: re_pattern(x))

# 1.2. 의미없는 짧은 글 제거
print("1.2 짧은 글 제거 중...")
df = df[df['Review'].apply(lambda x: len(x) > 15)]
df = df.reset_index(drop=True)
print(f"전처리 후 데이터 개수: {len(df)}")

# ============================================
# 2. 데이터 형태소 분리하기
# ============================================

# 2.1. 불용어 적용 및 형태소 분리
print("\n2.1 형태소 분리 준비 중...")

# 불용어 파일 로드
stopwords_df = pd.read_csv("ko-stopwords.csv")
stopwords = set(stopwords_df['stopwords'])

# Okt 초기화
okt = Okt()

def okt_pos_tagging(string):
    """형태소 분리 및 불용어 제거"""
    pos_words = okt.pos(string, stem=True, norm=True)
    result = [word for word, tag in pos_words 
              if word not in stopwords 
              if tag in {'Noun', 'Adjective', 'Verb'}]
    return result

# 2.2. 데이터 프레임에 추가
print("2.2 형태소 분리 실행 중...")
tqdm.pandas()
df['tagged_review'] = df['re_review'].progress_apply(lambda x: okt_pos_tagging(x))

print(f"\n형태소 분리 완료!")
print(f"처리된 데이터 개수: {len(df)}")
print(f"\n샘플 데이터:")
print(df[['Review', 'tagged_review']].head())

# 결과 저장
df.to_csv('preprocessed_data.csv', encoding='utf-8-sig', index=False)
print("\n전처리된 데이터가 'preprocessed_data.csv'에 저장되었습니다.")

