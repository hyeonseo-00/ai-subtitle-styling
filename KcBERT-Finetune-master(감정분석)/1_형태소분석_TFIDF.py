"""
형태소 분석 및 TF-IDF 분석 스크립트

[실행 순서 1] 형태소 분석 및 TF-IDF 분석
CSV 파일을 입력받아 형태소 분석을 수행하고, TF-IDF 벡터화 및 빈도 분석을 수행합니다.
"""
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional

from morphological_analysis import MorphologicalAnalyzer
from tfidf_analysis import TFIDFAnalyzer, FrequencyAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(csv_path: str, 
              text_column: str = "content",
              id_column: Optional[str] = None,
              encoding: str = 'utf-8') -> pd.DataFrame:
    """
    CSV 파일에서 데이터 로드
    
    Args:
        csv_path: CSV 파일 경로
        text_column: 텍스트 컬럼명
        id_column: ID 컬럼명
        encoding: 파일 인코딩
    
    Returns:
        DataFrame
    """
    logger.info(f"데이터 로딩 중: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except UnicodeDecodeError:
        logger.warning("UTF-8 인코딩 실패, cp949 시도 중...")
        df = pd.read_csv(csv_path, encoding='cp949')
    
    if text_column not in df.columns:
        raise ValueError(f"텍스트 컬럼 '{text_column}'을 찾을 수 없습니다.")
    
    # ID 컬럼 처리
    if id_column and id_column in df.columns:
        df = df.rename(columns={id_column: 'id'})
    else:
        df['id'] = df.index
    
    # 결측값 제거
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)
    
    logger.info(f"데이터 로딩 완료: {len(df)}개 문서")
    
    return df


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='형태소 분석 및 TF-IDF 분석 (실행 순서 1)')
    
    parser.add_argument('--input', type=str, required=True,
                       help='입력 CSV 파일 경로')
    parser.add_argument('--text_column', type=str, default='content',
                       help='텍스트 컬럼명 (기본값: content)')
    parser.add_argument('--id_column', type=str, default=None,
                       help='ID 컬럼명 (선택적)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='결과 저장 디렉토리')
    parser.add_argument('--morph_analyzer', type=str, default='kiwi',
                       choices=['kiwi', 'kkma', 'komoran', 'mecab', 'okt'],
                       help='형태소 분석기 타입')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 데이터 로드
    df = load_data(args.input, args.text_column, args.id_column)
    texts = df[args.text_column].tolist()
    
    # 2. 형태소 분석 및 키워드 추출
    logger.info("형태소 분석 시작...")
    morph_analyzer = MorphologicalAnalyzer(analyzer_type=args.morph_analyzer)
    
    morph_results = []
    keywords_list = []
    
    for text in texts:
        keywords = morph_analyzer.extract_keywords(text)
        keywords_list.append(keywords)
        morph_results.append({
            'keywords': ' '.join(keywords),
            'keyword_count': len(keywords)
        })
    
    morph_df = pd.DataFrame(morph_results)
    morph_df.to_csv(output_dir / "형태소분석_결과.csv",
                   index=False, encoding='utf-8-sig')
    logger.info("형태소 분석 완료")
    
    # 3. TF-IDF 벡터화
    logger.info("TF-IDF 벡터화 시작...")
    tfidf_analyzer = TFIDFAnalyzer()
    tokenized_texts = [' '.join(keywords) for keywords in keywords_list]
    tfidf_matrix = tfidf_analyzer.fit_transform(tokenized_texts)
    
    # 상위 특성 추출
    top_features = tfidf_analyzer.get_top_features(n=50)
    top_features_df = pd.DataFrame(
        top_features,
        columns=['feature', 'tfidf_score']
    )
    top_features_df.to_csv(output_dir / "TFIDF_상위특성.csv",
                          index=False, encoding='utf-8-sig')
    
    # 4. 단어 빈도 분석
    logger.info("단어 빈도 분석 시작...")
    freq_analyzer = FrequencyAnalyzer()
    all_tokens = [token for keywords in keywords_list for token in keywords]
    word_freq = freq_analyzer.word_frequency(all_tokens, top_n=50)
    word_freq_df = pd.DataFrame(
        word_freq,
        columns=['word', 'frequency']
    )
    word_freq_df.to_csv(output_dir / "단어빈도.csv",
                       index=False, encoding='utf-8-sig')
    
    # TF-IDF 행렬 저장 (다음 단계에서 사용)
    import pickle
    with open(output_dir / "tfidf_matrix.pkl", 'wb') as f:
        pickle.dump({
            'tfidf_matrix': tfidf_matrix,
            'feature_names': tfidf_analyzer.feature_names,
            'keywords_list': keywords_list
        }, f)
    
    logger.info(f"TF-IDF 행렬 생성 완료: {tfidf_matrix.shape}")
    logger.info(f"결과 저장 완료: {output_dir}")
    
    print(f"\n형태소 분석 및 TF-IDF 분석 완료!")
    print(f"저장 위치: {output_dir}")
    print(f"- 형태소분석_결과.csv")
    print(f"- TFIDF_상위특성.csv")
    print(f"- 단어빈도.csv")
    print(f"- tfidf_matrix.pkl (다음 단계에서 사용)")


if __name__ == '__main__':
    main()

