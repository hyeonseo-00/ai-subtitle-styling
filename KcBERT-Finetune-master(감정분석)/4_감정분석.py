"""
감정분석 스크립트
KcELECTRA 모델을 사용한 감정분석

[실행 순서 4] 감정분석
CSV 파일을 입력받아 KcELECTRA 모델로 감정분석을 수행합니다.
"""
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Optional

from sentiment_analysis import SentimentAnalyzer

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
    parser = argparse.ArgumentParser(description='감정분석 (실행 순서 4)')
    
    parser.add_argument('--input', type=str, required=True,
                       help='입력 CSV 파일 경로')
    parser.add_argument('--text_column', type=str, default='content',
                       help='텍스트 컬럼명 (기본값: content)')
    parser.add_argument('--id_column', type=str, default=None,
                       help='ID 컬럼명 (선택적)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='결과 저장 디렉토리')
    parser.add_argument('--model', type=str, 
                       default='beomi/KcELECTRA-base-v2022',
                       help='KcELECTRA 모델 이름')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 데이터 로드
    df = load_data(args.input, args.text_column, args.id_column)
    texts = df[args.text_column].tolist()
    
    # 2. 감정분석 모델 초기화
    logger.info(f"감정분석 모델 로딩 중: {args.model}")
    sentiment_analyzer = SentimentAnalyzer(model_name=args.model)
    
    # 3. 감정분석 수행
    logger.info(f"감정분석 시작: {len(texts)}개 문서")
    
    results = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        batch_results = sentiment_analyzer.predict_batch(
            batch, return_probs=True
        )
        results.extend(batch_results)
    
    # 결과 정리
    sentiment_df = pd.DataFrame([
        {
            'id': df.iloc[i]['id'],
            'content': texts[i],
            'sentiment_label': label,
            'sentiment_confidence': max(probs.values()),
            **probs
        }
        for i, (label, probs) in enumerate(results)
    ])
    
    # 4. 결과 저장
    sentiment_df.to_csv(output_dir / "감정분석_결과.csv",
                       index=False, encoding='utf-8-sig')
    
    logger.info("감정분석 완료")
    
    # 통계 출력
    print(f"\n감정분석 완료!")
    print(f"저장 위치: {output_dir / '감정분석_결과.csv'}")
    print(f"\n감정 분포:")
    print(sentiment_df['sentiment_label'].value_counts())


if __name__ == '__main__':
    main()

