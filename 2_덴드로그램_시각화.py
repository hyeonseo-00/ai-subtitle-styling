"""
덴드로그램 시각화 스크립트
계층적 클러스터링 결과를 덴드로그램으로 시각화

[실행 순서 2] 덴드로그램 생성
TF-IDF 행렬을 사용하여 계층적 클러스터링 덴드로그램을 생성합니다.
"""
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional

from dendrogram import DendrogramVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='덴드로그램 시각화 (실행 순서 2)')
    
    parser.add_argument('--input', type=str, required=True,
                       help='입력 CSV 파일 경로 (원본 데이터)')
    parser.add_argument('--tfidf_pkl', type=str, default='output/tfidf_matrix.pkl',
                       help='TF-IDF 행렬 pickle 파일 경로 (1단계에서 생성)')
    parser.add_argument('--text_column', type=str, default='content',
                       help='텍스트 컬럼명 (기본값: content)')
    parser.add_argument('--id_column', type=str, default=None,
                       help='ID 컬럼명 (선택적)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='결과 저장 디렉토리')
    parser.add_argument('--linkage_method', type=str, default='ward',
                       choices=['ward', 'complete', 'average', 'single'],
                       help='링크age 방법')
    parser.add_argument('--max_docs', type=int, default=None,
                       help='최대 문서 수 (None이면 전체, 너무 많으면 샘플링)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. TF-IDF 행렬 로드
    logger.info(f"TF-IDF 행렬 로드 중: {args.tfidf_pkl}")
    with open(args.tfidf_pkl, 'rb') as f:
        tfidf_data = pickle.load(f)
    
    tfidf_matrix = tfidf_data['tfidf_matrix']
    tfidf_array = tfidf_matrix.toarray()
    
    logger.info(f"TF-IDF 행렬 로드 완료: {tfidf_array.shape}")
    
    # 2. 원본 데이터 로드 (레이블용)
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding='cp949')
    
    # 문서 수 제한 (덴드로그램은 너무 많은 문서에서 느려질 수 있음)
    if args.max_docs and len(df) > args.max_docs:
        logger.info(f"문서 수가 많아 {args.max_docs}개로 샘플링합니다.")
        df = df.sample(n=args.max_docs, random_state=42).reset_index(drop=True)
        tfidf_array = tfidf_array[:args.max_docs]
    
    # 3. 덴드로그램 생성
    logger.info("덴드로그램 생성 중...")
    visualizer = DendrogramVisualizer(linkage_method=args.linkage_method)
    
    # 문서 레이블 생성
    if args.id_column and args.id_column in df.columns:
        labels = [f"Doc_{row[args.id_column]}" for _, row in df.iterrows()]
    else:
        labels = [f"Doc_{i}" for i in range(len(df))]
    
    # 덴드로그램 저장 경로
    dendrogram_path = output_dir / "덴드로그램.png"
    
    # 덴드로그램 생성 및 저장
    visualizer.plot_dendrogram(
        tfidf_array,
        labels=labels,
        save_path=str(dendrogram_path),
        show=False
    )
    
    logger.info(f"덴드로그램 저장 완료: {dendrogram_path}")
    
    print(f"\n덴드로그램 생성 완료!")
    print(f"저장 위치: {dendrogram_path}")
    print(f"처리된 문서 수: {len(df)}개")


if __name__ == '__main__':
    main()

