"""
KoBERT 기반 문장 벡터화(임베딩) 파이프라인
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional

from src.config import DATA_DIR, TEXT_COLUMN
from src.model import KoBERTEmbedder

logger = logging.getLogger(__name__)


def generate_embeddings_from_csv(input_csv: str, output_npy: str, 
                                 text_column: str = TEXT_COLUMN,
                                 batch_size: int = 32,
                                 pooling_method: str = 'cls'):
    """
    CSV 파일에서 텍스트를 읽어 임베딩을 생성하고 저장
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_npy: 출력 numpy 파일 경로
        text_column: 텍스트 컬럼 이름
        batch_size: 배치 크기
        pooling_method: 'cls' 또는 'mean'
    """
    logger.info(f"CSV 파일 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    
    if text_column not in df.columns:
        raise ValueError(f"컬럼 '{text_column}'이 데이터에 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
    
    texts = df[text_column].astype(str).tolist()
    logger.info(f"{len(texts)}개의 텍스트를 로드했습니다.")
    
    # 임베딩 생성
    logger.info("임베딩을 생성합니다...")
    embedder = KoBERTEmbedder()
    embeddings = embedder.get_embeddings_numpy(texts, batch_size=batch_size, 
                                               pooling_method=pooling_method)
    
    logger.info(f"임베딩 shape: {embeddings.shape}")
    
    # 저장
    np.save(output_npy, embeddings)
    logger.info(f"임베딩이 {output_npy}에 저장되었습니다.")
    
    return embeddings


def generate_embeddings_from_texts(texts: List[str], batch_size: int = 32,
                                   pooling_method: str = 'cls') -> np.ndarray:
    """
    텍스트 리스트에서 임베딩 생성
    
    Args:
        texts: 텍스트 리스트
        batch_size: 배치 크기
        pooling_method: 'cls' 또는 'mean'
    
    Returns:
        임베딩 numpy 배열 (num_texts, hidden_dim)
    """
    embedder = KoBERTEmbedder()
    embeddings = embedder.get_embeddings_numpy(texts, batch_size=batch_size,
                                              pooling_method=pooling_method)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="KoBERT 문장 벡터화")
    parser.add_argument("--input_csv", type=str, required=True, 
                       help="입력 CSV 파일 경로")
    parser.add_argument("--output_npy", type=str, required=True,
                       help="출력 numpy 파일 경로")
    parser.add_argument("--text_column", type=str, default=TEXT_COLUMN,
                       help="텍스트 컬럼 이름")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="배치 크기")
    parser.add_argument("--pooling_method", type=str, default='cls',
                       choices=['cls', 'mean'],
                       help="풀링 방법: 'cls' (CLS 토큰) 또는 'mean' (평균 풀링)")
    
    args = parser.parse_args()
    
    generate_embeddings_from_csv(
        args.input_csv,
        args.output_npy,
        args.text_column,
        args.batch_size,
        args.pooling_method
    )


if __name__ == "__main__":
    main()

