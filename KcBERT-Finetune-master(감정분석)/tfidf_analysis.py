"""
TF-IDF 분석 모듈
빈도분석 및 TF-IDF 계산
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TFIDFAnalyzer:
    """TF-IDF 및 빈도분석 클래스"""
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2)):
        """
        Args:
            max_features: 최대 특성 수
            min_df: 최소 문서 빈도
            max_df: 최대 문서 빈도 비율
            ngram_range: n-gram 범위 (예: (1, 2) = unigram + bigram)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
    
    def fit_transform(self, documents: List[str], tokenizer=None) -> np.ndarray:
        """
        문서 리스트에 대해 TF-IDF 계산
        
        Args:
            documents: 문서 리스트 (이미 토큰화된 문자열 또는 원본 텍스트)
            tokenizer: 커스텀 토크나이저 함수 (None이면 기본 사용)
        
        Returns:
            TF-IDF 행렬 (n_documents, n_features)
        """
        # 토크나이저가 제공되면 사용
        if tokenizer:
            documents = [' '.join(tokenizer(doc)) for doc in documents]
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            token_pattern=r'\S+'  # 공백으로 구분된 토큰
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"TF-IDF 행렬 생성 완료: {self.tfidf_matrix.shape}")
        
        return self.tfidf_matrix
    
    def get_top_features(self, n: int = 20, document_idx: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        상위 n개 특성 추출
        
        Args:
            n: 추출할 특성 수
            document_idx: 특정 문서 인덱스 (None이면 전체 평균)
        
        Returns:
            (특성명, TF-IDF 점수) 튜플 리스트
        """
        if self.tfidf_matrix is None:
            raise ValueError("먼저 fit_transform을 호출하세요.")
        
        if document_idx is not None:
            scores = self.tfidf_matrix[document_idx].toarray().flatten()
        else:
            scores = np.mean(self.tfidf_matrix.toarray(), axis=0)
        
        # 상위 n개 인덱스
        top_indices = np.argsort(scores)[::-1][:n]
        
        return [(self.feature_names[idx], float(scores[idx])) for idx in top_indices]
    
    def get_feature_importance(self, document_idx: int, top_n: int = 20) -> pd.DataFrame:
        """
        특정 문서의 특성 중요도
        
        Args:
            document_idx: 문서 인덱스
            top_n: 상위 n개
        
        Returns:
            DataFrame with columns: ['feature', 'tfidf_score']
        """
        top_features = self.get_top_features(n=top_n, document_idx=document_idx)
        
        df = pd.DataFrame(top_features, columns=['feature', 'tfidf_score'])
        df = df.sort_values('tfidf_score', ascending=False)
        
        return df
    
    def transform(self, documents: List[str], tokenizer=None) -> np.ndarray:
        """
        새로운 문서에 대해 TF-IDF 변환 (이미 fit된 경우)
        
        Args:
            documents: 문서 리스트
            tokenizer: 커스텀 토크나이저 함수
        
        Returns:
            TF-IDF 행렬
        """
        if self.vectorizer is None:
            raise ValueError("먼저 fit_transform을 호출하세요.")
        
        if tokenizer:
            documents = [' '.join(tokenizer(doc)) for doc in documents]
        
        return self.vectorizer.transform(documents)


class FrequencyAnalyzer:
    """빈도분석 클래스"""
    
    @staticmethod
    def word_frequency(tokens: List[str], top_n: int = 50) -> List[Tuple[str, int]]:
        """
        단어 빈도 계산
        
        Args:
            tokens: 토큰 리스트
            top_n: 상위 n개
        
        Returns:
            (단어, 빈도) 튜플 리스트
        """
        counter = Counter(tokens)
        return counter.most_common(top_n)
    
    @staticmethod
    def document_frequency(documents: List[List[str]]) -> Dict[str, int]:
        """
        문서 빈도 (DF) 계산
        
        Args:
            documents: 문서별 토큰 리스트
        
        Returns:
            {단어: 문서 수} 딕셔너리
        """
        doc_freq = Counter()
        
        for doc_tokens in documents:
            unique_tokens = set(doc_tokens)
            doc_freq.update(unique_tokens)
        
        return dict(doc_freq)
    
    @staticmethod
    def term_frequency(tokens: List[str]) -> Dict[str, int]:
        """
        용어 빈도 (TF) 계산
        
        Args:
            tokens: 토큰 리스트
        
        Returns:
            {단어: 빈도} 딕셔너리
        """
        return dict(Counter(tokens))
    
    @staticmethod
    def frequency_statistics(tokens: List[str]) -> Dict[str, Union[int, float]]:
        """
        빈도 통계
        
        Args:
            tokens: 토큰 리스트
        
        Returns:
            통계 딕셔너리
        """
        counter = Counter(tokens)
        total = len(tokens)
        unique = len(counter)
        
        return {
            'total_tokens': total,
            'unique_tokens': unique,
            'vocabulary_size': unique,
            'avg_frequency': total / unique if unique > 0 else 0,
            'most_common': counter.most_common(10)
        }

