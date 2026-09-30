"""
LDA (Latent Dirichlet Allocation) 토픽 모델링 및 시각화
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
from pathlib import Path
import logging
from typing import List, Tuple

from src.config import DATA_DIR, TEXT_COLUMN, OUTPUT_DIR

logger = logging.getLogger(__name__)


class LDAAnalyzer:
    """
    LDA 토픽 모델링 클래스
    """
    
    def __init__(self, n_topics: int = 5, max_features: int = 1000, 
                 ngram_range: Tuple[int, int] = (1, 2)):
        """
        Args:
            n_topics: 토픽 개수
            max_features: 최대 특성 수
            ngram_range: n-gram 범위
        """
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online',
            learning_offset=50.0,
            max_iter=10
        )
        self.doc_topic_dist = None
        self.topic_word_dist = None
        self.feature_names = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        LDA 모델 학습 및 변환
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            문서-토픽 분포 행렬
        """
        logger.info("LDA 모델을 학습합니다...")
        
        # Count 벡터화
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # LDA 학습
        self.doc_topic_dist = self.lda.fit_transform(doc_term_matrix)
        self.topic_word_dist = self.lda.components_
        
        logger.info(f"LDA 모델 학습 완료 (토픽 수: {self.n_topics})")
        logger.info(f"문서-토픽 분포 shape: {self.doc_topic_dist.shape}")
        
        return self.doc_topic_dist
    
    def get_top_words_per_topic(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        토픽별 상위 단어 추출
        
        Args:
            n_words: 토픽당 상위 단어 개수
        
        Returns:
            토픽별 상위 단어 리스트
        """
        if self.topic_word_dist is None:
            raise ValueError("먼저 fit_transform을 호출해야 합니다.")
        
        top_words_per_topic = []
        
        for topic_idx in range(self.n_topics):
            topic_weights = self.topic_word_dist[topic_idx]
            top_indices = np.argsort(topic_weights)[-n_words:][::-1]
            top_words = [(self.feature_names[i], topic_weights[i]) for i in top_indices]
            top_words_per_topic.append(top_words)
        
        return top_words_per_topic
    
    def print_topics(self, n_words: int = 10):
        """
        토픽별 상위 단어 출력
        
        Args:
            n_words: 토픽당 상위 단어 개수
        """
        top_words_per_topic = self.get_top_words_per_topic(n_words)
        
        for topic_idx, top_words in enumerate(top_words_per_topic):
            words_str = ', '.join([f"{word}({score:.4f})" for word, score in top_words])
            logger.info(f"토픽 {topic_idx + 1}: {words_str}")
    
    def visualize(self, texts: List[str], save_path: str = None):
        """
        pyLDAvis를 사용한 LDA 시각화
        
        Args:
            texts: 텍스트 리스트 (벡터화에 사용)
            save_path: HTML 파일 저장 경로
        """
        if self.topic_word_dist is None:
            raise ValueError("먼저 fit_transform을 호출해야 합니다.")
        
        logger.info("LDA 시각화를 생성합니다...")
        
        # 문서-단어 행렬 재생성 (pyLDAvis용)
        doc_term_matrix = self.vectorizer.transform(texts)
        
        # pyLDAvis 준비
        vis_data = pyLDAvis.sklearn.prepare(
            self.lda,
            doc_term_matrix,
            self.vectorizer,
            mds='tsne'
        )
        
        # 시각화 저장
        if save_path:
            pyLDAvis.save_html(vis_data, save_path)
            logger.info(f"LDA 시각화가 {save_path}에 저장되었습니다.")
        else:
            pyLDAvis.display(vis_data)
    
    def get_document_topics(self, texts: List[str]) -> pd.DataFrame:
        """
        문서별 주요 토픽 반환
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            문서-토픽 분포 DataFrame
        """
        if self.doc_topic_dist is None:
            raise ValueError("먼저 fit_transform을 호출해야 합니다.")
        
        df = pd.DataFrame(self.doc_topic_dist, 
                         columns=[f"Topic_{i+1}" for i in range(self.n_topics)])
        df['dominant_topic'] = df.idxmax(axis=1)
        df['text'] = texts
        
        return df


def analyze_lda(input_csv: str, n_topics: int = 5, output_dir: str = None,
                n_words: int = 10):
    """
    LDA 토픽 모델링 수행
    
    Args:
        input_csv: 입력 CSV 파일
        n_topics: 토픽 개수
        output_dir: 출력 디렉토리
        n_words: 토픽당 상위 단어 개수
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    logger.info(f"데이터 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    texts = df[TEXT_COLUMN].astype(str).tolist()
    
    # LDA 분석
    analyzer = LDAAnalyzer(n_topics=n_topics)
    analyzer.fit_transform(texts)
    
    # 토픽 출력
    analyzer.print_topics(n_words=n_words)
    
    # 시각화
    vis_path = output_dir / "lda_visualization.html"
    analyzer.visualize(texts, save_path=str(vis_path))
    
    # 문서-토픽 분포 저장
    doc_topics_df = analyzer.get_document_topics(texts)
    doc_topics_path = output_dir / "document_topics.csv"
    doc_topics_df.to_csv(doc_topics_path, index=False, encoding='utf-8')
    logger.info(f"문서-토픽 분포가 {doc_topics_path}에 저장되었습니다.")
    
    # 토픽별 상위 단어 저장
    import json
    top_words_per_topic = analyzer.get_top_words_per_topic(n_words)
    topics_dict = {}
    for topic_idx, top_words in enumerate(top_words_per_topic):
        topics_dict[f"topic_{topic_idx + 1}"] = [
            {'word': word, 'weight': float(weight)} for word, weight in top_words
        ]
    
    topics_path = output_dir / "lda_topics.json"
    with open(topics_path, 'w', encoding='utf-8') as f:
        json.dump(topics_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"LDA 분석 결과가 {output_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="LDA 토픽 모델링 및 시각화")
    parser.add_argument("--input_csv", type=str, required=True, help="입력 CSV 파일")
    parser.add_argument("--n_topics", type=int, default=5, help="토픽 개수")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), 
                       help="출력 디렉토리")
    parser.add_argument("--n_words", type=int, default=10, 
                       help="토픽당 상위 단어 개수")
    
    args = parser.parse_args()
    
    analyze_lda(args.input_csv, args.n_topics, args.output_dir, args.n_words)


if __name__ == "__main__":
    main()

