"""
TF-IDF 기반 중요도 점수 분석 및 시각화
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
import logging
from typing import List, Dict, Tuple

from src.config import DATA_DIR, TEXT_COLUMN, LABEL_COLUMN, OUTPUT_DIR

logger = logging.getLogger(__name__)


class TFIDFAnalyzer:
    """
    TF-IDF 기반 텍스트 분석 클래스
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Args:
            max_features: 최대 특성 수
            ngram_range: n-gram 범위
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # 한국어 불용어는 별도로 처리 가능
            min_df=2,  # 최소 문서 빈도
            max_df=0.95  # 최대 문서 빈도
        )
        self.tfidf_matrix = None
        self.feature_names = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        TF-IDF 벡터화 수행
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            TF-IDF 행렬
        """
        logger.info("TF-IDF 벡터화를 수행합니다...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF 행렬 shape: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix
    
    def get_top_features(self, n: int = 20, class_labels: List[str] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        클래스별 상위 중요 단어 추출
        
        Args:
            n: 상위 n개 단어
            class_labels: 클래스 레이블 리스트 (None이면 전체)
        
        Returns:
            클래스별 상위 단어 딕셔너리
        """
        if self.tfidf_matrix is None:
            raise ValueError("먼저 fit_transform을 호출해야 합니다.")
        
        top_features = {}
        
        if class_labels is None:
            # 전체 평균
            mean_scores = np.mean(self.tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-n:][::-1]
            top_words = [(self.feature_names[i], mean_scores[i]) for i in top_indices]
            top_features['all'] = top_words
        else:
            # 클래스별
            unique_labels = list(set(class_labels))
            for label in unique_labels:
                label_indices = [i for i, l in enumerate(class_labels) if l == label]
                if not label_indices:
                    continue
                
                label_scores = np.mean(self.tfidf_matrix[label_indices].toarray(), axis=0)
                top_indices = np.argsort(label_scores)[-n:][::-1]
                top_words = [(self.feature_names[i], label_scores[i]) for i in top_indices]
                top_features[label] = top_words
        
        return top_features
    
    def plot_top_features(self, top_features: Dict[str, List[Tuple[str, float]]], 
                          save_path: str = None):
        """
        상위 중요 단어 시각화
        
        Args:
            top_features: get_top_features 결과
            save_path: 저장 경로
        """
        n_classes = len(top_features)
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6))
        
        if n_classes == 1:
            axes = [axes]
        
        for idx, (label, words_scores) in enumerate(top_features.items()):
            words, scores = zip(*words_scores)
            
            axes[idx].barh(range(len(words)), scores)
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].set_xlabel('TF-IDF 점수')
            axes[idx].set_title(f'상위 중요 단어: {label}')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"그래프가 {save_path}에 저장되었습니다.")
        else:
            plt.show()
        plt.close()
    
    def generate_wordcloud(self, texts: List[str], class_labels: List[str] = None,
                          save_path: str = None):
        """
        워드클라우드 생성
        
        Args:
            texts: 텍스트 리스트
            class_labels: 클래스 레이블 (None이면 전체)
            save_path: 저장 경로 (디렉토리면 클래스별로 저장)
        """
        if class_labels is None:
            # 전체 워드클라우드
            text_combined = ' '.join(texts)
            wordcloud = WordCloud(
                font_path='malgun.ttf',  # Windows 한글 폰트
                width=800,
                height=400,
                background_color='white'
            ).generate(text_combined)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('전체 워드클라우드')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"워드클라우드가 {save_path}에 저장되었습니다.")
            else:
                plt.show()
            plt.close()
        else:
            # 클래스별 워드클라우드
            unique_labels = list(set(class_labels))
            for label in unique_labels:
                label_texts = [texts[i] for i, l in enumerate(class_labels) if l == label]
                text_combined = ' '.join(label_texts)
                
                wordcloud = WordCloud(
                    font_path='malgun.ttf',
                    width=800,
                    height=400,
                    background_color='white'
                ).generate(text_combined)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'워드클라우드: {label}')
                
                if save_path:
                    label_path = Path(save_path).parent / f"wordcloud_{label}.png"
                    plt.savefig(label_path, dpi=300, bbox_inches='tight')
                    logger.info(f"워드클라우드가 {label_path}에 저장되었습니다.")
                else:
                    plt.show()
                plt.close()


def analyze_tfidf(input_csv: str, output_dir: str = None, n_top_words: int = 20):
    """
    TF-IDF 분석 수행
    
    Args:
        input_csv: 입력 CSV 파일
        output_dir: 출력 디렉토리
        n_top_words: 상위 단어 개수
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    logger.info(f"데이터 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    texts = df[TEXT_COLUMN].astype(str).tolist()
    
    # TF-IDF 분석
    analyzer = TFIDFAnalyzer(max_features=1000, ngram_range=(1, 2))
    analyzer.fit_transform(texts)
    
    # 클래스별 상위 단어 추출
    if LABEL_COLUMN in df.columns:
        labels = df[LABEL_COLUMN].tolist()
        top_features = analyzer.get_top_features(n=n_top_words, class_labels=labels)
        
        # 시각화
        analyzer.plot_top_features(top_features, 
                                  save_path=str(output_dir / "tfidf_top_features.png"))
        
        # 워드클라우드
        analyzer.generate_wordcloud(texts, labels, 
                                   save_path=str(output_dir / "wordcloud.png"))
        
        # 결과 저장
        import json
        result_dict = {}
        for label, words_scores in top_features.items():
            result_dict[label] = [{'word': w, 'score': float(s)} for w, s in words_scores]
        
        with open(output_dir / "tfidf_top_features.json", 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"TF-IDF 분석 결과가 {output_dir}에 저장되었습니다.")
    else:
        logger.warning(f"레이블 컬럼 '{LABEL_COLUMN}'이 없어 전체 분석만 수행합니다.")
        top_features = analyzer.get_top_features(n=n_top_words)
        analyzer.plot_top_features(top_features, 
                                  save_path=str(output_dir / "tfidf_top_features.png"))


def main():
    parser = argparse.ArgumentParser(description="TF-IDF 중요도 분석")
    parser.add_argument("--input_csv", type=str, required=True, help="입력 CSV 파일")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), 
                       help="출력 디렉토리")
    parser.add_argument("--n_top_words", type=int, default=20, 
                       help="상위 단어 개수")
    
    args = parser.parse_args()
    
    analyze_tfidf(args.input_csv, args.output_dir, args.n_top_words)


if __name__ == "__main__":
    main()

