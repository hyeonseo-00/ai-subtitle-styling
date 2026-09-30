"""
통합 텍스트 마이닝 파이프라인
모든 분석을 한 번에 수행

[실행 순서 6] 전체 통합 파이프라인 (선택적)
1~5단계를 모두 순차적으로 실행합니다.
개별 단계를 따로 실행하는 것을 권장하지만, 한 번에 실행하고 싶을 때 사용합니다.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional
import argparse

from sentiment_analysis import SentimentAnalyzer
from morphological_analysis import MorphologicalAnalyzer
from tfidf_analysis import TFIDFAnalyzer, FrequencyAnalyzer
from cam_visualization import CAMVisualizer

# BERTopic 클러스터링 모듈 import
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("bertopic_clustering", "3_BERTopic_클러스터링.py")
bertopic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bertopic_module)
run_clustering = bertopic_module.run_clustering

from dendrogram import DendrogramVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='통합 파이프라인 (실행 순서 6 - 선택적)')
    
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
                       help='KcELECTRA 모델 이름 (감정분석용)')
    parser.add_argument('--morph_analyzer', type=str, default='kiwi',
                       choices=['kiwi', 'kkma', 'komoran', 'mecab', 'okt'],
                       help='형태소 분석기 타입')
    parser.add_argument('--embedding_model', type=str, 
                       default='jhgan/ko-sroberta-multitask',
                       help='임베딩 모델 이름 (BERTopic용)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("통합 파이프라인 시작")
    logger.info("=" * 60)
    
    # 데이터 로드
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding='cp949')
    
    if args.id_column and args.id_column in df.columns:
        df = df.rename(columns={args.id_column: 'id'})
    else:
        df['id'] = df.index
    
    df = df.dropna(subset=[args.text_column])
    texts = df[args.text_column].tolist()
    
    logger.info(f"데이터 로드 완료: {len(texts)}개 문서")
    
    # 1. 형태소 분석
    logger.info("1단계: 형태소 분석...")
    morph_analyzer = MorphologicalAnalyzer(analyzer_type=args.morph_analyzer)
    keywords_list = [morph_analyzer.extract_keywords(text) for text in texts]
    
    # 2. TF-IDF 분석
    logger.info("2단계: TF-IDF 분석...")
    tfidf_analyzer = TFIDFAnalyzer()
    tokenized_texts = [' '.join(keywords) for keywords in keywords_list]
    tfidf_matrix = tfidf_analyzer.fit_transform(tokenized_texts)
    
    # 3. 덴드로그램 생성
    logger.info("3단계: 덴드로그램 생성...")
    visualizer = DendrogramVisualizer(linkage_method='ward')
    labels = [f"Doc_{i}" for i in range(len(texts))]
    visualizer.plot_dendrogram(
        tfidf_matrix.toarray(),
        labels=labels,
        save_path=str(output_dir / "덴드로그램.png"),
        show=False
    )
    
    # 4. BERTopic 클러스터링
    logger.info("4단계: BERTopic 클러스터링...")
    df_for_clustering = pd.DataFrame({
        'id': df['id'].values,
        'content': texts
    })
    df_topics, df_topic_info, _ = run_clustering(
        df=df_for_clustering,
        embedding_model_name=args.embedding_model
    )
    df_topics.to_csv(output_dir / "문서별_토픽할당.csv",
                    index=False, encoding='utf-8-sig')
    df_topic_info.to_csv(output_dir / "토픽요약정보.csv",
                        index=False, encoding='utf-8-sig')
    
    # 5. 감정분석
    logger.info("5단계: 감정분석...")
    sentiment_analyzer = SentimentAnalyzer(model_name=args.model)
    results = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        batch_results = sentiment_analyzer.predict_batch(batch, return_probs=True)
        results.extend(batch_results)
    
    sentiment_df = pd.DataFrame([
        {
            'id': df.iloc[i]['id'],
            'sentiment_label': label,
            'sentiment_confidence': max(probs.values()),
        }
        for i, (label, probs) in enumerate(results)
    ])
    sentiment_df.to_csv(output_dir / "감정분석_결과.csv",
                       index=False, encoding='utf-8-sig')
    
    # 6. CAM 시각화
    logger.info("6단계: CAM 기회영역 시각화...")
    cam_visualizer = CAMVisualizer()
    cluster_labels = df_topics['topic_id'].values.tolist()
    satisfaction_scores = sentiment_df['sentiment_confidence'].values
    sentiment_labels = sentiment_df['sentiment_label'].values
    satisfaction_scores = satisfaction_scores * (2 * sentiment_labels - 1)
    
    opportunity_df = cam_visualizer.display_importance_map(
        cluster_labels=cluster_labels,
        action_labels=cluster_labels,
        satisfaction_scores=satisfaction_scores.tolist(),
        save_path=str(output_dir / "기회영역_분석.csv")
    )
    opportunity_df.to_csv(output_dir / "기회영역_분석.csv",
                         index=False, encoding='utf-8-sig')
    
    logger.info("=" * 60)
    logger.info("통합 파이프라인 완료")
    logger.info("=" * 60)
    
    print(f"\n전체 분석 완료! 결과는 {output_dir} 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    main()

