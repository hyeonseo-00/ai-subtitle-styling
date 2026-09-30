"""
CAM (Class Activation Map) 기회영역 시각화 스크립트
Importance 계산 및 기회영역 액션맵 생성

[실행 순서 5] CAM 기회영역 시각화
토픽 클러스터링 결과와 감정분석 결과를 결합하여 Importance 맵과 기회영역을 시각화합니다.
"""
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Optional

from cam_visualization import CAMVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CAM 기회영역 시각화 (실행 순서 5)')
    
    parser.add_argument('--topics_csv', type=str, 
                       default='output/문서별_토픽할당.csv',
                       help='문서별 토픽 할당 CSV 파일 경로 (3단계에서 생성)')
    parser.add_argument('--sentiment_csv', type=str,
                       default='output/감정분석_결과.csv',
                       help='감정분석 결과 CSV 파일 경로 (4단계에서 생성)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 토픽 할당 데이터 로드
    logger.info(f"토픽 할당 데이터 로드 중: {args.topics_csv}")
    topics_df = pd.read_csv(args.topics_csv, encoding='utf-8-sig')
    
    # 2. 감정분석 결과 로드
    logger.info(f"감정분석 결과 로드 중: {args.sentiment_csv}")
    sentiment_df = pd.read_csv(args.sentiment_csv, encoding='utf-8-sig')
    
    # 3. 데이터 병합 (ID 기준)
    if 'id' in topics_df.columns and 'id' in sentiment_df.columns:
        merged_df = pd.merge(topics_df, sentiment_df, on='id', how='inner')
    else:
        # ID가 없으면 인덱스로 병합
        merged_df = pd.concat([topics_df, sentiment_df], axis=1)
    
    logger.info(f"병합된 데이터: {len(merged_df)}개 문서")
    
    # 4. CAM 시각화
    logger.info("CAM (Importance 맵) 생성 중...")
    cam_visualizer = CAMVisualizer()
    
    # 토픽 레이블 추출
    cluster_labels = merged_df['topic_id'].values.tolist()
    action_labels = merged_df['topic_id'].values.tolist()  # 토픽을 액션으로 사용
    
    # Satisfaction 점수 추출
    satisfaction_scores = merged_df['sentiment_confidence'].values
    sentiment_labels = merged_df['sentiment_label'].values
    # 레이블에 따라 부호 조정 (0이면 음수, 1이면 양수)
    satisfaction_scores = satisfaction_scores * (2 * sentiment_labels - 1)
    
    # Opportunity 분석 수행
    opportunity_df = cam_visualizer.display_importance_map(
        cluster_labels=cluster_labels,
        action_labels=action_labels,
        satisfaction_scores=satisfaction_scores.tolist(),
        save_path=str(output_dir / "기회영역_분석.csv")
    )
    
    # 5. 결과 저장
    opportunity_df.to_csv(output_dir / "기회영역_분석.csv",
                         index=False, encoding='utf-8-sig')
    
    logger.info("CAM (Importance 맵) 생성 완료")
    
    print(f"\n기회영역 분석 완료!")
    print(f"저장 위치: {output_dir}")
    print(f"- 기회영역_분석.csv")
    print(f"- 기회영역_분석_plot.png (기회영역 액션맵)")


if __name__ == '__main__':
    main()

