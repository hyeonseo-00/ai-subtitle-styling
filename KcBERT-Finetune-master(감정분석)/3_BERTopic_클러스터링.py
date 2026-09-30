"""
BERTopic 기반 토픽 클러스터링 스크립트

[실행 순서 3] BERTopic 클러스터링
CSV 파일을 입력받아 BERTopic으로 토픽 클러스터링을 수행합니다.
BERTopic은 문서 임베딩을 사용하여 자동으로 토픽을 발견합니다.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# BERTopic 및 관련 라이브러리
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)

# ============================================================================
# 설정 상수 (코드 상단에서 쉽게 변경 가능)
# ============================================================================

# SentenceTransformer 모델 선택
# 한국어 멀티태스크 모델 (기본값)
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# 다른 옵션들:
# - "jhgan/ko-sbert-nli"  # 한국어 NLI 모델
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 멀티링구얼
# - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 더 큰 멀티링구얼 모델

# UMAP 파라미터 (차원 축소)
UMAP_N_NEIGHBORS = 15  # 이웃 수 (기본값: 15, 작을수록 지역적 구조 강조)
UMAP_N_COMPONENTS = 5  # 축소할 차원 수 (기본값: 5)
UMAP_METRIC = 'cosine'  # 거리 메트릭 ('cosine', 'euclidean', 'manhattan' 등)
UMAP_MIN_DIST = 0.0  # 최소 거리 (0.0~1.0, 기본값: 0.0)

# HDBSCAN 파라미터 (클러스터링)
HDBSCAN_MIN_CLUSTER_SIZE = 10  # 최소 클러스터 크기 (기본값: 10)
HDBSCAN_MIN_SAMPLES = 5  # 최소 샘플 수 (기본값: 5, None이면 min_cluster_size와 동일)
HDBSCAN_METRIC = 'euclidean'  # 거리 메트릭
HDBSCAN_CLUSTER_SELECTION_METHOD = 'eom'  # 'eom' 또는 'leaf'

# BERTopic 파라미터
BERTOPIC_TOP_N_WORDS = 10  # 토픽당 대표 단어 수
BERTOPIC_VERBOSE = True  # 진행 상황 출력 여부
BERTOPIC_CALCULATE_PROBABILITIES = True  # 토픽 확률 계산 여부

# 출력 파일 경로
OUTPUT_TOPICS_PER_DOC = "문서별_토픽할당.csv"
OUTPUT_TOPIC_SUMMARY = "토픽요약정보.csv"


# ============================================================================
# 데이터 로딩 함수
# ============================================================================

def load_data(csv_path: str, 
              text_column: str = "content",
              id_column: Optional[str] = None,
              encoding: str = 'utf-8') -> pd.DataFrame:
    """
    CSV 파일에서 데이터 로드
    
    Args:
        csv_path: CSV 파일 경로
        text_column: 텍스트 컬럼명 (기본값: "content")
        id_column: ID 컬럼명 (None이면 인덱스 사용)
        encoding: 파일 인코딩 (기본값: 'utf-8')
    
    Returns:
        DataFrame with 'id' and 'content' columns
    """
    logger.info(f"데이터 로딩 중: {csv_path}")
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except UnicodeDecodeError:
        # UTF-8 실패 시 다른 인코딩 시도
        logger.warning("UTF-8 인코딩 실패, cp949 시도 중...")
        df = pd.read_csv(csv_path, encoding='cp949')
    
    # 텍스트 컬럼 확인
    if text_column not in df.columns:
        raise ValueError(f"텍스트 컬럼 '{text_column}'을 찾을 수 없습니다. "
                        f"사용 가능한 컬럼: {list(df.columns)}")
    
    # ID 컬럼 처리
    if id_column and id_column in df.columns:
        df = df.rename(columns={id_column: 'id'})
    else:
        # ID 컬럼이 없으면 인덱스로 생성
        df['id'] = df.index
    
    # 필요한 컬럼만 선택
    result_df = df[['id', text_column]].copy()
    result_df = result_df.rename(columns={text_column: 'content'})
    
    # 결측값 제거
    result_df = result_df.dropna(subset=['content'])
    
    # 빈 텍스트 제거
    result_df = result_df[result_df['content'].str.strip().str.len() > 0]
    
    # 인덱스 리셋
    result_df = result_df.reset_index(drop=True)
    
    logger.info(f"데이터 로딩 완료: {len(result_df)}개 문서")
    
    return result_df


# ============================================================================
# 임베딩 모델 생성 함수
# ============================================================================

def build_embedding_model(model_name: str = EMBEDDING_MODEL_NAME,
                          device: Optional[str] = None) -> SentenceTransformer:
    """
    SentenceTransformer 임베딩 모델 생성
    
    Args:
        model_name: 모델 이름 또는 경로
        device: 'cuda', 'cpu', 또는 None (자동 선택)
    
    Returns:
        SentenceTransformer 모델
    """
    logger.info(f"임베딩 모델 로딩 중: {model_name}")
    
    # GPU 사용 가능 여부 확인
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    
    logger.info(f"사용 디바이스: {device}")
    
    # 모델 로드
    model = SentenceTransformer(model_name, device=device)
    
    logger.info("임베딩 모델 로딩 완료")
    
    return model


# ============================================================================
# BERTopic 모델 생성 함수
# ============================================================================

def build_bertopic_model(embedding_model: Optional[SentenceTransformer] = None,
                         embedding_model_name: str = EMBEDDING_MODEL_NAME,
                         umap_params: Optional[Dict] = None,
                         hdbscan_params: Optional[Dict] = None,
                         **bertopic_kwargs) -> BERTopic:
    """
    BERTopic 모델 생성 및 설정
    
    Args:
        embedding_model: SentenceTransformer 모델 (None이면 자동 생성)
        embedding_model_name: 임베딩 모델 이름 (embedding_model이 None일 때 사용)
        umap_params: UMAP 파라미터 딕셔너리 (None이면 기본값 사용)
        hdbscan_params: HDBSCAN 파라미터 딕셔너리 (None이면 기본값 사용)
        **bertopic_kwargs: BERTopic 추가 파라미터
    
    Returns:
        BERTopic 모델
    """
    logger.info("BERTopic 모델 생성 중...")
    
    # 임베딩 모델 설정
    if embedding_model is None:
        embedding_model = build_embedding_model(embedding_model_name)
    
    # UMAP 파라미터 설정
    if umap_params is None:
        umap_params = {
            'n_neighbors': UMAP_N_NEIGHBORS,
            'n_components': UMAP_N_COMPONENTS,
            'metric': UMAP_METRIC,
            'min_dist': UMAP_MIN_DIST,
            'random_state': 42
        }
    
    umap_model = UMAP(**umap_params)
    logger.info(f"UMAP 설정: {umap_params}")
    
    # HDBSCAN 파라미터 설정
    if hdbscan_params is None:
        hdbscan_params = {
            'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,
            'min_samples': HDBSCAN_MIN_SAMPLES,
            'metric': HDBSCAN_METRIC,
            'cluster_selection_method': HDBSCAN_CLUSTER_SELECTION_METHOD,
            'prediction_data': True  # 새로운 데이터 예측을 위해 필요
        }
    
    hdbscan_model = HDBSCAN(**hdbscan_params)
    logger.info(f"HDBSCAN 설정: {hdbscan_params}")
    
    # BERTopic 기본 파라미터
    default_bertopic_params = {
        'top_n_words': BERTOPIC_TOP_N_WORDS,
        'verbose': BERTOPIC_VERBOSE,
        'calculate_probabilities': BERTOPIC_CALCULATE_PROBABILITIES
    }
    default_bertopic_params.update(bertopic_kwargs)
    
    # BERTopic 모델 생성
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        **default_bertopic_params
    )
    
    logger.info("BERTopic 모델 생성 완료")
    
    return topic_model


# ============================================================================
# 클러스터링 실행 함수
# ============================================================================

def run_clustering(df: pd.DataFrame,
                  topic_model: Optional[BERTopic] = None,
                  embedding_model_name: str = EMBEDDING_MODEL_NAME,
                  save_model_path: Optional[str] = None,
                  load_model_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, BERTopic]:
    """
    BERTopic을 사용한 토픽 클러스터링 실행
    
    Args:
        df: 입력 DataFrame (최소한 'id', 'content' 컬럼 필요)
        topic_model: BERTopic 모델 (None이면 새로 생성)
        embedding_model_name: 임베딩 모델 이름
        save_model_path: 모델 저장 경로 (None이면 저장 안 함)
        load_model_path: 저장된 모델 로드 경로 (None이면 새로 생성)
    
    Returns:
        (df_topics, df_topic_info, topic_model) 튜플
        - df_topics: 문서별 토픽 할당 DataFrame
        - df_topic_info: 토픽 요약 정보 DataFrame
        - topic_model: 학습된 BERTopic 모델
    """
    logger.info("=" * 60)
    logger.info("BERTopic 클러스터링 시작")
    logger.info("=" * 60)
    
    # 데이터 확인
    if 'content' not in df.columns or 'id' not in df.columns:
        raise ValueError("DataFrame에 'id'와 'content' 컬럼이 필요합니다.")
    
    documents = df['content'].tolist()
    doc_ids = df['id'].tolist()
    
    logger.info(f"처리할 문서 수: {len(documents)}")
    
    # 모델 로드 또는 생성
    if load_model_path:
        logger.info(f"저장된 모델 로드 중: {load_model_path}")
        topic_model = BERTopic.load(load_model_path)
    elif topic_model is None:
        logger.info("새로운 BERTopic 모델 생성 중...")
        topic_model = build_bertopic_model(embedding_model_name=embedding_model_name)
    
    # ========================================================================
    # 1단계: 임베딩 생성
    # ========================================================================
    logger.info("1단계: 문서 임베딩 생성 중...")
    # BERTopic이 내부적으로 임베딩을 생성하므로 별도 처리 불필요
    
    # ========================================================================
    # 2단계: 토픽 모델링 (BERTopic fit)
    # ========================================================================
    logger.info("2단계: 토픽 모델링 수행 중...")
    logger.info("이 과정은 문서 수에 따라 시간이 걸릴 수 있습니다.")
    
    topics, probs = topic_model.fit_transform(documents)
    
    logger.info(f"토픽 모델링 완료: {len(set(topics)) - (1 if -1 in topics else 0)}개 토픽 발견")
    logger.info(f"노이즈 문서 수: {topics.count(-1) if isinstance(topics, list) else (topics == -1).sum()}")
    
    # ========================================================================
    # 3단계: 결과 DataFrame 생성
    # ========================================================================
    logger.info("3단계: 결과 DataFrame 생성 중...")
    
    # 문서별 토픽 할당 DataFrame
    df_topics = pd.DataFrame({
        'id': doc_ids,
        'content': documents,
        'topic_id': topics,
        'topic_prob': probs if probs is not None else [None] * len(topics)
    })
    
    # 토픽 요약 정보 DataFrame
    topic_info = topic_model.get_topic_info()
    df_topic_info = topic_info.copy()
    
    # 토픽 이름과 대표 키워드 추가
    topic_names = []
    topic_keywords = []
    
    for topic_id in df_topic_info['Topic'].values:
        if topic_id == -1:
            topic_names.append("노이즈 (Noise)")
            topic_keywords.append([])
        else:
            # 토픽의 대표 단어들 가져오기
            words = topic_model.get_topic(topic_id)
            if words:
                keywords = [word for word, _ in words[:5]]  # 상위 5개 단어
                topic_keywords.append(keywords)
                # 토픽 이름 생성 (상위 3개 단어 조합)
                name = "_".join(keywords[:3])
                topic_names.append(name)
            else:
                topic_names.append(f"Topic_{topic_id}")
                topic_keywords.append([])
    
    df_topic_info['Name'] = topic_names
    df_topic_info['Representation'] = topic_keywords
    
    # 컬럼 순서 재정렬
    df_topic_info = df_topic_info[['Topic', 'Count', 'Name', 'Representation']]
    
    logger.info("결과 DataFrame 생성 완료")
    
    # ========================================================================
    # 4단계: 모델 저장 (선택적)
    # ========================================================================
    if save_model_path:
        logger.info(f"모델 저장 중: {save_model_path}")
        topic_model.save(save_model_path)
        logger.info("모델 저장 완료")
    
    logger.info("=" * 60)
    logger.info("BERTopic 클러스터링 완료")
    logger.info("=" * 60)
    
    return df_topics, df_topic_info, topic_model


# ============================================================================
# 결과 저장 함수
# ============================================================================

def save_results(df_topics: pd.DataFrame,
                df_topic_info: pd.DataFrame,
                output_dir: str = "output",
                topics_per_doc_filename: str = OUTPUT_TOPICS_PER_DOC,
                topic_summary_filename: str = OUTPUT_TOPIC_SUMMARY) -> None:
    """
    클러스터링 결과를 CSV 파일로 저장
    
    Args:
        df_topics: 문서별 토픽 할당 DataFrame
        df_topic_info: 토픽 요약 정보 DataFrame
        output_dir: 출력 디렉토리
        topics_per_doc_filename: 문서별 토픽 파일명
        topic_summary_filename: 토픽 요약 파일명
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 문서별 토픽 할당 저장
    topics_path = output_path / topics_per_doc_filename
    df_topics.to_csv(topics_path, index=False, encoding='utf-8-sig')
    logger.info(f"문서별 토픽 할당 저장: {topics_path}")
    
    # 토픽 요약 정보 저장
    summary_path = output_path / topic_summary_filename
    df_topic_info.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"토픽 요약 정보 저장: {summary_path}")


# ============================================================================
# 결과 출력 및 시각화 함수
# ============================================================================

def print_topic_summary(df_topic_info: pd.DataFrame,
                       df_topics: pd.DataFrame,
                       top_n: int = 10,
                       show_examples: bool = True,
                       examples_per_topic: int = 2) -> None:
    """
    토픽 요약 정보 출력
    
    Args:
        df_topic_info: 토픽 요약 정보 DataFrame
        df_topics: 문서별 토픽 할당 DataFrame
        top_n: 출력할 상위 토픽 수
        show_examples: 대표 문서 예시 출력 여부
        examples_per_topic: 토픽당 예시 문서 수
    """
    print("\n" + "=" * 80)
    print("토픽 요약 정보")
    print("=" * 80)
    
    # 노이즈 제외하고 상위 토픽만 선택
    df_filtered = df_topic_info[df_topic_info['Topic'] != -1].head(top_n)
    
    for idx, row in df_filtered.iterrows():
        topic_id = row['Topic']
        count = row['Count']
        name = row['Name']
        keywords = row['Representation']
        
        print(f"\n[토픽 {topic_id}] {name}")
        print(f"  문서 수: {count}개")
        print(f"  대표 키워드: {', '.join(keywords) if keywords else 'N/A'}")
        
        # 대표 문서 예시 출력
        if show_examples:
            topic_docs = df_topics[df_topics['topic_id'] == topic_id]
            if len(topic_docs) > 0:
                print(f"  대표 문서 예시:")
                for i, (_, doc_row) in enumerate(topic_docs.head(examples_per_topic).iterrows()):
                    content = doc_row['content']
                    # 텍스트 길이 제한
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"    {i+1}. {content}")
    
    # 노이즈 토픽 정보
    noise_count = df_topic_info[df_topic_info['Topic'] == -1]['Count'].values
    if len(noise_count) > 0:
        print(f"\n[노이즈 토픽] 문서 수: {noise_count[0]}개")
    
    print("\n" + "=" * 80)


# ============================================================================
# 메인 실행 함수
# ============================================================================

def main(csv_path: str,
        text_column: str = "content",
        id_column: Optional[str] = None,
        output_dir: str = "output",
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        save_model: bool = False,
        load_model: Optional[str] = None,
        print_summary: bool = True,
        top_n_topics: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 파이프라인 실행
    
    Args:
        csv_path: 입력 CSV 파일 경로
        text_column: 텍스트 컬럼명
        id_column: ID 컬럼명
        output_dir: 출력 디렉토리
        embedding_model_name: 임베딩 모델 이름
        save_model: 모델 저장 여부
        load_model: 저장된 모델 로드 경로
        print_summary: 요약 정보 출력 여부
        top_n_topics: 출력할 상위 토픽 수
    
    Returns:
        (df_topics, df_topic_info) 튜플
    """
    # 1. 데이터 로드
    df = load_data(csv_path, text_column=text_column, id_column=id_column)
    
    # 2. 클러스터링 실행
    model_path = None
    if save_model:
        model_path = Path(output_dir) / "bertopic_model"
    
    df_topics, df_topic_info, topic_model = run_clustering(
        df=df,
        embedding_model_name=embedding_model_name,
        save_model_path=str(model_path) if model_path else None,
        load_model_path=load_model
    )
    
    # 3. 결과 저장
    save_results(df_topics, df_topic_info, output_dir=output_dir)
    
    # 4. 요약 정보 출력
    if print_summary:
        print_topic_summary(df_topic_info, df_topics, top_n=top_n_topics)
    
    return df_topics, df_topic_info


# ============================================================================
# 실행 예시
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='BERTopic 클러스터링 (실행 순서 3)')
    parser.add_argument('--input', type=str, required=True,
                       help='입력 CSV 파일 경로')
    parser.add_argument('--text_column', type=str, default='content',
                       help='텍스트 컬럼명 (기본값: content)')
    parser.add_argument('--id_column', type=str, default=None,
                       help='ID 컬럼명 (선택적)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='결과 저장 디렉토리')
    parser.add_argument('--embedding_model', type=str, 
                       default='jhgan/ko-sroberta-multitask',
                       help='임베딩 모델 이름')
    parser.add_argument('--save_model', action='store_true',
                       help='모델 저장 여부')
    
    args = parser.parse_args()
    
    # 실행
    df_topics, df_topic_info = main(
        csv_path=args.input,
        text_column=args.text_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
        embedding_model_name=args.embedding_model,
        save_model=args.save_model
    )
    
    print(f"\n클러스터링 완료! 결과는 {args.output_dir} 디렉토리에 저장되었습니다.")
    print(f"- 문서별_토픽할당.csv: 문서별 토픽 할당")
    print(f"- 토픽요약정보.csv: 토픽 요약 정보")

