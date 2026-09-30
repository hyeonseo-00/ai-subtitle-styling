"""
공통 유틸리티 함수: 시드 고정, 로깅, 메트릭 계산 등
"""
import random
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    시드 고정 함수 (재현성을 위해)
    
    Args:
        seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"시드가 {seed}로 고정되었습니다.")


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, id2label: Dict[int, str]) -> Dict[str, float]:
    """
    분류 메트릭 계산 (Accuracy, F1-macro, F1-weighted)
    
    Args:
        predictions: 예측 레이블 (numpy array)
        labels: 실제 레이블 (numpy array)
        id2label: ID to label 매핑 딕셔너리
    
    Returns:
        메트릭 딕셔너리
    """
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    return metrics


def print_classification_report(predictions: np.ndarray, labels: np.ndarray, id2label: Dict[int, str]):
    """
    분류 리포트 출력
    
    Args:
        predictions: 예측 레이블
        labels: 실제 레이블
        id2label: ID to label 매핑
    """
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(labels, predictions, target_names=label_names)
    logger.info("\n분류 리포트:\n" + report)


def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, id2label: Dict[int, str], 
                          save_path: str = None):
    """
    혼동 행렬 시각화
    
    Args:
        predictions: 예측 레이블
        labels: 실제 레이블
        id2label: ID to label 매핑
        save_path: 저장 경로 (None이면 표시만)
    """
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.ylabel('실제 레이블')
    plt.xlabel('예측 레이블')
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"혼동 행렬이 {save_path}에 저장되었습니다.")
    else:
        plt.show()
    plt.close()

