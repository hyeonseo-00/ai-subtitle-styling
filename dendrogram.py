"""
덴드로그램 시각화 모듈
계층적 클러스터링 결과를 덴드로그램으로 시각화
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False


class DendrogramVisualizer:
    """덴드로그램 시각화 클래스"""
    
    def __init__(self, linkage_method: str = 'ward', metric: str = 'euclidean'):
        """
        Args:
            linkage_method: 'ward', 'complete', 'average', 'single' 중 선택
            metric: 거리 측정 방법 ('euclidean', 'cosine', 'manhattan' 등)
        """
        self.linkage_method = linkage_method
        self.metric = metric
        self.linkage_matrix = None
        self.distance_matrix = None
    
    def compute_linkage(self, feature_matrix: np.ndarray, 
                       method: Optional[str] = None,
                       metric: Optional[str] = None) -> np.ndarray:
        """
        계층적 클러스터링 링크age 행렬 계산
        
        Args:
            feature_matrix: 특성 행렬 (n_documents, n_features)
            method: 링크age 방법 (None이면 초기화 시 설정값 사용)
            metric: 거리 측정 방법 (None이면 초기화 시 설정값 사용)
        
        Returns:
            링크age 행렬
        """
        method = method or self.linkage_method
        metric = metric or self.metric
        
        # ward 방법은 metric을 사용하지 않음
        if method == 'ward':
            self.linkage_matrix = linkage(feature_matrix, method=method, metric='euclidean')
        else:
            # 거리 행렬 계산
            if metric == 'cosine':
                # 코사인 거리 직접 계산
                from sklearn.metrics.pairwise import cosine_distances
                self.distance_matrix = squareform(cosine_distances(feature_matrix))
            else:
                self.distance_matrix = pdist(feature_matrix, metric=metric)
            
            self.linkage_matrix = linkage(self.distance_matrix, method=method)
        
        logger.info(f"링크age 행렬 계산 완료: {self.linkage_matrix.shape}")
        
        return self.linkage_matrix
    
    def plot_dendrogram(self, feature_matrix: np.ndarray,
                       labels: Optional[List[str]] = None,
                       max_d: Optional[float] = None,
                       figsize: Tuple[int, int] = (15, 8),
                       title: str = "계층적 클러스터링 덴드로그램",
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
        """
        덴드로그램 시각화
        
        Args:
            feature_matrix: 특성 행렬
            labels: 문서 레이블 리스트 (None이면 인덱스 사용)
            max_d: 수평선을 그을 거리 (클러스터 수 결정)
            figsize: 그림 크기
            title: 제목
            save_path: 저장 경로 (None이면 저장 안 함)
            show: 화면에 표시 여부
        
        Returns:
            matplotlib Figure 객체
        """
        # 링크age 행렬 계산
        if self.linkage_matrix is None:
            self.compute_linkage(feature_matrix)
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 덴드로그램 그리기
        dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax
        )
        
        # 수평선 그리기 (선택적)
        if max_d is not None:
            ax.axhline(y=max_d, color='r', linestyle='--', linewidth=2, label=f'Cut at distance={max_d}')
            ax.legend()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('문서', fontsize=12)
        ax.set_ylabel('거리', fontsize=12)
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"덴드로그램 저장: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def get_clusters_from_dendrogram(self, feature_matrix: np.ndarray,
                                    n_clusters: int,
                                    criterion: str = 'maxclust') -> np.ndarray:
        """
        덴드로그램에서 클러스터 할당
        
        Args:
            feature_matrix: 특성 행렬
            n_clusters: 클러스터 수
            criterion: 'maxclust' 또는 'distance'
        
        Returns:
            클러스터 레이블 배열
        """
        if self.linkage_matrix is None:
            self.compute_linkage(feature_matrix)
        
        labels = fcluster(self.linkage_matrix, n_clusters, criterion=criterion)
        
        return labels
    
    def find_optimal_cut(self, feature_matrix: np.ndarray,
                        max_clusters: int = 10) -> Dict[str, float]:
        """
        최적의 클러스터 수 찾기 (엘보우 방법)
        
        Args:
            feature_matrix: 특성 행렬
            max_clusters: 최대 클러스터 수
        
        Returns:
            {클러스터 수: 거리} 딕셔너리
        """
        if self.linkage_matrix is None:
            self.compute_linkage(feature_matrix)
        
        # 각 클러스터 수에 대한 거리 계산
        distances = {}
        for k in range(2, max_clusters + 1):
            labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
            # 클러스터 간 평균 거리 계산
            unique_labels = np.unique(labels)
            if len(unique_labels) == k:
                # 링크age 행렬에서 해당 클러스터 수의 거리 찾기
                # 마지막 (n-k)번째 병합의 거리
                idx = len(feature_matrix) - k
                if idx >= 0 and idx < len(self.linkage_matrix):
                    distances[k] = float(self.linkage_matrix[idx, 2])
        
        return distances
    
    def plot_cluster_analysis(self, feature_matrix: np.ndarray,
                             labels: Optional[List[str]] = None,
                             max_clusters: int = 10,
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        클러스터 분석 시각화 (덴드로그램 + 거리 그래프)
        
        Args:
            feature_matrix: 특성 행렬
            labels: 문서 레이블
            max_clusters: 최대 클러스터 수
            figsize: 그림 크기
            save_path: 저장 경로
        
        Returns:
            matplotlib Figure 객체
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 덴드로그램
        if self.linkage_matrix is None:
            self.compute_linkage(feature_matrix)
        
        dendrogram(self.linkage_matrix, labels=labels, leaf_rotation=90, 
                  leaf_font_size=8, ax=axes[0])
        axes[0].set_title('계층적 클러스터링 덴드로그램', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('문서', fontsize=10)
        axes[0].set_ylabel('거리', fontsize=10)
        
        # 거리 그래프
        distances = self.find_optimal_cut(feature_matrix, max_clusters)
        if distances:
            k_values = list(distances.keys())
            dist_values = list(distances.values())
            axes[1].plot(k_values, dist_values, marker='o', linewidth=2, markersize=8)
            axes[1].set_title('클러스터 수에 따른 병합 거리', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('클러스터 수', fontsize=10)
            axes[1].set_ylabel('병합 거리', fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"클러스터 분석 그래프 저장: {save_path}")
        
        return fig

