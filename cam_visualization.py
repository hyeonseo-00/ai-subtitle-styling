"""
CAM (Class Activation Map) 시각화 모듈
기회영역 액션맵과 Importance 계산 및 시각화
Day3 실습 노트북의 로직을 참고
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional, Tuple
import logging

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    logging.warning("adjustText가 설치되지 않았습니다. 텍스트 레이블 자동 조정이 비활성화됩니다.")

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False


class CAMVisualizer:
    """Class Activation Map 시각화 클래스"""
    
    def __init__(self):
        """CAM 시각화 초기화"""
        self.scaler_importance = MinMaxScaler(feature_range=(0, 10))
        self.scaler_satisfaction = MinMaxScaler(feature_range=(-10, 10))
    
    def calculate_importance(self, cluster_labels: List[int], 
                             action_labels: List[int]) -> Dict[str, float]:
        """
        Importance 점수 계산 (전체 토픽의 비중)
        Day3 Cell 42-50 로직 참고
        
        Args:
            cluster_labels: 클러스터(액터) 레이블 리스트
            action_labels: 액션 레이블 리스트
        
        Returns:
            {ActorX_ActionY: importance_score} 딕셔너리
        """
        # Actor_Action 조합 생성
        action_flags = [
            f'Actor{actor}_Action{action}' 
            for actor, action in zip(cluster_labels, action_labels)
        ]
        
        # 빈도 계산
        frequency = Counter(action_flags)
        total_count = sum(frequency.values())
        
        # Importance 계산 (비율 * 100)
        importance_dict = {}
        for item, value in frequency.items():
            importance = (value / total_count) * 100
            importance_dict[item] = importance
        
        logger.info(f"Importance 계산 완료: {len(importance_dict)}개 액션")
        
        return importance_dict
    
    def normalize_importance(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Importance 점수 정규화 (0~10)
        Day3 Cell 46-50 로직 참고
        
        Args:
            importance_dict: Importance 딕셔너리
        
        Returns:
            정규화된 Importance 딕셔너리
        """
        data = np.array(list(importance_dict.values())).reshape(-1, 1)
        transformed_data = self.scaler_importance.fit_transform(data)
        score_result = transformed_data.flatten().tolist()
        score_result = [round(i, 4) for i in score_result]
        
        normalized_dict = {}
        for key, new_value in zip(importance_dict.keys(), score_result):
            normalized_dict[key] = new_value
        
        return normalized_dict
    
    def normalize_satisfaction(self, satisfaction_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Satisfaction 점수 정규화 (-10~10)
        Day3 Cell 33-38 로직 참고
        
        Args:
            satisfaction_dict: Satisfaction 딕셔너리
        
        Returns:
            정규화된 Satisfaction 딕셔너리
        """
        data = np.array(list(satisfaction_dict.values())).reshape(-1, 1)
        transformed_data = self.scaler_satisfaction.fit_transform(data)
        score_result = transformed_data.flatten().tolist()
        score_result = [round(i, 4) for i in score_result]
        
        normalized_dict = {}
        for key, new_value in zip(satisfaction_dict.keys(), score_result):
            normalized_dict[key] = new_value
        
        return normalized_dict
    
    def calculate_opportunity_score(self, satisfaction: float, importance: float) -> float:
        """
        Opportunity 점수 계산
        Day3 Cell 54-56 로직 참고
        Opportunity = Importance + Max(Importance - Satisfaction, 0)
        
        Args:
            satisfaction: Satisfaction 점수
            importance: Importance 점수
        
        Returns:
            Opportunity 점수
        """
        result = importance + max(importance - satisfaction, 0)
        return round(result, 4)
    
    def create_opportunity_dataframe(self, 
                                    importance_dict: Dict[str, float],
                                    satisfaction_dict: Dict[str, float]) -> pd.DataFrame:
        """
        Opportunity 데이터프레임 생성
        Day3 Cell 39-56 로직 참고
        
        Args:
            importance_dict: Importance 딕셔너리
            satisfaction_dict: Satisfaction 딕셔너리
        
        Returns:
            Opportunity 데이터프레임
        """
        # 데이터프레임 생성
        df = pd.DataFrame({
            'Action': list(importance_dict.keys()),
            'importance': list(importance_dict.values()),
            'satisfaction': [satisfaction_dict.get(action, 0) for action in importance_dict.keys()]
        })
        
        # Opportunity 점수 계산
        opportunity_list = []
        for _, row in df.iterrows():
            score = self.calculate_opportunity_score(row['satisfaction'], row['importance'])
            opportunity_list.append(score)
        
        df['opportunity_score'] = opportunity_list
        
        return df
    
    def plot_opportunity_area(self, 
                              df: pd.DataFrame,
                              figsize: Tuple[int, int] = (17, 10),
                              save_path: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        """
        기회영역 액션맵 시각화
        Day3 Cell 59-61 로직 참고
        
        Args:
            df: Opportunity 데이터프레임 (Action, importance, satisfaction, opportunity_score 컬럼)
            figsize: 그림 크기
            save_path: 저장 경로
            show: 화면에 표시 여부
        
        Returns:
            matplotlib Figure 객체
        """
        actions = df['Action'].values
        importance = df['importance'].values
        satisfaction = df['satisfaction'].values
        
        # 색상 생성
        colors = np.random.rand(len(actions), 3)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Action별 점 찍기
        for i, action in enumerate(actions):
            ax.scatter(importance[i], satisfaction[i], 
                     c=[colors[i]], label=action, s=50, edgecolors='black')
        
        # 범례
        ax.legend(title='Actions', fontsize=8, title_fontsize=10, 
                 loc='best', bbox_to_anchor=(1, 1))
        
        # 축 타이틀
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Satisfaction', fontsize=12)
        
        # 만족도 기준선
        if len(satisfaction) > 0:
            satisfaction_mean = satisfaction.mean()
            xdata = [0, 10]
            ydata = [satisfaction_mean, 10]
            ax.plot(xdata, ydata, 'k-', linewidth=1, linestyle='--', alpha=0.5)
        
        # 중요도 기준선
        if len(importance) > 0:
            importance_mean = importance.mean()
            x_data = [importance_mean, 10]
            y_data = [-10, 10]
            ax.plot(x_data, y_data, 'k-', linewidth=1, linestyle='--', alpha=0.5)
        
        # 포인트에 텍스트 추가
        texts = []
        for i, action in enumerate(actions):
            texts.append(ax.text(importance[i], satisfaction[i], action, 
                               fontsize=13, ha='left'))
        
        # 텍스트 자동 조정 (adjustText 사용 가능한 경우)
        if HAS_ADJUST_TEXT:
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey', lw=1))
        else:
            logger.warning("adjustText가 없어 텍스트 레이블이 겹칠 수 있습니다.")
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-10.5, 10.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"기회영역 액션맵 저장: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def display_importance_map(self, 
                              cluster_labels: List[int],
                              action_labels: List[int],
                              satisfaction_scores: Optional[List[float]] = None,
                              save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Importance 맵 생성 및 표시
        전체 파이프라인: Importance 계산 -> 정규화 -> Opportunity 계산 -> 시각화
        
        Args:
            cluster_labels: 클러스터(액터) 레이블 리스트
            action_labels: 액션 레이블 리스트
            satisfaction_scores: Satisfaction 점수 리스트 (None이면 감정분석 점수 사용)
            save_path: 저장 경로
        
        Returns:
            Opportunity 데이터프레임
        """
        # 1. Importance 계산
        importance_dict = self.calculate_importance(cluster_labels, action_labels)
        
        # 2. Importance 정규화
        normalized_importance = self.normalize_importance(importance_dict)
        
        # 3. Satisfaction 딕셔너리 생성
        if satisfaction_scores is None:
            # 기본값으로 0 설정 (실제로는 감정분석 결과를 사용해야 함)
            satisfaction_dict = {action: 0.0 for action in normalized_importance.keys()}
            logger.warning("Satisfaction 점수가 제공되지 않아 기본값 0을 사용합니다.")
        else:
            # Satisfaction 딕셔너리 생성
            action_flags = [
                f'Actor{actor}_Action{action}' 
                for actor, action in zip(cluster_labels, action_labels)
            ]
            # 평균 계산
            satisfaction_dict = {}
            for action in normalized_importance.keys():
                indices = [i for i, flag in enumerate(action_flags) if flag == action]
                if indices:
                    avg_satisfaction = np.mean([satisfaction_scores[i] for i in indices])
                    satisfaction_dict[action] = avg_satisfaction
                else:
                    satisfaction_dict[action] = 0.0
        
        # 4. Satisfaction 정규화
        normalized_satisfaction = self.normalize_satisfaction(satisfaction_dict)
        
        # 5. Opportunity 데이터프레임 생성
        opportunity_df = self.create_opportunity_dataframe(
            normalized_importance, normalized_satisfaction
        )
        
        # 6. 시각화
        if save_path:
            plot_path = save_path.replace('.csv', '_plot.png')
        else:
            plot_path = None
        
        self.plot_opportunity_area(opportunity_df, save_path=plot_path, show=False)
        
        return opportunity_df

