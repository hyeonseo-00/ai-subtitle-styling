"""
데이터 로딩 및 전처리 유틸리티
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import AutoTokenizer
import logging

from src.config import MODEL_NAME, MAX_LENGTH, TEXT_COLUMN, LABEL_COLUMN, LABEL2ID

logger = logging.getLogger(__name__)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    CSV 파일에서 데이터셋 로드
    
    Args:
        csv_path: CSV 파일 경로
    
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"{csv_path}에서 {len(df)}개의 샘플을 로드했습니다.")
        return df
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {e}")
        raise


def encode_labels(labels: List[str], label2id: Dict[str, int]) -> np.ndarray:
    """
    텍스트 레이블을 숫자 ID로 인코딩
    
    Args:
        labels: 텍스트 레이블 리스트
        label2id: 레이블 to ID 매핑 딕셔너리
    
    Returns:
        인코딩된 레이블 배열
    """
    encoded = []
    for label in labels:
        if label in label2id:
            encoded.append(label2id[label])
        else:
            logger.warning(f"알 수 없는 레이블: {label}, neutral로 매핑합니다.")
            encoded.append(label2id.get("neutral", 1))
    return np.array(encoded)


class EmotionDataset(Dataset):
    """
    감정 분석을 위한 PyTorch Dataset 클래스
    """
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = MAX_LENGTH):
        """
        Args:
            texts: 텍스트 리스트
            labels: 레이블 리스트 (None이면 추론용)
            tokenizer: Hugging Face 토크나이저
            max_length: 최대 토큰 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = max_length
        
        # 토크나이저에 pad_token이 없으면 추가
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        샘플 하나를 반환
        
        Returns:
            input_ids, attention_mask, labels (있는 경우)를 포함한 딕셔너리
        """
        text = str(self.texts[idx])
        
        # 토크나이징
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten()
        }
        
        # 레이블이 있으면 추가 (학습용)
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return result


def create_dataloader(df: pd.DataFrame, tokenizer, batch_size: int = 16, 
                     shuffle: bool = True, is_training: bool = True):
    """
    DataLoader 생성 헬퍼 함수
    
    Args:
        df: pandas DataFrame
        tokenizer: 토크나이저
        batch_size: 배치 크기
        shuffle: 셔플 여부
        is_training: 학습용인지 (레이블 필요 여부)
    
    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import DataLoader
    
    texts = df[TEXT_COLUMN].tolist()
    
    if is_training:
        labels = encode_labels(df[LABEL_COLUMN].tolist(), LABEL2ID)
        dataset = EmotionDataset(texts, labels, tokenizer)
    else:
        dataset = EmotionDataset(texts, None, tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Windows 호환성을 위해 0으로 설정
    )
    
    return dataloader

