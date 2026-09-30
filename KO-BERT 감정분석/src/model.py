"""
KoBERT 기반 감정 분석 모델 래퍼
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import Optional, Tuple
import logging

from src.config import MODEL_NAME, NUM_LABELS, ID2LABEL

logger = logging.getLogger(__name__)


class KoBERTEmotionClassifier(nn.Module):
    """
    KoBERT 기반 감정 분류 모델
    """
    
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS, 
                 dropout: float = 0.1):
        """
        Args:
            model_name: Hugging Face 모델 이름
            num_labels: 분류할 레이블 개수
            dropout: 드롭아웃 비율
        """
        super(KoBERTEmotionClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.id2label = ID2LABEL
        self.config.label2id = {v: k for k, v in ID2LABEL.items()}
        
        # 분류를 위한 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
        logger.info(f"모델 {model_name}을 로드했습니다. (레이블 수: {num_labels})")
    
    def forward(self, input_ids, attention_mask, labels: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            input_ids: 토큰 ID
            attention_mask: 어텐션 마스크
            labels: 레이블 (학습 시)
        
        Returns:
            loss (학습 시), logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs


class KoBERTEmbedder:
    """
    KoBERT를 사용한 문장 임베딩 추출 클래스
    """
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Args:
            model_name: Hugging Face 모델 이름
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드 (분류 헤드 없이)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"임베딩 모델 {model_name}을 {self.device}에 로드했습니다.")
    
    def get_embeddings(self, texts: list, batch_size: int = 32, 
                      pooling_method: str = 'cls') -> torch.Tensor:
        """
        문장 리스트에서 임베딩 추출
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            pooling_method: 'cls' (CLS 토큰) 또는 'mean' (평균 풀링)
        
        Returns:
            임베딩 텐서 (num_texts, hidden_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 토크나이징
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 모델 forward
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                
                # 풀링
                if pooling_method == 'cls':
                    # [CLS] 토큰 사용
                    batch_embeddings = hidden_states[:, 0, :]  # (batch_size, hidden_dim)
                elif pooling_method == 'mean':
                    # Attention mask 기준 평균 풀링
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                else:
                    raise ValueError(f"알 수 없는 pooling_method: {pooling_method}")
                
                embeddings.append(batch_embeddings.cpu())
        
        # 모든 배치 합치기
        all_embeddings = torch.cat(embeddings, dim=0)
        return all_embeddings
    
    def get_embeddings_numpy(self, texts: list, batch_size: int = 32, 
                            pooling_method: str = 'cls') -> 'np.ndarray':
        """
        문장 리스트에서 임베딩 추출 (numpy 배열 반환)
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            pooling_method: 'cls' 또는 'mean'
        
        Returns:
            임베딩 numpy 배열 (num_texts, hidden_dim)
        """
        import numpy as np
        embeddings_tensor = self.get_embeddings(texts, batch_size, pooling_method)
        return embeddings_tensor.numpy()

