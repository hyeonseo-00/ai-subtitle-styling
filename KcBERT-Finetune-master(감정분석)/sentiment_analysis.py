"""
KcELECTRA 기반 감정분석 모듈
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """KcELECTRA 모델을 사용한 감정분석 클래스"""
    
    def __init__(self, model_name: str = "beomi/KcELECTRA-base-v2022", 
                 device: str = None, num_labels: int = 2):
        """
        Args:
            model_name: HuggingFace 모델 이름 또는 로컬 경로
            device: 'cuda' 또는 'cpu' (None이면 자동 선택)
            num_labels: 감정 레이블 수 (기본값: 2 = 긍정/부정)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Loading model from {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def predict(self, text: str, return_probs: bool = False) -> Union[int, Tuple[int, Dict[str, float]]]:
        """
        단일 텍스트에 대한 감정 예측
        
        Args:
            text: 분석할 텍스트
            return_probs: True면 확률값도 반환
            
        Returns:
            예측 레이블 (int) 또는 (레이블, 확률 딕셔너리) 튜플
        """
        return self.predict_batch([text], return_probs)[0]
    
    def predict_batch(self, texts: List[str], return_probs: bool = False, 
                     batch_size: int = 32) -> Union[List[int], List[Tuple[int, Dict[str, float]]]]:
        """
        여러 텍스트에 대한 감정 예측 (배치 처리)
        
        Args:
            texts: 분석할 텍스트 리스트
            return_probs: True면 확률값도 반환
            batch_size: 배치 크기
            
        Returns:
            예측 레이블 리스트 또는 (레이블, 확률 딕셔너리) 튜플 리스트
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 토크나이징
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # GPU로 이동
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 예측
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # CPU로 이동하여 numpy로 변환
            predictions = predictions.cpu().numpy()
            probs = probs.cpu().numpy()
            
            # 결과 처리
            for j, pred in enumerate(predictions):
                if return_probs:
                    prob_dict = {f"label_{k}": float(probs[j][k]) for k in range(self.num_labels)}
                    results.append((int(pred), prob_dict))
                else:
                    results.append(int(pred))
        
        return results
    
    def predict_with_confidence(self, text: str) -> Dict[str, Union[int, float]]:
        """
        신뢰도 점수와 함께 예측
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            {'label': 예측 레이블, 'confidence': 신뢰도 점수, 'probabilities': 확률 딕셔너리}
        """
        pred, probs = self.predict(text, return_probs=True)
        confidence = max(probs.values())
        
        return {
            'label': pred,
            'confidence': confidence,
            'probabilities': probs
        }
    
    def fine_tune_for_sentiment(self, train_texts: List[str], train_labels: List[int],
                                num_epochs: int = 3, learning_rate: float = 2e-5):
        """
        감정분석을 위한 파인튜닝 (선택적)
        
        Args:
            train_texts: 학습 텍스트 리스트
            train_labels: 학습 레이블 리스트
            num_epochs: 에폭 수
            learning_rate: 학습률
        """
        # 파인튜닝 로직은 필요시 구현
        logger.warning("Fine-tuning method not implemented. Use run_seq_cls.py for training.")
        pass

