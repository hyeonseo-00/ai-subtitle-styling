"""
KoBERT 감정 분석 추론 스크립트
"""
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import logging

from src.config import MODEL_NAME, ID2LABEL, MAX_LENGTH

logger = logging.getLogger(__name__)


class EmotionInference:
    """
    감정 분석 추론 클래스
    """
    
    def __init__(self, model_dir: str = None, model_name: str = MODEL_NAME):
        """
        Args:
            model_dir: 학습된 모델 디렉토리 (None이면 사전 학습 모델 사용)
            model_name: 기본 모델 이름
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_dir:
            logger.info(f"모델을 {model_dir}에서 로드합니다...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            logger.info(f"사전 학습 모델 {model_name}을 로드합니다...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        # 레이블 매핑
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            self.id2label = self.model.config.id2label
        else:
            self.id2label = ID2LABEL
        
        logger.info(f"모델이 {self.device}에 로드되었습니다.")
    
    def predict(self, text: str, return_probs: bool = True) -> Dict:
        """
        단일 문장 감정 예측
        
        Args:
            text: 입력 텍스트
            return_probs: 확률 반환 여부
        
        Returns:
            예측 결과 딕셔너리
        """
        return self.predict_batch([text], return_probs)[0]
    
    def predict_batch(self, texts: List[str], return_probs: bool = True) -> List[Dict]:
        """
        배치 문장 감정 예측
        
        Args:
            texts: 텍스트 리스트
            return_probs: 확률 반환 여부
        
        Returns:
            예측 결과 리스트
        """
        # 토크나이징
        encoded = self.tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # 결과 구성
        results = []
        for i in range(len(texts)):
            pred_id = predictions[i].item()
            pred_label = self.id2label.get(pred_id, f"label_{pred_id}")
            
            result = {
                'text': texts[i],
                'predicted_label': pred_label,
                'predicted_id': pred_id
            }
            
            if return_probs:
                prob_dict = {}
                for label_id, label_name in self.id2label.items():
                    prob_dict[label_name] = probs[i][label_id].item()
                result['probabilities'] = prob_dict
            
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="KoBERT 감정 분석 추론")
    parser.add_argument("--model_dir", type=str, default=None, 
                       help="학습된 모델 디렉토리 (None이면 사전 학습 모델 사용)")
    parser.add_argument("--text", type=str, default=None, help="예측할 단일 텍스트")
    parser.add_argument("--text_file", type=str, default=None, 
                       help="예측할 텍스트가 있는 파일 (한 줄에 하나씩)")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="결과를 저장할 파일 경로")
    
    args = parser.parse_args()
    
    # 추론 객체 생성
    inferencer = EmotionInference(model_dir=args.model_dir)
    
    # 텍스트 입력 처리
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # 대화형 모드
        print("감정 분석을 시작합니다. 텍스트를 입력하세요 (종료: 'quit' 또는 'exit'):")
        texts = []
        while True:
            text = input("> ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text.strip():
                texts.append(text.strip())
    
    if not texts:
        logger.error("입력 텍스트가 없습니다.")
        return
    
    # 예측 수행
    results = inferencer.predict_batch(texts, return_probs=True)
    
    # 결과 출력
    for result in results:
        print(f"\n텍스트: {result['text']}")
        print(f"예측 레이블: {result['predicted_label']}")
        if 'probabilities' in result:
            print("확률:")
            for label, prob in sorted(result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"  {label}: {prob:.4f}")
    
    # 파일 저장
    if args.output_file:
        import json
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"결과가 {args.output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()

