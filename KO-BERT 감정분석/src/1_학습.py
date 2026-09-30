"""
KoBERT 감정 분석 모델 학습 스크립트
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import numpy as np
from tqdm import tqdm
import logging
import os

from src.config import *
from src.data_utils import load_dataset, create_dataloader
from src.model import KoBERTEmotionClassifier
from src.utils import set_seed, compute_metrics, print_classification_report, plot_confusion_matrix

logger = logging.getLogger(__name__)


def train_with_trainer(train_df, val_df, output_dir: str = None):
    """
    Hugging Face Trainer를 사용한 학습
    
    Args:
        train_df: 학습 데이터 DataFrame
        val_df: 검증 데이터 DataFrame
        output_dir: 출력 디렉토리
    """
    output_dir = output_dir or str(BEST_MODEL_DIR)
    
    # 시드 고정
    set_seed(RANDOM_SEED)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    model = KoBERTEmotionClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 데이터셋 생성
    train_dataset = create_dataloader(train_df, tokenizer, BATCH_SIZE, shuffle=True, is_training=True).dataset
    val_dataset = create_dataloader(val_df, tokenizer, BATCH_SIZE, shuffle=False, is_training=True).dataset
    
    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),  # GPU가 있으면 mixed precision 사용
    )
    
    # 메트릭 계산 함수
    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return compute_metrics(predictions, labels, ID2LABEL)
    
    # Trainer 생성
    trainer = Trainer(
        model=model.model,  # 내부 모델 사용
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 학습 시작
    logger.info("학습을 시작합니다...")
    trainer.train()
    
    # 최종 평가
    logger.info("최종 평가를 수행합니다...")
    eval_results = trainer.evaluate()
    logger.info(f"최종 평가 결과: {eval_results}")
    
    # 모델 저장
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"모델이 {output_dir}에 저장되었습니다.")
    
    # 검증 데이터로 예측 및 리포트 생성
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    print_classification_report(pred_labels, true_labels, ID2LABEL)
    plot_confusion_matrix(pred_labels, true_labels, ID2LABEL, 
                         save_path=str(OUTPUT_DIR / "confusion_matrix.png"))


def train_manual(train_df, val_df, output_dir: str = None):
    """
    수동 학습 루프 (Trainer 대신 직접 구현)
    
    Args:
        train_df: 학습 데이터 DataFrame
        val_df: 검증 데이터 DataFrame
        output_dir: 출력 디렉토리
    """
    output_dir = output_dir or str(BEST_MODEL_DIR)
    
    # 시드 고정
    set_seed(RANDOM_SEED)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    model = KoBERTEmotionClassifier()
    model.to(device)
    
    # 옵티마이저 및 스케줄러
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # DataLoader 생성
    train_loader = create_dataloader(train_df, tokenizer, BATCH_SIZE, shuffle=True, is_training=True)
    val_loader = create_dataloader(val_df, tokenizer, BATCH_SIZE, shuffle=False, is_training=True)
    
    best_f1 = 0.0
    
    # 학습 루프
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="학습 중"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"평균 학습 손실: {avg_train_loss:.4f}")
        
        # 검증
        model.eval()
        val_predictions = []
        val_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="검증 중"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(np.array(val_predictions), np.array(val_labels), ID2LABEL)
        
        logger.info(f"평균 검증 손실: {avg_val_loss:.4f}")
        logger.info(f"검증 메트릭: {metrics}")
        
        # 최고 성능 모델 저장
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            model.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"최고 성능 모델 저장 (F1-macro: {best_f1:.4f})")
    
    logger.info(f"\n학습 완료! 최고 F1-macro: {best_f1:.4f}")
    logger.info(f"모델이 {output_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="KoBERT 감정 분석 모델 학습")
    parser.add_argument("--train_csv", type=str, default=str(TRAIN_CSV), help="학습 데이터 CSV 경로")
    parser.add_argument("--val_csv", type=str, default=str(VAL_CSV), help="검증 데이터 CSV 경로")
    parser.add_argument("--output_dir", type=str, default=str(BEST_MODEL_DIR), help="모델 저장 디렉토리")
    parser.add_argument("--use_trainer", action="store_true", help="Hugging Face Trainer 사용 여부")
    
    args = parser.parse_args()
    
    # 데이터 로드
    logger.info("데이터를 로드합니다...")
    train_df = load_dataset(args.train_csv)
    val_df = load_dataset(args.val_csv)
    
    logger.info(f"학습 데이터: {len(train_df)}개")
    logger.info(f"검증 데이터: {len(val_df)}개")
    
    # 학습 실행
    if args.use_trainer:
        train_with_trainer(train_df, val_df, args.output_dir)
    else:
        train_manual(train_df, val_df, args.output_dir)


if __name__ == "__main__":
    main()

