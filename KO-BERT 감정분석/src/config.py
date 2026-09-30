"""
설정 파일: 경로, 하이퍼파라미터, 모델 이름 등 관리
"""
import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

# 체크포인트 경로
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# KoBERT 모델 설정
MODEL_NAME = "monologg/kobert"  # 또는 "skt/kobert-base-v1"
MAX_LENGTH = 128  # 토큰 최대 길이
HIDDEN_DIM = 768  # KoBERT 기본 hidden dimension

# 학습 하이퍼파라미터
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = 100
SAVE_STEPS = 500
LOGGING_STEPS = 50

# 데이터 설정
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# 레이블 매핑 (필요에 따라 수정)
# 예시: 3-class 감정 분류
LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# 시드 고정
RANDOM_SEED = 42

# 디렉토리 생성
for dir_path in [DATA_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

