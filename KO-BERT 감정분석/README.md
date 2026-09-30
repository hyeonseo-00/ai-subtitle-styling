# KoBERT 기반 한국어 감정 분석 프로젝트

이 프로젝트는 Hugging Face의 KoBERT 모델을 활용하여 한국어 텍스트의 감정을 분류하고, 문장을 벡터로 변환하는 파이프라인을 제공합니다.

## 프로젝트 개요

- **감정 분류**: KoBERT를 fine-tuning하여 한국어 텍스트를 감정 레이블(긍정/부정/중립 등)로 분류
- **문장 벡터화**: KoBERT를 사용하여 문장을 고정 길이 벡터로 변환 (클러스터링, 시각화, 추가 분석용)
- **TF-IDF 분석**: 텍스트의 중요 단어 추출 및 시각화
- **LDA 토픽 모델링**: 잠재 디리클레 할당을 통한 토픽 분석 및 시각화

## 사용 모델 및 라이브러리

- **모델**: `monologg/kobert` (Hugging Face)
- **주요 라이브러리**:
  - `transformers`: Hugging Face 모델 및 토크나이저
  - `torch`: PyTorch
  - `pandas`, `numpy`: 데이터 처리
  - `scikit-learn`: 메트릭 및 분석
  - `pyLDAvis`: LDA 시각화
  - `wordcloud`: 워드클라우드 생성

## 프로젝트 구조

```
.
├── data/                    # 데이터 디렉토리
│   ├── train.csv           # 학습 데이터
│   ├── val.csv             # 검증 데이터
│   └── test.csv            # 테스트 데이터 (선택)
├── checkpoints/            # 모델 체크포인트
│   └── best_model/         # 최고 성능 모델
├── outputs/                # 분석 결과 출력
├── src/                    # 소스 코드
│   ├── config.py          # 설정 파일
│   ├── utils.py           # 공통 유틸리티
│   ├── data_utils.py      # 데이터 로딩 및 전처리
│   ├── model.py           # KoBERT 모델 래퍼
│   ├── 1_학습.py          # 1단계: 모델 학습 스크립트
│   ├── 2_추론.py          # 2단계: 감정 예측 추론 스크립트
│   ├── 3_임베딩생성.py    # 3단계: 문장 벡터화
│   ├── 4_TFIDF분석.py     # 4단계: TF-IDF 중요도 분석
│   └── 5_LDA시각화.py     # 5단계: LDA 토픽 모델링 및 시각화
├── requirements.txt        # 패키지 의존성
└── README.md              # 이 파일
```

## 데이터 형식

CSV 파일은 다음 형식을 따라야 합니다:

```csv
text,label
이 제품이 정말 좋아요,positive
서비스가 별로예요,negative
그냥 평범해요,neutral
```

- `text`: 한국어 문장 (블로그, 카페, 지식인, 레딧 번역 등)
- `label`: 감정 레이블 (예: `positive`, `negative`, `neutral`)

레이블은 `src/config.py`의 `LABEL2ID`에서 수정할 수 있습니다.

## 설치 방법

1. 저장소 클론 또는 다운로드

2. 가상환경 생성 (권장):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 학습 (1단계)

학습 데이터와 검증 데이터를 `data/` 디렉토리에 준비한 후:

```bash
# Hugging Face Trainer 사용
python src/1_학습.py --train_csv data/train.csv --val_csv data/val.csv --use_trainer

# 수동 학습 루프 사용
python src/1_학습.py --train_csv data/train.csv --val_csv data/val.csv
```

학습된 모델은 `checkpoints/best_model/`에 저장됩니다.

### 2. 추론 (감정 예측) (2단계)

#### 단일 문장 예측:
```bash
python src/2_추론.py --model_dir checkpoints/best_model --text "이 제품 소리 알림이 너무 좋아요"
```

#### 파일에서 여러 문장 예측:
```bash
python src/2_추론.py --model_dir checkpoints/best_model --text_file input.txt --output_file results.json
```

#### 대화형 모드:
```bash
python src/2_추론.py --model_dir checkpoints/best_model
```

출력 예시:
```
텍스트: 이 제품 소리 알림이 너무 좋아요
예측 레이블: positive
확률:
  positive: 0.8523
  neutral: 0.1021
  negative: 0.0456
```

### 3. 문장 벡터화 (임베딩 생성) (3단계)

CSV 파일에서 문장 임베딩을 생성:

```bash
python src/3_임베딩생성.py --input_csv data/train.csv --output_npy data/train_embeddings.npy --text_column text
```

생성된 임베딩은 numpy 배열로 저장되며, 이후 클러스터링이나 시각화에 활용할 수 있습니다:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 임베딩 로드
embeddings = np.load('data/train_embeddings.npy')

# KMeans 클러스터링
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(embeddings)

# t-SNE 시각화
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')
plt.title('문장 임베딩 t-SNE 시각화')
plt.savefig('outputs/tsne_visualization.png')
```

### 4. TF-IDF 중요도 분석 (4단계)

```bash
python src/4_TFIDF분석.py --input_csv data/train.csv --output_dir outputs/tfidf --n_top_words 20
```

생성되는 파일:
- `tfidf_top_features.png`: 클래스별 상위 중요 단어 그래프
- `wordcloud.png`: 워드클라우드 이미지
- `tfidf_top_features.json`: 상위 단어 및 점수 JSON

### 5. LDA 토픽 모델링 및 시각화 (5단계)

```bash
python src/5_LDA시각화.py --input_csv data/train.csv --n_topics 5 --output_dir outputs/lda --n_words 10
```

생성되는 파일:
- `lda_visualization.html`: 인터랙티브 LDA 시각화 (브라우저에서 열기)
- `document_topics.csv`: 문서별 토픽 분포
- `lda_topics.json`: 토픽별 상위 단어

## 설정 변경

주요 설정은 `src/config.py`에서 수정할 수 있습니다:

- `MODEL_NAME`: 사용할 KoBERT 모델 (예: `"monologg/kobert"`, `"skt/kobert-base-v1"`)
- `MAX_LENGTH`: 토큰 최대 길이
- `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`: 학습 하이퍼파라미터
- `LABEL2ID`: 레이블 매핑

## 주의사항

1. **한글 폰트**: 워드클라우드 생성 시 Windows에서는 `malgun.ttf`를 사용합니다. 다른 OS에서는 적절한 한글 폰트 경로로 수정해야 합니다.

2. **GPU 사용**: CUDA가 설치되어 있으면 자동으로 GPU를 사용합니다.

3. **데이터 인코딩**: CSV 파일은 UTF-8 인코딩을 사용해야 합니다.

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 참고 자료

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [KoBERT 모델](https://huggingface.co/monologg/kobert)
- [PyTorch 문서](https://pytorch.org/docs/)

