# 텍스트 마이닝 환경 설정 가이드

이 가이드는 KcELECTRA 기반 감정분석 및 텍스트 마이닝 환경을 설정하는 방법을 설명합니다.

## 📋 목차

1. [가상 환경 생성](#1-가상-환경-생성)
2. [환경 활성화 및 필수 라이브러리 설치](#2-환경-활성화-및-필수-라이브러리-설치)
3. [사용 방법](#3-사용-방법)
4. [주요 기능](#4-주요-기능)

---

## 1. 가상 환경 생성

수업에서 다루는 TF-IDF, 클러스터링(scikit-learn), LDA(gensim) 등과 파이썬 3.10을 한 번에 설치합니다.

### Conda 환경 생성

```bash
conda create -n textmining python=3.10 gensim numpy scipy pandas scikit-learn matplotlib seaborn -c conda-forge
```

> **참고**: Jupyter가 빠진 것을 확인하세요. 필요시 별도로 설치할 수 있습니다.

---

## 2. 환경 활성화 및 필수 라이브러리 설치

### 2.1 가상 환경 활성화

**스크립트 실행 전 항상 필요합니다:**

```bash
conda activate textmining
```

### 2.2 PyTorch 설치

KcELECTRA의 필수 의존성입니다. **PC 환경에 맞는 옵션 1개만 선택하세요.**

#### [옵션 1] NVIDIA GPU가 있는 경우 (권장)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### [옵션 2] CPU만 사용하는 경우

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### 2.3 KcELECTRA 및 기타 수업 라이브러리 설치

pip를 사용하여 설치합니다:

```bash
pip install transformers accelerate konlpy kiwipiepy nltk squarify wordcloud openpyxl
```

**설치되는 라이브러리 설명:**

- `transformers`, `accelerate`: KcELECTRA 구동용
- `konlpy`, `kiwipiepy`, `nltk`: 형태소 분석기
- `squarify`, `wordcloud`: 시각화
- `openpyxl`: 엑셀 파일 로드용

### 2.4 프로젝트 의존성 설치

프로젝트 루트 디렉토리에서:

```bash
pip install -r requirements.txt
```

---

## 3. 사용 방법

### 3.1 기본 사용법

CSV 파일을 입력받아 전체 파이프라인을 실행합니다:

```bash
python text_mining_pipeline.py --input data.csv --text_column text
```

### 3.2 주요 옵션

```bash
python text_mining_pipeline.py \
    --input data.csv \                    # 입력 CSV 파일
    --text_column text \                  # 텍스트 컬럼명 (기본값: text)
    --id_column id \                      # ID 컬럼명 (선택적)
    --model beomi/KcELECTRA-base-v2022 \ # 모델 이름
    --morph_analyzer kiwi \               # 형태소 분석기 (kiwi, kkma, komoran, mecab, okt)
    --output_dir output \                 # 결과 저장 디렉토리
    --n_clusters 5 \                      # 클러스터 수 (None이면 자동 결정)
    --clustering_method kmeans            # 클러스터링 방법 (kmeans, hierarchical)
```

### 3.3 Python 코드에서 사용

```python
from text_mining_pipeline import TextMiningPipeline

# 파이프라인 초기화
pipeline = TextMiningPipeline(
    model_name="beomi/KcELECTRA-base-v2022",
    morph_analyzer="kiwi",
    output_dir="output"
)

# 전체 파이프라인 실행
result_df = pipeline.run_full_pipeline(
    csv_path="data.csv",
    text_column="text",
    n_clusters=5,
    clustering_method="kmeans"
)

print(result_df.head())
```

### 3.4 개별 모듈 사용

#### 감정분석

```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.predict("이 영화 정말 재미있어요!")
print(result)  # (1, {'label_0': 0.1, 'label_1': 0.9})
```

#### 형태소 분석

```python
from morphological_analysis import MorphologicalAnalyzer

morph = MorphologicalAnalyzer(analyzer_type="kiwi")
keywords = morph.extract_keywords("오늘 날씨가 정말 좋네요")
print(keywords)  # ['오늘', '날씨', '좋']
```

#### TF-IDF 분석

```python
from tfidf_analysis import TFIDFAnalyzer

tfidf = TFIDFAnalyzer()
matrix = tfidf.fit_transform(documents)
top_features = tfidf.get_top_features(n=20)
```

#### 클러스터링

```python
from clustering import DocumentClustering

clusterer = DocumentClustering(n_clusters=5, method="kmeans")
clusterer.fit(tfidf_matrix)
labels = clusterer.labels_
```

#### 덴드로그램

```python
from dendrogram import DendrogramVisualizer

visualizer = DendrogramVisualizer(linkage_method="ward")
visualizer.plot_dendrogram(tfidf_matrix, save_path="dendrogram.png")
```

---

## 4. 주요 기능

### 4.1 감정분석

- KcELECTRA 모델을 사용한 감정 분류
- 배치 처리 지원
- 신뢰도 점수 제공

### 4.2 형태소 분석

- 다중 형태소 분석기 지원 (Kiwi, Kkma, Komoran, Mecab, Okt)
- 키워드 추출
- 명사/동사/형용사 필터링

### 4.3 빈도분석 & TF-IDF

- 단어 빈도 분석
- 문서 빈도 (DF) 계산
- TF-IDF 행렬 생성
- 상위 특성 추출

### 4.4 문서 클러스터링

- K-means 클러스터링
- 계층적 클러스터링
- 최적 클러스터 수 자동 탐색
- 클러스터링 성능 평가

### 4.5 덴드로그램

- 계층적 클러스터링 시각화
- 최적 클러스터 수 탐색
- 고해상도 이미지 저장

---

## 5. 출력 파일

파이프라인 실행 후 `output` 디렉토리에 다음 파일들이 생성됩니다:

- `final_results.csv`: 통합 분석 결과
- `sentiment_analysis.csv`: 감정분석 결과
- `morphological_analysis.csv`: 형태소 분석 결과
- `top_tfidf_features.csv`: 상위 TF-IDF 특성
- `word_frequency.csv`: 단어 빈도
- `clustering_metrics.csv`: 클러스터링 성능 지표
- `dendrogram.png`: 덴드로그램 이미지

---

## 6. 문제 해결

### 형태소 분석기 오류

**문제**: `konlpy` 또는 `kiwipiepy` 설치 오류

**해결**:
```bash
# Java 설치 확인 (konlpy 필요)
# Windows: https://www.oracle.com/java/technologies/downloads/
# Mac: brew install openjdk
# Linux: sudo apt-get install default-jdk

# Kiwi 설치 (권장)
pip install kiwipiepy
```

### GPU 메모리 부족

**문제**: CUDA out of memory

**해결**: 배치 크기 줄이기
```python
pipeline.sentiment_analyzer.predict_batch(texts, batch_size=16)
```

### 한글 폰트 오류

**문제**: 덴드로그램에서 한글이 깨짐

**해결**: Windows의 경우 `Malgun Gothic` 폰트가 자동으로 설정됩니다.
다른 OS의 경우 `dendrogram.py`의 폰트 설정을 수정하세요.

---

## 7. 참고 자료

- [KcELECTRA 모델](https://huggingface.co/beomi/KcELECTRA-base-v2022)
- [Transformers 문서](https://huggingface.co/docs/transformers)
- [scikit-learn 문서](https://scikit-learn.org/)
- [Kiwi 형태소 분석기](https://github.com/bab2min/kiwipiepy)

---

## 8. 라이선스

이 프로젝트는 원본 KcBERT-Finetune 프로젝트를 기반으로 합니다.

