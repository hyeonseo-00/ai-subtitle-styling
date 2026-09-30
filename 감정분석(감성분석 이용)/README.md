# 감정분석 및 클러스터링 프로젝트

이 프로젝트는 한국어 텍스트 데이터를 기반으로 감정분석과 클러스터링을 수행하는 파이썬 스크립트 모음입니다.

## 프로젝트 구조

```
.
├── 01_data_preprocessing.py      # 데이터 전처리 및 형태소 분리
├── 02_doc2vec_clustering.py      # Doc2Vec 벡터화 및 계층적 클러스터링
├── 03_lda_topic_modeling.py      # LDA 토픽 모델링
├── 04_sentiment_analysis.py      # 감성 분석 및 Opportunity 분석
├── requirements.txt              # 필요한 패키지 목록
└── README.md                      # 이 파일
```

## 사용된 기술 및 방법론

### 1. 데이터 전처리 (01_data_preprocessing.py)
- **정규표현식**: 특수문자 및 숫자 제거
- **KoNLPy의 Okt**: 한국어 형태소 분석
- **불용어 제거**: ko-stopwords.csv 파일 사용

### 2. 벡터화 및 클러스터링 (02_doc2vec_clustering.py)
- **Doc2Vec**: 문서를 벡터로 변환 (Gensim 라이브러리)
  - vector_size: 200
  - window: 3
  - dm: 1 (PV-DM 모델)
- **계층적 클러스터링**: AgglomerativeClustering (ward linkage)
- **최적 클러스터 수 결정**: 실루엣 지수(Silhouette Score) 사용
- **TF-IDF**: 클러스터별 주요 키워드 추출

### 3. 토픽 모델링 (03_lda_topic_modeling.py)
- **LDA (Latent Dirichlet Allocation)**: 각 클러스터 내 세부 토픽 분석
- **최적 토픽 수 결정**: 
  - Perplexity (낮을수록 좋음)
  - Coherence (높을수록 좋음)
- **LDAvis**: 토픽 모델 시각화

### 4. 감성 분석 (04_sentiment_analysis.py)
- **감성 사전**: SentiWord_info.json 사용
- **Satisfaction 점수**: 감성 점수의 평균값을 -10~10 범위로 정규화
- **Importance 점수**: 액터-액션 조합의 빈도 기반 중요도 (0~10 범위로 정규화)
- **Opportunity 점수**: Importance + Max(Importance - Satisfaction, 0)
- **Opportunity Area 시각화**: Importance-Satisfaction 2D 그래프

## 필요한 파일

실행 전에 다음 파일들이 필요합니다:

1. `한글 크롤링.csv`: 원본 데이터 파일 (Review 컬럼 포함)
2. `ko-stopwords.csv`: 한국어 불용어 사전
3. `SentiWord_info.json`: 감성 사전 파일

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Java 설치 (KoNLPy 사용을 위해 필요):
   - Windows: https://www.oracle.com/java/technologies/downloads/
   - Mac: `brew install openjdk`
   - Linux: `sudo apt-get install default-jdk`

## 실행 순서

스크립트는 순서대로 실행해야 합니다:

```bash
# 1. 데이터 전처리 및 형태소 분리
python 01_data_preprocessing.py

# 2. Doc2Vec 벡터화 및 클러스터링
python 02_doc2vec_clustering.py

# 3. LDA 토픽 모델링
python 03_lda_topic_modeling.py

# 4. 감성 분석 및 Opportunity 분석
python 04_sentiment_analysis.py
```

## 출력 파일

### 01_data_preprocessing.py
- `preprocessed_data.csv`: 전처리된 데이터

### 02_doc2vec_clustering.py
- `clustering_result.pkl`: 클러스터링 결과
- `dendrogram.png`: 덴드로그램 시각화
- `silhouette_score.png`: 실루엣 지수 그래프
- `cluster{N}_tf_idf.csv`: 각 클러스터별 TF-IDF 상위 단어

### 03_lda_topic_modeling.py
- `Cluster{N}.csv`: 각 클러스터별 LDA 결과
- `Cluster{N}.pkl`: 각 클러스터별 LDA 결과 (pickle)
- `cluster{N}_topic_selection.png`: 토픽 수 선택 그래프
- `ldavis_cluster{N}.html`: LDAvis 시각화 (웹 브라우저에서 열기)

### 04_sentiment_analysis.py
- `opportunity.csv`: 최종 Satisfaction, Importance, Opportunity 점수
- `OpportunityArea.png`: Opportunity Area 시각화

## 주요 파라미터

### Doc2Vec
- `vector_size`: 200 (문서 벡터 크기)
- `window`: 3 (문맥 윈도우 크기)
- `alpha`: 0.025 (초기 학습률)
- `epochs`: 100 (학습 횟수)

### 클러스터링
- `linkage`: 'ward' (병합 기준)
- 클러스터 수: 실루엣 지수로 자동 결정 (2~14 범위에서 탐색)

### LDA
- `passes`: 20 (전체 코퍼스 반복 횟수)
- `iterations`: 50 (각 문서 반복 횟수)
- 토픽 수: Coherence로 자동 결정 (2~9 범위에서 탐색)

## 주의사항

1. **데이터 크기**: 대용량 데이터의 경우 실행 시간이 오래 걸릴 수 있습니다.
2. **메모리**: Doc2Vec 학습 시 충분한 메모리가 필요합니다.
3. **Java**: KoNLPy 사용을 위해 Java가 설치되어 있어야 합니다.
4. **파일 경로**: 스크립트는 현재 디렉토리에서 파일을 찾습니다.

## 문제 해결

### KoNLPy 오류
- Java가 설치되어 있는지 확인
- `pip install --upgrade konlpy` 실행

### Gensim 오류
- `pip install --upgrade scipy numpy gensim` 실행

### 메모리 부족
- 데이터를 샘플링하여 실행
- Doc2Vec의 vector_size를 줄이기

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

