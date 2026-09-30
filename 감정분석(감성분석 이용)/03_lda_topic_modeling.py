"""
03_lda_topic_modeling.py
LDA 토픽 모델링

기능:
- 각 클러스터별 LDA 토픽 모델링
- Perplexity와 Coherence를 통한 최적 토픽 수 결정
- LDAvis 시각화
"""

import pandas as pd
import pickle
from tqdm import tqdm
import warnings
import gensim
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
import os

warnings.filterwarnings('ignore')

# 클러스터링 결과 로드
print("클러스터링 결과 로드 중...")
with open('clustering_result.pkl', 'rb') as f:
    df = pickle.load(f)

# 형태소 분리 결과를 리스트로 변환
import ast
if isinstance(df['tagged_review'].iloc[0], str):
    df['tagged_review'] = df['tagged_review'].apply(lambda x: ast.literal_eval(x))

print(f"전체 데이터 개수: {len(df)}")
print(f"클러스터 개수: {df['cluster'].nunique()}")
print(f"클러스터 분포:")
print(df['cluster'].value_counts().sort_index())

# ============================================
# 각 클러스터별로 LDA 모델링 수행
# ============================================

clusters = sorted(df['cluster'].unique())

for cluster_num in clusters:
    print(f"\n{'='*60}")
    print(f"클러스터 {cluster_num} 처리 중...")
    print(f"{'='*60}")
    
    df_cluster = df[df['cluster'] == cluster_num].copy()
    print(f"클러스터 {cluster_num} 데이터 개수: {len(df_cluster)}")
    
    # ============================================
    # 1. LDA를 위한 데이터 전처리
    # ============================================
    
    # 1.1. 전체 단어의 사전 만들고 각 문서에 매칭하기
    print("\n1.1 단어 사전 생성 중...")
    all_documents = list(df_cluster['tagged_review'])
    dictionary = Dictionary(all_documents)
    
    # 빈도가 너무 낮거나 높은 단어 제거
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    
    corpus = []
    for doc in all_documents:
        corpus.append(dictionary.doc2bow(doc))
    
    print(f"사전 크기: {len(dictionary)}")
    print(f"문서 개수: {len(corpus)}")
    
    # ============================================
    # 2. LDA 모델 만들기
    # ============================================
    
    # 2.1. LDA 기본 모델 만들기
    print("\n2.1 LDA 기본 모델 생성 중...")
    topic_num = 3  # 초기 토픽 수
    
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus,
        num_topics=topic_num,
        id2word=dictionary,
        passes=20,
        iterations=50,
        random_state=42
    )
    
    print(f"\n토픽 {topic_num}개로 생성된 모델의 상위 단어:")
    for topic_id, topic_desc in ldamodel.print_topics(num_words=5):
        print(f"Topic {topic_id}: {topic_desc}")
    
    # 2.2. LDA 토픽 수 선정 (Perplexity & Coherence)
    print("\n2.2 최적 토픽 수 탐색 중...")
    
    # Perplexity 계산
    print("Perplexity 계산 중...")
    perplexity_values = []
    for i in range(2, 10):
        ldamodel_temp = gensim.models.ldamodel.LdaModel(
            corpus, 
            num_topics=i, 
            id2word=dictionary,
            passes=10,
            iterations=30,
            random_state=42
        )
        perplexity_v = ldamodel_temp.log_perplexity(corpus)
        perplexity_values.append(perplexity_v)
    
    # Coherence 계산
    print("Coherence 계산 중...")
    coherence_values = []
    top_n = 3
    
    for i in tqdm(range(2, 10)):
        ldamodel_temp = gensim.models.ldamodel.LdaModel(
            corpus, 
            num_topics=i, 
            id2word=dictionary,
            passes=10,
            iterations=30,
            random_state=42
        )
        coherence_model = CoherenceModel(
            model=ldamodel_temp,
            texts=all_documents,
            dictionary=dictionary,
            topn=top_n
        )
        coherence_v = coherence_model.get_coherence()
        coherence_values.append(coherence_v)
    
    # 그래프 그리기
    x = range(2, 10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(x, perplexity_values, marker='o')
    ax1.set_xlabel('Number of Topics')
    ax1.set_ylabel('Perplexity Score')
    ax1.set_title(f'Cluster {cluster_num} - Perplexity')
    ax1.grid(True)
    
    ax2.plot(x, coherence_values, marker='o', color='orange')
    ax2.set_xlabel('Number of Topics')
    ax2.set_ylabel('Coherence Score')
    ax2.set_title(f'Cluster {cluster_num} - Coherence')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'cluster{cluster_num}_topic_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 최적 토픽 수 선택 (Coherence가 가장 높은 값)
    optimal_topics = x[coherence_values.index(max(coherence_values))]
    print(f"\n최적 토픽 수: {optimal_topics} (Coherence: {max(coherence_values):.4f})")
    
    # 최적 토픽 수로 최종 모델 생성
    print(f"\n2.3 최적 토픽 수({optimal_topics})로 최종 모델 생성 중...")
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus,
        num_topics=optimal_topics,
        id2word=dictionary,
        passes=20,
        iterations=50,
        random_state=42
    )
    
    print(f"\n최종 토픽 모델:")
    topics = ldamodel.show_topics(num_topics=-1, formatted=True)
    for topic_id, topic_desc in topics:
        print(f"Topic {topic_id}: {topic_desc}")
    
    # 2.3. Action 넘버 매칭
    print("\n2.4 문서별 토픽 할당 중...")
    action_align = []
    
    for doc in tqdm(ldamodel.get_document_topics(corpus)):
        label = []
        value = []
        for score in doc:
            label.append(score[0])
            value.append(score[1])
        max_index = np.argmax(value)
        action_n = label[max_index]
        action_align.append(action_n)
    
    df_cluster['action_cluster'] = action_align
    print(f"\n액션 클러스터 분포:")
    print(df_cluster['action_cluster'].value_counts().sort_index())
    
    # ============================================
    # 3. LDA 시각화 (LDAvis)
    # ============================================
    
    try:
        print("\n3. LDAvis 시각화 생성 중...")
        import pyLDAvis.gensim_models as gensimvis
        import pyLDAvis
        
        prepared_data = gensimvis.prepare(ldamodel, corpus, dictionary)
        pyLDAvis.save_html(prepared_data, f'ldavis_cluster{cluster_num}.html')
        print(f"LDAvis가 'ldavis_cluster{cluster_num}.html'에 저장되었습니다.")
    except Exception as e:
        print(f"LDAvis 생성 중 오류 발생: {e}")
    
    # 결과 저장
    df_cluster_result = df_cluster[['Review', 'cluster', 'tagged_review', 'action_cluster']].copy()
    df_cluster_result.to_csv(f'Cluster{cluster_num}.csv', encoding='utf-8-sig', index=False)
    
    with open(f'Cluster{cluster_num}.pkl', 'wb') as f:
        pickle.dump(df_cluster_result, f)
    
    print(f"\n클러스터 {cluster_num} 결과가 'Cluster{cluster_num}.csv'와 'Cluster{cluster_num}.pkl'에 저장되었습니다.")

print(f"\n{'='*60}")
print("모든 클러스터의 LDA 토픽 모델링이 완료되었습니다!")
print(f"{'='*60}")

