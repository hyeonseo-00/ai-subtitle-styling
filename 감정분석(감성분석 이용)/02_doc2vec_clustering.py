"""
02_doc2vec_clustering.py
Doc2Vec 벡터화 및 계층적 클러스터링

기능:
- Doc2Vec을 사용한 문서 벡터화
- 병합 계층적 클러스터링 (AgglomerativeClustering, ward linkage)
- 실루엣 지수를 통한 최적 클러스터 수 결정
- TF-IDF를 사용한 클러스터 해석
"""

import pandas as pd
import pickle
from tqdm import tqdm
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.cluster import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# 전처리된 데이터 로드
print("전처리된 데이터 로드 중...")
df = pd.read_csv('preprocessed_data.csv')

# 형태소 분리 결과를 리스트로 변환 (CSV에서 로드 시 문자열로 저장되므로)
import ast
if isinstance(df['tagged_review'].iloc[0], str):
    df['tagged_review'] = df['tagged_review'].apply(lambda x: ast.literal_eval(x))

print(f"데이터 개수: {len(df)}")

# ============================================
# 3. 벡터화 (Doc2Vec)
# ============================================

# 3.1. doc2vec 준비
print("\n3.1 Doc2Vec 준비 중...")
tagged_corpus_list = []

for n, i in enumerate(df['tagged_review']):
    tag = "document{}".format(n)
    tagged_corpus_list.append(TaggedDocument(tags=[tag], words=i))

print(f"태그된 문서 개수: {len(tagged_corpus_list)}")

# 3.2. doc2vec 학습
print("\n3.2 Doc2Vec 모델 학습 중...")
model_doc2vec = Doc2Vec(
    vector_size=200,
    alpha=0.025,
    min_alpha=0.01,
    window=3,
    min_count=1,
    dm=1
)

# 단어 사전 장착
model_doc2vec.build_vocab(tagged_corpus_list)

# 학습
model_doc2vec.train(
    tagged_corpus_list, 
    total_examples=model_doc2vec.corpus_count, 
    epochs=100
)

print("Doc2Vec 학습 완료!")

# 3.3. 벡터 값 데이터 프레임에 추가
print("\n3.3 벡터 추출 중...")
vector_list = []
for i in range(len(df)):
    doc2vec = model_doc2vec.dv["document{}".format(i)]
    vector_list.append(doc2vec)

df['vector'] = vector_list
print("벡터화 완료!")

# ============================================
# 4. 병합 계층적 클러스터링
# ============================================

# 4.1. ward 기준으로 덴드로그램 그리기
print("\n4.1 덴드로그램 생성 중...")
model_linkage = linkage(list(df['vector']), 'ward')

plt.figure(figsize=(10, 5))
dendrogram(
    model_linkage,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=False
)
plt.title('Dendrogram (Ward Linkage)')
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("덴드로그램이 'dendrogram.png'에 저장되었습니다.")

# 4.2. 실루엣 지수 확인해서 토픽 갯수 정하기
print("\n4.2 실루엣 지수 계산 중...")
n_cluster = []
clustering_score = []

for i in tqdm(range(2, 15)):
    cluster_model = AgglomerativeClustering(n_clusters=i, linkage='ward')
    cluster_label = cluster_model.fit_predict(list(df['vector']))
    score = silhouette_score(list(df['vector']), cluster_label)
    n_cluster.append(i)
    clustering_score.append(score)

# 실루엣 지수 그래프
plt.figure(figsize=(10, 6))
plt.plot(n_cluster, clustering_score, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of Clusters')
plt.grid(True)
plt.savefig('silhouette_score.png', dpi=300, bbox_inches='tight')
plt.close()

result = pd.DataFrame({'n_cluster': n_cluster, 'score': clustering_score})
print("\n실루엣 지수 결과:")
print(result)
print(f"\n최고 실루엣 지수: {max(clustering_score)} (클러스터 수: {n_cluster[clustering_score.index(max(clustering_score))]})")

# 4.3. 가장 적절한 클러스터링 갯수 df에 삽입
# 최적 클러스터 수 선택 (실루엣 지수가 가장 높은 값)
optimal_clusters = n_cluster[clustering_score.index(max(clustering_score))]
print(f"\n4.3 최적 클러스터 수({optimal_clusters})로 클러스터링 실행 중...")

cluster_model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
cluster_label = cluster_model.fit_predict(list(df['vector']))
df['cluster'] = cluster_label

print(f"클러스터링 완료! 클러스터 분포:")
print(df['cluster'].value_counts().sort_index())

# ============================================
# 5. 해석하기: TF-IDF
# ============================================

print("\n5. TF-IDF 계산 중...")
all_document = []

for i in df['cluster'].unique():
    pos_tagging = df[df['cluster'] == i]['tagged_review']
    
    document = ''
    for pos in pos_tagging:
        doc = ' '.join(pos) + ' '
        document += doc
    all_document.append(document)

# TF-IDF 벡터라이저
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_document)

# 키워드 도출
feature_name = vectorizer.get_feature_names_out()
tfidf_value = tfidf_matrix.toarray()

# 데이터프레임으로 변환
tfidf_df = pd.DataFrame(tfidf_value, columns=feature_name)
tfidf_df.index = df['cluster'].unique()
tfidf_df_T = tfidf_df.T

# 각 클러스터별 TF-IDF 상위 단어 저장
for i in tfidf_df_T.columns:
    tfidfvalue = tfidf_df_T[i].sort_values(ascending=False)
    data = {'tfidf_word': tfidfvalue.index, 'tfidf': tfidfvalue.values}
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'cluster{i}_tf_idf.csv', encoding='utf-8-sig', index=False)

print("TF-IDF 결과가 각 클러스터별로 저장되었습니다.")

# 결과 저장
with open('clustering_result.pkl', 'wb') as f:
    pickle.dump(df, f)

print("\n클러스터링 결과가 'clustering_result.pkl'에 저장되었습니다.")
print(f"\n최종 데이터프레임:")
print(df[['Review', 'cluster']].head(10))

