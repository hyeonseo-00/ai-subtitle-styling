"""
04_sentiment_analysis.py
감성 분석 및 Opportunity 분석

기능:
- SentiWord_info.json을 사용한 감성 분석
- Satisfaction 점수 계산 및 정규화
- Importance 점수 계산 및 정규화
- Opportunity 점수 계산 및 시각화
"""

import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
import warnings
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from adjustText import adjust_text
import glob
import ast

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================
# 데이터 로드
# ============================================

print("클러스터별 데이터 로드 중...")
pickle_file_path = glob.glob('Cluster*.pkl')

df = pd.DataFrame()

for p in pickle_file_path:
    with open(p, 'rb') as f:
        new_df = pickle.load(f)
    df = pd.concat([df, new_df])

df = df.reset_index(drop=True)

# 형태소 분리 결과를 리스트로 변환
if isinstance(df['tagged_review'].iloc[0], str):
    df['tagged_review'] = df['tagged_review'].apply(lambda x: ast.literal_eval(x))

print(f"전체 데이터 개수: {len(df)}")
print(f"클러스터 분포:")
print(df['cluster'].value_counts().sort_index())

# ============================================
# 1. Satisfaction (감성 분석)
# ============================================

# 1.1. 감성사전 불러오기
print("\n1.1 감성사전 로드 중...")
with open('SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
    sent_dicts = json.load(f)

print(f"감성사전 단어 개수: {len(sent_dicts)}")

sent_df = pd.DataFrame(sent_dicts)
print(f"\n감성 극성 분포:")
print(sent_df['polarity'].value_counts())

# 1.2. 감성점수 구하는 함수 만들기
def sentiment_score(sent_dicts, token_list):
    """토큰 리스트에서 감성 점수를 찾아 반환"""
    result_list = []
    for token in token_list:
        for s in sent_dicts:
            if s['word'] == token:
                result = (s['word'], s['polarity'])
                result_list.append(result)
    return result_list

# 1.3. 감성점수 구하기
print("\n1.3 감성점수 계산 중...")

# 1.3.1. 적합하게 형태소 재분리
from konlpy.tag import Okt
from kiwipiepy import Kiwi

okt = Okt()
kiwi = Kiwi()

def okt_pos_tagging(string):
    """형태소 분리 (Kiwi로 띄어쓰기 보정 후 Okt로 형태소 분석)"""
    string = kiwi.space(string)
    pos_words = okt.morphs(string, stem=True, norm=True)
    return pos_words

# 1.3.2. 감정점수 적용하기
print("형태소 분리 및 감성 점수 계산 중...")
sentiment = []

for i in tqdm(df['Review']):
    token = okt_pos_tagging(i)
    score = sentiment_score(sent_dicts, token)
    sentiment.append(score)

# 감성 점수 합계 계산
print("감성 점수 합계 계산 중...")
avg_sents = []

for sents in tqdm(sentiment):
    sent_score = sum([int(i[1]) for i in sents])
    avg_sents.append(sent_score)

df['sentiment_score'] = avg_sents
print(f"\n감성 점수 통계:")
print(df['sentiment_score'].describe())

# 1.4. 전체 액터와 액션에 대해서 감성점수 계산
print("\n1.4 액터-액션별 감성점수 계산 중...")

# 1.4.1. dict 형식으로 출력
sents_dict = dict()

for actor in df['cluster'].unique():
    actor_df = df[df['cluster'] == actor]
    
    for action in actor_df['action_cluster'].unique():
        action_scores = actor_df[actor_df['action_cluster'] == action]['sentiment_score']
        action_score = np.mean(action_scores)
        sents_dict[f'Actor{actor}_Action{action}'] = action_score

print(f"액터-액션 조합 개수: {len(sents_dict)}")
print(f"\n감성 점수 샘플:")
for key, value in list(sents_dict.items())[:5]:
    print(f"{key}: {value:.4f}")

# 1.4.2. 정규화 (-10~10)
print("\n1.4.2 감성 점수 정규화 중...")
data = sents_dict.values()
data = np.array(list(data)).reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-10, 10))
transformed_data = scaler.fit_transform(data)

score_result = transformed_data.flatten().tolist()
score_result = [round(i, 4) for i in score_result]

for key, new_value in zip(sents_dict.keys(), score_result):
    sents_dict[key] = new_value

sents_df = pd.DataFrame(sents_dict.items(), columns=['Action', 'satisfaction'])
print("정규화 완료!")

# ============================================
# 2. Importance
# ============================================

# 2.1. Importance 점수 구하기
print("\n2.1 Importance 점수 계산 중...")
importance_check = []

for actor, action in zip(df['cluster'], df['action_cluster']):
    action_flag = 'Actor' + str(actor) + '_' + 'Action' + str(action)
    importance_check.append(action_flag)

frequency = Counter(importance_check)
importance_dict = dict()

total_count = sum(frequency.values())

for item, value in frequency.items():
    importance = (value / total_count) * 100
    importance_dict[item] = importance

print(f"Importance 점수 샘플:")
for key, value in list(importance_dict.items())[:5]:
    print(f"{key}: {value:.4f}%")

# 2.2. Importance 점수 정규화(0~10)
print("\n2.2 Importance 점수 정규화 중...")
data = importance_dict.values()
data = np.array(list(data)).reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 10))
transformed_data = scaler.fit_transform(data)

score_result = transformed_data.flatten().tolist()
score_result = [round(i, 4) for i in score_result]

for key, new_value in zip(importance_dict.keys(), score_result):
    importance_dict[key] = new_value

importence_list = []

for action in sents_df['Action']:
    importence_list.append(importance_dict[action])

sents_df['importance'] = importence_list
print("정규화 완료!")

# ============================================
# 3. Opportunity
# ============================================

# 3.1. Opportunity score
print("\n3.1 Opportunity 점수 계산 중...")
def Opportunity_score(satisfaction, importance):
    """Opportunity = Importance + Max(Importance - Satisfaction, 0)"""
    result = importance + max(importance - satisfaction, 0)
    return result

opportunity_list = []

for i, j in zip(sents_df['satisfaction'], sents_df['importance']):
    score_result = Opportunity_score(i, j)
    opportunity_list.append(score_result)

sents_df['opportunity_score'] = opportunity_list

print("Opportunity 점수 계산 완료!")
print(f"\n최종 결과:")
print(sents_df.head(10))

# 3.2. Opportunity area 시각화
print("\n3.2 Opportunity Area 시각화 생성 중...")
actions = sents_df.Action
colors = np.random.rand(len(actions), 3)
importance = sents_df.importance
satisfaction = sents_df.satisfaction

plt.figure(figsize=(17, 10))

# Action별 점 찍기
for i, action in enumerate(actions):
    plt.scatter(
        importance[i], 
        satisfaction[i], 
        c=[colors[i]], 
        label=action, 
        s=50, 
        edgecolors='black'
    )

# 범례
plt.legend(
    title='Actions', 
    fontsize=8, 
    title_fontsize=10, 
    loc='best', 
    bbox_to_anchor=(1, 1)
)

# 축 타이틀
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Satisfaction', fontsize=12)
plt.title('Opportunity Area', fontsize=14)

# 만족도 기준선
xdata = [0, 10]
ydata = [satisfaction.mean(), satisfaction.mean()]
plt.plot(xdata, ydata, 'k--', label=f'Satisfaction Mean ({satisfaction.mean():.2f})')

# 중요도 기준선
x_data = [importance.mean(), importance.mean()]
y_data = [-10, 10]
plt.plot(x_data, y_data, 'k--', label=f'Importance Mean ({importance.mean():.2f})')

# 포인트에 텍스트 추가
texts = []
for i, action in enumerate(actions):
    texts.append(plt.text(importance[i], satisfaction[i], action, fontsize=13, ha='left'))

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey', lw=1))

plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.savefig('OpportunityArea.png', dpi=300, bbox_inches='tight')
plt.close()

print("Opportunity Area 그래프가 'OpportunityArea.png'에 저장되었습니다.")

# 결과 저장
sents_df.to_csv('opportunity.csv', encoding='utf-8-sig', index=False)
print("\n최종 결과가 'opportunity.csv'에 저장되었습니다.")

print(f"\n{'='*60}")
print("감성 분석 및 Opportunity 분석이 완료되었습니다!")
print(f"{'='*60}")
print(f"\n최종 결과 요약:")
print(sents_df.describe())

