import pandas as pd
import requests
import time
import re
from tqdm import tqdm
from datetime import datetime

# --- 설정 ---
INPUT_FILE = "combined_cleaned_final.csv"
OUTPUT_FILE = "final_with_numbered_comments.csv"

# Reddit API 요청 시 필수
HEADERS = {
    "User-Agent": "MyRedditResearchScript/0.3 by u_yourusername"
}
SLEEP_SEC = 0.1  # (필수) 각 게시글 요청 사이의 딜레이
MAX_RETRIES_429 = 3

# --- 텍스트 클리닝 함수 (원본) ---
def clean_text(text):
    """
    텍스트에서 이모지, 특수문자, 불필요한 공백을 제거하는 함수
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 댓글 파싱을 위한 재귀 함수 ---
def flatten_comments(comment_children, all_bodies):
    """
    트리 구조의 댓글(children)을 받아, 모든 댓글 본문(body)을
    all_bodies 리스트에 재귀적으로 추가합니다.
    """
    if not isinstance(comment_children, list):
        return

    for comment in comment_children:
        if comment.get("kind") != "t1": # 't1'이 댓글을 의미
            continue
            
        data = comment.get("data", {})
        body = data.get("body")
        
        if body and body != "[deleted]" and body != "[removed]":
            all_bodies.append(body)
            
        # 대댓글(replies)이 있으면, 그 replies의 children에 대해 재귀 호출
        replies_data = data.get("replies", {}).get("data", {})
        children = replies_data.get("children", [])
        if children:
            flatten_comments(children, all_bodies)


def fetch_comments_for_post(post_url):
    """
    게시글 URL 하나를 받아서, 모든 댓글 본문(body)의 [리스트]를 반환합니다.
    (변경점: 문자열 대신 리스트 반환)
    """
    try:
        json_url = post_url.split('?')[0]
        if json_url.endswith('/'):
            json_url = json_url[:-1]
        json_url += ".json"
    except Exception:
        return [] # 오류 시 빈 리스트 반환

    # API 요청 (지수 백오프 적용)
    for attempt in range(MAX_RETRIES_429):
        res = requests.get(json_url, headers=HEADERS)
        
        if res.status_code == 429:
            wait_time = SLEEP_SEC * (2 ** attempt)
            time.sleep(wait_time)
            continue
        elif res.status_code != 200:
            return [] # 오류 시 빈 리스트 반환
        else:
            break
    else:
        return [] # 429 최대 재시도 실패 시 빈 리스트 반환

    # JSON 파싱 및 댓글 추출
    try:
        data = res.json()
        comment_children = data[1].get("data", {}).get("children", [])
        
        all_comment_bodies = [] # [댓글1, 댓글2, ...]
        flatten_comments(comment_children, all_comment_bodies)
        
        return all_comment_bodies # (중요) 리스트 자체를 반환
        
    except Exception as e:
        return [] # 파싱 오류 시 빈 리스트 반환

# --- (신규) 댓글 리스트를 번호 매겨 포맷팅하는 함수 ---
def format_comments(comment_list):
    """
    댓글 본문(body)의 리스트를 받아서,
    클리닝 후 "[댓글]\n1. ...\n2. ..." 형태의 문자열로 만듭니다.
    """
    if not comment_list:
        return "" # 댓글 리스트가 비어있으면 빈 문자열 반환

    cleaned_comments = []
    for comment in comment_list:
        cleaned = clean_text(comment) # 각 댓글을 개별적으로 클리닝
        if cleaned: # 클리닝 후 내용이 남은 댓글만 추가
            cleaned_comments.append(cleaned)
            
    if not cleaned_comments:
        return "" # 클리닝 후 댓글이 없으면 빈 문자열 반환

    # 번호 매기기 (1. 댓글내용)
    numbered_lines = [
        f"{i}. {comment}" 
        for i, comment in enumerate(cleaned_comments, start=1)
    ]
    
    # 최종 포맷: [댓글] 헤더 + 줄바꿈 + 번호 매겨진 댓글들
    return "[댓글]\n" + "\n".join(numbered_lines)

# --- (신규) content와 formatted_comments를 안전하게 합치는 함수 ---
def combine_content_and_comments(row):
    """
    (apply.axis=1 용) 댓글이 있을 때만 content에 합쳐서 반환
    """
    content = row['content']
    formatted_comments = row['formatted_comments']
    
    if formatted_comments: # 포맷팅된 댓글이 있을 경우에만
        return f"{content}\n\n{formatted_comments}"
    else: # 댓글이 없으면 원본 content만 반환
        return content

if __name__ == "__main__":
    
    print(f"'{INPUT_FILE}' 파일을 읽어옵니다...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"!! 오류: '{INPUT_FILE}'을 찾을 수 없습니다. 스크립트를 종료합니다.")
        exit()

    print(f"총 {len(df)}개의 게시글에 대해 댓글 수집을 시작합니다.")
    print(f"각 요청당 {SLEEP_SEC}초 대기하므로, 예상 완료 시간은 약 {len(df) * SLEEP_SEC / 60:.1f} 분입니다.")
    
    tqdm.pandas(desc="댓글 수집 중")

    def apply_fetch_with_sleep(url):
        result_list = fetch_comments_for_post(url) # 댓글 [리스트]를 받음
        time.sleep(SLEEP_SEC) # API 예의
        return result_list

    # 1. 'comments_list' 컬럼 생성 (각 셀이 [리스트]임)
    df['comments_list'] = df['url'].progress_apply(apply_fetch_with_sleep)

    print("댓글 수집 완료. 포맷팅을 시작합니다...")

    # 2. 'comments_list'를 'formatted_comments' 문자열로 변환
    df['formatted_comments'] = df['comments_list'].apply(format_comments)

    # 3. 'content'와 'formatted_comments'를 합치기
    # (댓글이 없는 행은 content만 남도록 안전하게 합침)
    df['content'] = df.apply(combine_content_and_comments, axis=1)
    
    # 4. 최종 컬럼 선택 및 저장
    final_cols = ['keyword', 'title', 'content', 'date', 'url']
    df_final = df[final_cols].copy() # .copy()로 경고 방지

    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 모든 작업 완료! 파일 저장: {OUTPUT_FILE}")
    
    # 샘플 확인
    print("\n--- 최종 결과 샘플 (첫 번째 행의 content) ---")
    if not df_final.empty:
        print(df_final.iloc[0]['content'])
    else:
        print("데이터가 없습니다.")
