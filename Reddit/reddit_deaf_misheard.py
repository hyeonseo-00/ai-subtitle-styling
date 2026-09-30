import requests
import pandas as pd
from datetime import datetime
import time
 
BASE_URL = "https://www.reddit.com"
 
# Reddit 권장 형태: 자기용도 + Reddit 계정명 같이 써주면 좋음
HEADERS = {
    "User-Agent": "MyRedditResearchScript/0.1 by u_yourusername"
}
 
KEYWORDS = ["deaf", "misheard"]
 
# 서브 쿼리용 토큰 (너무 많이 쓰면 또 429 맞음, 일단 모음만)
SUB_TOKENS = ["", "a", "e", "i", "o", "u"]   # ""는 그냥 keyword 단독
 
MAX_PAGES_PER_SUBQUERY = 3   # 각 쿼리당 최대 3페이지 (= 300개)
SLEEP_SEC = 2.0              # 요청 간 기본 딜레이(초)
 
MAX_RETRIES_429 = 3          # 429 나왔을 때 재시도 최대 횟수
 
 
def fetch_search_with_backoff(query, max_pages=3, sleep_sec=2.0):
    """
    /search.json 에서 단순 쿼리로 페이징 돌리는 함수.
    429(Too Many Requests) 나오면 지수 백오프로 잠깐 기다렸다가 재시도.
    """
    all_rows = []
    after = None
 
    for page in range(max_pages):
        for attempt in range(MAX_RETRIES_429):
            params = {
                "q": query,
                "limit": 100,
                "sort": "new",
                "type": "link",
                "t": "all",  # 전체 기간
            }
            if after:
                params["after"] = after
 
            print(f"[QUERY='{query}'] PAGE {page+1}, attempt {attempt+1} 요청 중...")
            res = requests.get(f"{BASE_URL}/search.json", headers=HEADERS, params=params)
 
            if res.status_code == 429:
                # Too Many Requests → 지수 백오프
                wait_time = sleep_sec * (2 ** attempt)
                print(f"  > 429 (레이트리밋). {wait_time}초 대기 후 재시도.")
                time.sleep(wait_time)
                continue  # 같은 page 다시 시도
            elif res.status_code != 200:
                print(f"  > 요청 실패: {res.status_code}, 이 쿼리 중단.")
                return all_rows
            else:
                # 정상 응답
                break
        else:
            # MAX_RETRIES_429 번 모두 429면 이 쿼리는 더 진행 안 함
            print("  > 429가 계속 발생해서 이 쿼리는 더 이상 진행 안 함.")
            return all_rows
 
        data = res.json().get("data", {})
        children = data.get("children", [])
        if not children:
            print("  > 더 이상 결과 없음 (children 비어있음).")
            break
 
        for c in children:
            d = c.get("data", {})
            post_id = d.get("id")
 
            title = d.get("title", "") or ""
            content = d.get("selftext", "") or ""
            created = d.get("created_utc", None)
 
            if created is not None:
                date_str = datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = None
 
            all_rows.append({
                "post_id": post_id,
                "title": title,
                "content": content,
                "date": date_str,
                "url": BASE_URL + d.get("permalink", "")
            })
 
        after = data.get("after")
        if not after:
            print("  > after 토큰 없음 → 마지막 페이지.")
            break
 
        time.sleep(sleep_sec)
 
    return all_rows
 
 
def fetch_keyword_many_by_subtokens_with_backoff(
    keyword,
    sub_tokens,
    max_pages_per_subquery=3,
    sleep_sec=2.0,
):
    """
    keyword + sub_token 쿼리를 여러 개 던지면서 최대한 많은 글 수집.
    429를 고려해서 살살 수집하는 버전.
    """
    print(f"\n========== 키워드 '{keyword}' 수집 시작 ==========")
 
    all_rows = []
    seen_post_ids = set()
 
    for token in sub_tokens:
        # ""이면 그냥 keyword만, 아니면 "keyword token"
        query = keyword if token == "" else f"{keyword} {token}"
        print(f"\n[{keyword}] 서브쿼리 '{query}' 실행")
 
        rows = fetch_search_with_backoff(
            query,
            max_pages=max_pages_per_subquery,
            sleep_sec=sleep_sec,
        )
 
        new_count = 0
        for r in rows:
            pid = r.get("post_id")
            if pid and pid not in seen_post_ids:
                seen_post_ids.add(pid)
                r["keyword"] = keyword
                r["sub_token"] = token
                all_rows.append(r)
                new_count += 1
 
        print(f"[{keyword}] 서브쿼리 '{query}'에서 신규 수집: {new_count}개, 누적: {len(all_rows)}개")
 
        # 서브쿼리 사이에도 살짝 쉬어주기
        time.sleep(sleep_sec)
 
    print(f"[{keyword}] 최종 수집 개수: {len(all_rows)}개\n")
    return all_rows
 
 
if __name__ == "__main__":
    all_data = []
 
    for kw in KEYWORDS:
        rows = fetch_keyword_many_by_subtokens_with_backoff(
            kw,
            sub_tokens=SUB_TOKENS,
            max_pages_per_subquery=MAX_PAGES_PER_SUBQUERY,
            sleep_sec=SLEEP_SEC,
        )
        all_data.extend(rows)
 
    df = pd.DataFrame(all_data)
 
    if {"keyword", "post_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["keyword", "post_id"])
 
    print("총 수집 개수:", len(df))
    print(df.head())
 
    output_name = "reddit_global_deaf_misheard_multisearch_backoff.csv"
    df.to_csv(output_name, index=False, encoding="utf-8-sig")
    print(f"CSV 저장 완료: {output_name}")
