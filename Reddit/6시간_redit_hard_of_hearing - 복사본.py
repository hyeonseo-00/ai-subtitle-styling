import requests
import pandas as pd
from datetime import datetime
import time
import os
import json
 
BASE_URL = "https://www.reddit.com"
 
# Reddit 권장 형태: 자기용도 + Reddit 계정명 같이 써주면 좋음
HEADERS = {
    "User-Agent": "MyRedditResearchScript/0.1 by u_yourusername"
}
 
# 서브레딧 크롤링 설정
SUBREDDIT = "hardofhearing"  # r/hardofhearing
SORT_TYPE = "new"  # 최신순 유지
 
# 서브 쿼리용 토큰 (15시간 러닝을 위한 설정)
# 토큰 자체는 의미 없고, 반복 구간을 나누는 용도로만 사용됨
SUB_TOKENS = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",  # 알파벳 26개
]  # 총 36개 토큰

# 토큰당 페이지 수 (36 토큰 × 15페이지 = 540페이지 ≒ 최대 54,000게시글 수준)
# 댓글 수집 + 429 백오프까지 고려하면 대략 12~15시간 구간에서 종료될 가능성이 큼
MAX_PAGES_PER_TOKEN = 15    # 각 토큰당 최대 15페이지
SLEEP_SEC = 3.0            # 요청 간 기본 딜레이(초) - 레이트 리밋 방지
COMMENT_SLEEP_SEC = 3.0    # 댓글 수집 간 딜레이(초) - 429 방지
 
MAX_RETRIES_429 = 5        # 429 나왔을 때 재시도 최대 횟수 증가
 
 
def fetch_comments(post_permalink, max_retries=3):
    """게시글의 댓글을 가져오는 함수 (429 에러 처리 포함)"""
    for attempt in range(max_retries):
        try:
            comment_url = f"{BASE_URL}{post_permalink}.json"
            res = requests.get(comment_url, headers=HEADERS, timeout=15)
            
            if res.status_code == 429:
                # 429 에러 시 더 긴 대기
                wait_time = COMMENT_SLEEP_SEC * (2 ** attempt) + 10  # 최소 10초 추가
                time.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue
                else:
                    return ""  # 최대 재시도 횟수 초과
            elif res.status_code != 200:
                return ""
            
            # JSON 디코딩 에러 방지
            try:
                if not res.text or not res.text.strip():
                    if attempt < max_retries - 1:
                        time.sleep(COMMENT_SLEEP_SEC)
                        continue
                    return ""
                data = res.json()
            except json.JSONDecodeError as e:
                print(f"  JSON 디코딩 실패 (응답 길이: {len(res.text)}): {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(COMMENT_SLEEP_SEC * (attempt + 1))
                    continue
                return ""
            if len(data) < 2:
                return ""
            
            # 두 번째 항목이 댓글 리스트
            comments_data = data[1].get("data", {}).get("children", [])
            comments_text = []
            
            for comment_item in comments_data:
                if comment_item.get("kind") == "more":
                    continue
                comment_data = comment_item.get("data", {})
                body = comment_data.get("body", "").strip()
                if body:
                    comments_text.append(body)
            
            return "\n\n".join(comments_text) if comments_text else ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(COMMENT_SLEEP_SEC * (attempt + 1))
                continue
            return ""
    return ""


def fetch_subreddit_with_backoff(subreddit, sort_type="new", max_pages=25, sleep_sec=2.0):
    """
    /r/{subreddit}/{sort}.json 에서 게시글을 페이징하면서 수집 (검색 없이 최신순만).
    429(Too Many Requests) 나오면 지수 백오프로 잠깐 기다렸다가 재시도.
    """
    all_rows = []
    after = None
    
    url = f"{BASE_URL}/r/{subreddit}/{sort_type}.json"
 
    for page in range(max_pages):
        for attempt in range(MAX_RETRIES_429):
            params = {
                "limit": 100,
            }
            if after:
                params["after"] = after
 
            print(f"[r/{subreddit}/{sort_type}] PAGE {page+1}, attempt {attempt+1} 요청 중...")
            res = requests.get(url, headers=HEADERS, params=params)
 
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
            permalink = d.get("permalink", "")
 
            if created is not None:
                date_str = datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = None
 
            # 댓글 수집 및 content에 추가 (429 에러 처리 포함)
            comments = fetch_comments(permalink, max_retries=3)
            if comments:
                if content:
                    content = content + "\n\n[댓글]\n" + comments
                else:
                    content = "\n\n[댓글]\n" + comments

            all_rows.append({
                "post_id": post_id,
                "keyword": subreddit,  # 서브레딧명을 keyword로 사용
                "title": title,
                "content": content,
                "date": date_str,
                "url": BASE_URL + permalink
            })
            
            # 댓글 수집으로 인한 딜레이 (429 방지를 위해 더 길게)
            time.sleep(COMMENT_SLEEP_SEC)
            
            # 진행 상황 출력 (50개마다)
            if len(all_rows) % 50 == 0:
                print(f"  진행: {len(all_rows)}개 게시글 수집 완료")
 
        after = data.get("after")
        if not after:
            print("  > after 토큰 없음 → 마지막 페이지.")
            break
 
        time.sleep(sleep_sec)
 
    return all_rows
 
 
def fetch_subreddit_many_by_tokens(
    subreddit,
    sort_type="new",
    sub_tokens=None,
    max_pages_per_token=25,
    sleep_sec=2.0,
    output_filename=None,
):
    """
    여러 토큰으로 서브레딧 게시글을 수집 (토큰마다 연속적으로 크롤링하여 더 많은 결과 수집).
    429를 고려해서 살살 수집하는 버전.
    토큰마다 중간 저장을 수행하여 중단되어도 데이터 손실 방지.
    """
    if sub_tokens is None:
        sub_tokens = [""]
    
    if output_filename is None:
        output_filename = f"reddit_{subreddit}_subreddit_hard_of_hearing_26.csv"
    
    print(f"\n========== r/{subreddit} 서브레딧 크롤링 시작 (정렬: {sort_type}) ==========")
    print(f"토큰 개수: {len(sub_tokens)}개 (각 토큰당 {max_pages_per_token}페이지)\n")
    print(f"총 예상 페이지: {len(sub_tokens)} × {max_pages_per_token} = {len(sub_tokens) * max_pages_per_token}페이지\n")
    print(f"중간 저장 파일: {output_filename}\n")
 
    all_rows = []
    seen_post_ids = set()
    global_after = None  # 전역 after 토큰으로 연속 크롤링
 
    for token_idx, token in enumerate(sub_tokens):
        token_label = f"토큰 '{token}'" if token else "전체 최신순"
        print(f"\n[r/{subreddit}] {token_label} 실행 ({token_idx+1}/{len(sub_tokens)})")
 
        rows, last_after = fetch_subreddit_with_backoff_continuation(
            subreddit,
            sort_type=sort_type,
            max_pages=max_pages_per_token,
            start_after=global_after,  # 이전 토큰의 마지막 위치에서 이어서 시작
            sleep_sec=sleep_sec,
        )
        
        global_after = last_after  # 다음 토큰을 위한 after 업데이트
 
        new_count = 0
        duplicate_count = 0  # 중복으로 제외된 게시글 수
        new_rows_for_save = []
        
        for r in rows:
            pid = r.get("post_id")
            url = r.get("url", "")
            # post_id와 url 둘 다 체크하여 중복 방지
            if pid and str(pid) not in seen_post_ids and url not in seen_post_ids:
                seen_post_ids.add(str(pid))
                seen_post_ids.add(url)
                all_rows.append(r)
                new_rows_for_save.append(r)
                new_count += 1
            else:
                duplicate_count += 1
 
        print(f"[r/{subreddit}] {token_label}에서:")
        print(f"  - 수집된 게시글: {len(rows)}개")
        print(f"  - 중복 제외: {duplicate_count}개")
        print(f"  - 신규 수집: {new_count}개")
        print(f"  - 누적: {len(all_rows)}개")
        
        # 토큰마다 중간 저장 (new3.csv에 새로 수집한 데이터만 저장, new2.csv는 절대 훼손하지 않음)
        if new_rows_for_save:
            try:
                new_df = pd.DataFrame(new_rows_for_save)
                # 헤더는 첫 저장 시만, 이후는 append
                file_exists = os.path.exists(output_filename)
                new_df.to_csv(
                    output_filename,
                    mode="a" if file_exists else "w",
                    header=not file_exists,
                    index=False,
                    encoding="utf-8-sig"
                )
                print(f"  중간 저장 완료: {output_filename} (추가 {len(new_rows_for_save)}개, 누적: {len(all_rows)}개)")
            except Exception as e:
                print(f"  중간 저장 실패 (무시하고 계속): {e}")
 
        # 토큰 사이에도 살짝 쉬어주기
        time.sleep(sleep_sec)
 
    # 최종 저장 (new3.csv에 새로 수집한 데이터만 저장, new2.csv는 절대 훼손하지 않음)
    try:
        if all_rows:
            final_df = pd.DataFrame(all_rows)
            if "url" in final_df.columns:
                final_df = final_df.drop_duplicates(subset=["url"])
            # new3.csv에 새로 수집한 데이터만 저장 (기존 데이터는 포함하지 않음)
            final_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
            print(f"\n최종 저장 완료: {output_filename} (새로 수집한 데이터 {len(final_df):,}개)")
    except Exception as e:
        print(f"\n최종 저장 실패: {e}")
 
    print(f"\n[r/{subreddit}] 최종 수집 개수: {len(all_rows):,}개\n")
    return all_rows
 
 
def fetch_subreddit_with_backoff_continuation(subreddit, sort_type="new", max_pages=25, start_after=None, sleep_sec=2.0):
    """
    /r/{subreddit}/{sort}.json 에서 게시글을 페이징하면서 수집 (이전 위치에서 이어서 시작 가능).
    429(Too Many Requests) 나오면 지수 백오프로 잠깐 기다렸다가 재시도.
    """
    all_rows = []
    after = start_after  # 이전 위치에서 시작
    
    url = f"{BASE_URL}/r/{subreddit}/{sort_type}.json"
 
    for page in range(max_pages):
        for attempt in range(MAX_RETRIES_429):
            params = {
                "limit": 100,
            }
            if after:
                params["after"] = after
 
            print(f"[r/{subreddit}/{sort_type}] PAGE {page+1}, attempt {attempt+1} 요청 중...")
            res = requests.get(url, headers=HEADERS, params=params)
 
            if res.status_code == 429:
                # Too Many Requests → 지수 백오프 (더 긴 대기)
                wait_time = sleep_sec * (2 ** attempt) + 15  # 최소 15초 추가
                print(f"  > 429 (레이트리밋). {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
                continue  # 같은 page 다시 시도
            elif res.status_code != 200:
                print(f"  > 요청 실패: {res.status_code}, 이 쿼리 중단.")
                return all_rows, after
            else:
                # 정상 응답 - JSON 디코딩 확인
                try:
                    if not res.text or not res.text.strip():
                        print(f"  > 빈 응답 수신. 재시도...")
                        time.sleep(sleep_sec)
                        continue
                    # JSON 파싱 시도
                    res.json()  # 유효성 확인만
                    break
                except json.JSONDecodeError as e:
                    print(f"  > JSON 디코딩 실패 (응답 길이: {len(res.text)}): {str(e)[:100]}")
                    if attempt < MAX_RETRIES_429 - 1:
                        time.sleep(sleep_sec * (attempt + 1))
                        continue
                    else:
                        print(f"  > JSON 디코딩 실패로 인한 중단.")
                        return all_rows, after
        else:
            # MAX_RETRIES_429 번 모두 429면 더 긴 대기 후 계속 진행
            print(f"  > 429가 {MAX_RETRIES_429}번 연속 발생. 120초 대기 후 계속 진행...")
            time.sleep(120)  # 2분 대기
            # 중단하지 않고 계속 진행
 
        # JSON 파싱 (위에서 유효성 확인 완료)
        try:
            data = res.json().get("data", {})
        except json.JSONDecodeError as e:
            print(f"  > JSON 파싱 중 에러: {str(e)[:100]}")
            break  # 이 페이지는 건너뛰고 다음으로
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
            permalink = d.get("permalink", "")

            if created is not None:
                date_str = datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = None

            # 댓글 수집 및 content에 추가 (429 에러 처리 포함)
            comments = fetch_comments(permalink, max_retries=3)
            if comments:
                if content:
                    content = content + "\n\n[댓글]\n" + comments
                else:
                    content = "\n\n[댓글]\n" + comments

            all_rows.append({
                "post_id": post_id,
                "keyword": subreddit,  # 서브레딧명을 keyword로 사용
                "title": title,
                "content": content,
                "date": date_str,
                "url": BASE_URL + permalink
            })
            
            # 댓글 수집으로 인한 딜레이 (429 방지를 위해 더 길게)
            time.sleep(COMMENT_SLEEP_SEC)
            
            # 진행 상황 출력 (50개마다)
            if len(all_rows) % 50 == 0:
                print(f"  진행: {len(all_rows)}개 게시글 수집 완료")
        
        after = data.get("after")
        if not after:
            print("  > after 토큰 없음 → 마지막 페이지.")
            break

        time.sleep(sleep_sec)

    return all_rows, after


if __name__ == "__main__":
    output_name = f"reddit_{SUBREDDIT}_subreddit_hard_of_hearing_26.csv"
    
    all_data = fetch_subreddit_many_by_tokens(
        SUBREDDIT,
        sort_type=SORT_TYPE,
        sub_tokens=SUB_TOKENS,
        max_pages_per_token=MAX_PAGES_PER_TOKEN,
        sleep_sec=SLEEP_SEC,
        output_filename=output_name,  # 중간 저장 파일명 전달
    )
    
    # 최종 결과 출력
    df = pd.DataFrame(all_data)
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])
    
    print(f"\n========== 최종 결과 ==========")
    print(f"총 수집 개수: {len(df):,}")
    print(df.head())
    print(f"\nCSV 저장 완료: {output_name}")
 
