# --- 데이터 수집(Crawling)을 위한 라이브러리 ---
from selenium import webdriver as wb  # type: ignore
from selenium.webdriver.common.by import By  # type: ignore
from selenium.webdriver.common.keys import Keys  # type: ignore
from selenium.common.exceptions import NoSuchElementException, TimeoutException  # type: ignore
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore
import re  # 정규식 사용

# --- 데이터 처리 및 분석을 위한 라이브러리 ---
import pandas as pd
import time
from datetime import datetime, timedelta
from urllib.parse import quote
import pickle
import sys
import os

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def smart_scroll(driver, scroll_step=1200, pause=0.5, container_selectors=None):
    """
    네이버 검색 페이지 특성을 고려한 다중 스크롤 시도
    """
    container_selectors = container_selectors or ['#main_pack', '#content', '#wrap', '#container', 'body', 'html']
    
    scroll_scripts = [
        ("window.scrollBy(0, arguments[0]);", scroll_step),
        ("window.scrollTo(0, document.body.scrollHeight);", None),
        ("window.scrollTo(0, document.documentElement.scrollHeight);", None)
    ]
    
    for script, value in scroll_scripts:
        try:
            if value is None:
                driver.execute_script(script)
            else:
                driver.execute_script(script, value)
            time.sleep(pause * 0.35)
        except Exception:
            continue
    
    for selector in container_selectors:
        try:
            driver.execute_script("""
                const target = document.querySelector(arguments[0]);
                const delta = arguments[1];
                if (!target) { return; }
                if (target === document.body || target === document.documentElement) {
                    window.scrollBy(0, delta);
                    window.scrollTo(0, target.scrollHeight);
                } else {
                    target.scrollTop = target.scrollHeight;
                    target.dispatchEvent(new Event('scroll', {bubbles: true}));
                }
            """, selector, scroll_step)
            time.sleep(pause * 0.2)
        except Exception:
            continue
    
    try:
        driver.execute_script("""
            window.dispatchEvent(new Event('scroll', {bubbles: true}));
            if (typeof WheelEvent !== 'undefined') {
                window.dispatchEvent(new WheelEvent('wheel', {deltaY: arguments[0], bubbles: true}));
            }
        """, scroll_step)
        time.sleep(pause * 0.15)
    except Exception:
        pass
    
    try:
        return driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
    except Exception:
        return None

def recover_scroll(driver, scroll_step=1200, pause=0.5):
    """
    스크롤 반응이 없는 경우를 대비한 보조 동작
    """
    try:
        driver.execute_script("window.scrollBy(0, -300);")
        time.sleep(pause * 0.4)
        driver.execute_script("window.scrollBy(0, arguments[0]);", max(400, scroll_step // 2))
        time.sleep(pause * 0.4)
    except Exception:
        pass

def collect_blog_urls(driver, blog_url_pattern):
    """
    현재 페이지에서 블로그 URL을 수집
    """
    temp_urls = set()
    
    try:
        title_links = driver.find_elements(By.CLASS_NAME, 'title_link')
        for link in title_links:
            try:
                href = link.get_attribute('href')
                if href and blog_url_pattern.search(href):
                    temp_urls.add(blog_url_pattern.search(href).group(0))
            except Exception:
                continue
    except Exception:
        pass
    
    try:
        blog_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href^="https://blog.naver.com/"]')
        for elem in blog_elements:
            try:
                href = elem.get_attribute('href')
                if href and blog_url_pattern.search(href):
                    temp_urls.add(blog_url_pattern.search(href).group(0))
            except Exception:
                continue
    except Exception:
        pass
    
    return temp_urls

def calculate_date_ranges(start_date_str='20230101', end_date_str='20251114', max_urls=2500):
    """
    날짜 범위 계산
    - 1년 단위로 끊기
    """
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    date_ranges = []
    
    current_start = start_date
    while current_start < end_date:
        # 1년 후
        current_end_1y = current_start + timedelta(days=365)
        
        # end_date를 넘지 않도록
        if current_end_1y > end_date:
            current_end_1y = end_date
        
        # 1년 단위 사용
        date_ranges.append({
            'start': current_start.strftime('%Y%m%d'),
            'end': current_end_1y.strftime('%Y%m%d'),
            'period': '1year'
        })
        
        current_start = current_end_1y + timedelta(days=1)
        if current_start >= end_date:
            break
    
    return date_ranges

def crawl_naver_blog(keyword, start_date, end_date, max_urls=2500):
    """
    네이버 블로그 크롤링
    """
    print(f"\n{'='*60}")
    print(f"네이버 블로그 크롤링 시작")
    print(f"키워드: {keyword}")
    print(f"기간: {start_date} ~ {end_date}")
    print(f"최대 URL 수: {max_urls}")
    print(f"{'='*60}\n")
    
    # URL 인코딩 (특수문자 포함 키워드 제대로 인코딩)
    encoded_keyword = quote(keyword, safe='')
    
    # 네이버 블로그 검색 URL 생성
    blog_search_url = f'https://search.naver.com/search.naver?ssc=tab.blog.all&query={encoded_keyword}&sm=tab_opt&nso=so%3Ar%2Cp%3Afrom{start_date}to{end_date}'
    
    print(f"생성된 URL: {blog_search_url}\n")
    
    # 크롬 브라우저 실행 (옵션 설정 - 안정성 강화 및 로그 억제)
    options = wb.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-gpu')  # GPU 가속 비활성화
    options.add_argument('--disable-extensions')  # 확장 프로그램 비활성화
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-infobars')
    # 로그 및 에러 메시지 억제
    options.add_argument('--log-level=3')  # INFO 레벨 이상만 출력 (FATAL, ERROR만)
    options.add_argument('--disable-logging')  # 로깅 비활성화
    options.add_argument('--disable-background-networking')  # 백그라운드 네트워킹 비활성화
    options.add_argument('--disable-background-timer-throttling')  # 백그라운드 타이머 제한 비활성화
    options.add_argument('--disable-backgrounding-occluded-windows')  # 가려진 창 백그라운드 처리 비활성화
    options.add_argument('--disable-breakpad')  # 크래시 리포팅 비활성화
    options.add_argument('--disable-component-update')  # 컴포넌트 업데이트 비활성화
    options.add_argument('--disable-default-apps')  # 기본 앱 비활성화
    options.add_argument('--disable-sync')  # 동기화 비활성화
    options.add_argument('--disable-background-mode')  # 백그라운드 모드 비활성화
    options.add_argument('--disable-features=TranslateUI')  # 번역 UI 비활성화
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('prefs', {
        'profile.default_content_setting_values.notifications': 2,  # 알림 차단
        'profile.default_content_settings.popups': 0,  # 팝업 차단
    })
    options.add_experimental_option('useAutomationExtension', False)
    options.page_load_strategy = 'normal'  # 페이지 로드 전략
    
    try:
        driver = wb.Chrome(options=options)
    except Exception as e:
        print(f"⚠️  Chrome 드라이버 실행 실패, 기본 옵션으로 재시도: {e}")
        driver = wb.Chrome()
    
    driver.maximize_window()
    driver.implicitly_wait(10)  # 암묵적 대기 시간 설정
    
    # 블로그 URL 패턴 정규식
    blog_url_pattern = re.compile(r'https?://blog\.naver\.com/[^/]+/\d+')
    
    try:
        # 페이지 이동
        driver.get(blog_search_url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        print("✅ 브라우저 실행 및 페이지 이동 완료!")
        
        # "상세 검색결과 보기" 버튼 클릭
        print("\n🔍 '상세 검색결과 보기' 버튼 찾는 중...")
        time.sleep(2)  # 페이지 로딩 대기
        
        try:
            # 여러 선택자로 버튼 찾기
            detail_buttons = driver.find_elements(By.CSS_SELECTOR, 
                'a.more_link, a[class*="more_link"], a[onclick*="goOtherCR"], a[href*="ssc=tab.blog.all"]')
            
            clicked = False
            for btn in detail_buttons:
                try:
                    if btn.is_displayed() and btn.is_enabled():
                        btn_text = btn.text.strip()
                        btn_class = btn.get_attribute('class') or ''
                        if '상세' in btn_text or '상세 검색' in btn_text or 'more_link' in btn_class:
                            driver.execute_script("arguments[0].click();", btn)
                            print("✅ '상세 검색결과 보기' 버튼 클릭 완료!")
                            time.sleep(3)  # 페이지 전환 대기
                            clicked = True
                            break
                except Exception as e:
                    continue
            
            if not clicked:
                print("⚠️  '상세 검색결과 보기' 버튼을 찾을 수 없습니다. 계속 진행...")
        except Exception as e:
            print(f"⚠️  버튼 클릭 중 오류: {e}. 계속 진행...")
        
        print(f"\n{'='*60}")
        print("고정 100회 스크롤 기반 URL 수집 시작")
        print(f"{'='*60}\n")
        
        fixed_scroll_count = 100
        scroll_pause = 1.0
        
        time.sleep(2)  # 스크롤 시작 전 추가 대기
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
        except Exception:
            body = None
        
        # 스크롤 높이 추적 변수
        last_height = driver.execute_script("return document.body.scrollHeight")
        no_change_count = 0  # 스크롤 높이가 변하지 않은 횟수
        
        print("  📜 스크롤 다운 중...")
        for scroll_round in range(fixed_scroll_count):
            actual_scrolls = scroll_round + 1
            try:
                if body is None:
                    body = driver.find_element(By.TAG_NAME, 'body')
                body.send_keys(Keys.END)
            except Exception:
                try:
                    body = driver.find_element(By.TAG_NAME, 'body')
                    body.send_keys(Keys.END)
                except Exception:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)
            
            # 새로운 콘텐츠가 로드되었는지 확인
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                no_change_count += 1
                if no_change_count >= 5:  # 5번 연속 변화 없으면 중단
                    print(f"  ⚠️  스크롤 높이 변화 없음 ({actual_scrolls}/{fixed_scroll_count}번째 스크롤). 조기 종료.")
                    break
            else:
                no_change_count = 0
                last_height = new_height
            
            # 진행 상황 출력 (10번마다)
            if (scroll_round + 1) % 10 == 0:
                print(f"    진행: {scroll_round + 1}/{fixed_scroll_count}번 스크롤 완료")
                try:
                    more_buttons = driver.find_elements(By.CSS_SELECTOR, 'a.more, button.more, .more, [class*="more"], [class*="btn_more"]')
                    for btn in more_buttons:
                        try:
                            if btn.is_displayed() and btn.is_enabled():
                                driver.execute_script("arguments[0].click();", btn)
                                time.sleep(1)
                                print("  🔄 더보기 버튼 클릭")
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
        
        print("  ✅ 스크롤 다운 완료!")
        
        # 스크롤 완료 후 추가 대기 (동적 콘텐츠 로딩)
        print("  ⏳ 최종 콘텐츠 로딩 대기 중...")
        time.sleep(2)
        
        # 스크롤 완료 후 한 번에 URL 수집
        print("  🔍 URL 수집 중...")
        all_seen_urls = collect_blog_urls(driver, blog_url_pattern)
        href_list = list(all_seen_urls)
        
        print(f"\n✅ 총 {len(href_list)}개의 블로그 URL 수집 완료! (총 {actual_scrolls}번 스크롤)")
        
        # URL 수가 1000개 이하면 1년 단위로 변경하라는 메시지
        if len(href_list) < 1000:
            print(f"\n⚠️  URL 수가 1000개 미만입니다 ({len(href_list)}개).")
            print("   1년 단위로 기간을 확장하는 것을 고려하세요.")
        
        # 최대 URL 수 제한
        if len(href_list) > max_urls:
            href_list = href_list[:max_urls]
            print(f"⚠️  최대 URL 수({max_urls})로 제한: {len(href_list)}개")
        
        # 각 블로그 포스트에서 데이터 추출
        print(f"\n{'='*60}")
        print("블로그 포스트 데이터 추출 시작...")
        print(f"{'='*60}\n")
        
        all_data = []
        
        for i, url in enumerate(href_list, 1):
            try:
                print(f"[{i}/{len(href_list)}] 처리 중: {url[:60]}...")
                driver.get(url)
                time.sleep(2)  # 페이지 로딩 대기
                
                # 네이버 블로그는 본문이 'mainFrame'이라는 iframe 안에 있음
                try:
                    driver.switch_to.frame('mainFrame')
                except:
                    pass  # iframe이 없을 수도 있음
                
                # 제목 추출 (신형/구형 에디터 모두 시도)
                title = "N/A"
                title_selectors = [
                    (By.CSS_SELECTOR, '.se-title-text'),
                    (By.CSS_SELECTOR, '.pcol1 > span'),
                    (By.CSS_SELECTOR, '.se-title'),
                    (By.TAG_NAME, 'h1')
                ]
                
                for selector_type, selector in title_selectors:
                    try:
                        title_elem = driver.find_element(selector_type, selector)
                        title = title_elem.text.strip()
                        if title and title != "N/A":
                            break
                    except:
                        continue
                
                # 본문 추출 (신형/구형 에디터 모두 시도)
                content = "N/A"
                content_selectors = [
                    (By.CSS_SELECTOR, '.se-main-container'),
                    (By.CSS_SELECTOR, '#postViewArea'),
                    (By.CSS_SELECTOR, '.se-component-content'),
                    (By.CSS_SELECTOR, '.post-view')
                ]
                
                for selector_type, selector in content_selectors:
                    try:
                        content_elem = driver.find_element(selector_type, selector)
                        content = content_elem.text.strip()
                        # 줄바꿈을 공백으로 변경
                        content = content.replace('\n', ' ').replace('\r', ' ')
                        # 연속된 공백 제거
                        content = ' '.join(content.split())
                        if content and len(content) > 10:
                            break
                    except:
                        continue
                
                # 날짜 추출 (신형/구형 에디터 모두 시도)
                date = "N/A"
                date_selectors = [
                    (By.CSS_SELECTOR, '.se_publishDate'),
                    (By.CSS_SELECTOR, '.date'),
                    (By.CSS_SELECTOR, '.publish_date'),
                    (By.CSS_SELECTOR, '.post-date')
                ]
                
                for selector_type, selector in date_selectors:
                    try:
                        date_elem = driver.find_element(selector_type, selector)
                        date = date_elem.text.strip()
                        if date and date != "N/A":
                            # 날짜 형식 정리 (예: 2025.01.15. 오후 3:00 -> 2025.01.15.)
                            if '.' in date:
                                date_parts = date.split('.')
                                if len(date_parts) >= 3:
                                    date = f"{date_parts[0]}.{date_parts[1]}.{date_parts[2]}."
                            break
                    except:
                        continue
                
                # iframe에서 빠져나와 기본 콘텐츠로 전환
                driver.switch_to.default_content()
                
                # content가 너무 짧거나 없으면 제외
                if content == "N/A" or not content or len(content.strip()) < 20:
                    print(f"  ⏭️  content 없음 또는 너무 짧음으로 건너뜀: {title[:30]}...")
                    continue
                
                # title이 없으면 제외
                if title == "N/A" or not title or len(title.strip()) == 0:
                    print(f"  ⏭️  title 없음으로 건너뜀")
                    continue
                
                # 데이터 저장 (keyword, title, content, date, url 순서)
                all_data.append({
                    'keyword': keyword,
                    'title': title,
                    'content': content,
                    'date': date,
                    'url': url
                })
                
                print(f"  ✅ 수집 완료: {title[:30]}...")
                
            except Exception as e:
                print(f"  ❌ 오류 발생, 건너뜀: {e}")
                # iframe에서 빠져나오기
                try:
                    driver.switch_to.default_content()
                except:
                    pass
                # 오류 발생 시 데이터 저장하지 않고 건너뜀
                continue
        
        print(f"\n✅ 총 {len(all_data)}개의 블로그 포스트 데이터 수집 완료!")
        
        return all_data
        
    except Exception as e:
        print(f"❌ 크롤링 오류: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        # 브라우저 종료
        driver.quit()

def clean_keyword(keyword):
    """
    keyword에서 제외 연산자(-)와 제외 키워드를 제거하고,
    포함 키워드와 + 연산자만 유지
    예: "청각장애" +불편 -알리 -광군 -> "청각장애" +불편
    """
    if pd.isna(keyword) or not keyword:
        return keyword
    
    # 문자열로 변환
    keyword_str = str(keyword)
    
    # 제외 연산자(-)와 그 뒤의 모든 내용 제거
    # - 연산자가 나타나는 첫 번째 위치를 찾아서 그 앞부분만 유지
    if ' -' in keyword_str:
        keyword_str = keyword_str.split(' -')[0]
    elif keyword_str.startswith('-'):
        # -로 시작하는 경우는 빈 문자열 반환 (이상한 경우)
        return keyword_str
    
    # 앞뒤 공백 제거
    keyword_str = keyword_str.strip()
    
    return keyword_str

def create_safe_keyword_name(keyword):
    """
    키워드를 파일명에 사용할 수 있도록 안전한 문자열로 변환
    파일명 형식: 크롤링데이터(네이버블로그, 키워드명)
    """
    # 특수문자 제거 및 공백 처리
    safe_name = keyword.replace('"', '').replace('+', '포함').replace('-', '제외')
    safe_name = safe_name.replace(' ', '_').replace('|', '_')
    # 파일명에 사용할 수 없는 문자 제거 (단, 파일명 자체의 괄호는 유지)
    invalid_chars = ['<', '>', ':', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    # 너무 긴 파일명 방지
    return safe_name[:50]  # 파일명 길이 제한

def save_data(all_data, keyword, source_type='네이버블로그'):
    """
    데이터를 CSV 파일로 저장
    파일명: 네이버블로그_청각장애(2).csv
    """
    if not all_data:
        print("저장할 데이터가 없습니다.")
        return
    
    # DataFrame 생성 (keyword, title, content, date, url 순서)
    df_final = pd.DataFrame(all_data, columns=['keyword', 'title', 'content', 'date', 'url'])
    
    # keyword 컬럼 정리 (제외 키워드 제거)
    df_final['keyword'] = df_final['keyword'].apply(clean_keyword)
    
    # 파일명: 네이버블로그_청각장애(2).csv
    csv_filename = '네이버블로그_청각장애(2).csv'
    
    # CSV 저장 (기존 파일이 있으면 append, 없으면 새로 생성)
    if os.path.exists(csv_filename):
        # 기존 파일 읽기
        df_existing = pd.read_csv(csv_filename, encoding='utf-8-sig')
        # 기존 파일의 keyword도 정리
        if 'keyword' in df_existing.columns:
            df_existing['keyword'] = df_existing['keyword'].apply(clean_keyword)
        # 새 데이터와 통합
        df_final = pd.concat([df_existing, df_final], ignore_index=True)
        # 중복 제거 (URL 기준)
        df_final = df_final.drop_duplicates(subset=['url'], keep='first')
        print(f"  기존 파일에 추가하여 저장합니다.")
    
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장 완료: {csv_filename} ({len(df_final)}행)")
    
    return csv_filename

def main():
    """
    메인 함수 - 네이버 블로그 크롤링 실행
    """
    # 키워드 목록 설정
    keywords = [
        '"청각장애" +불편 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"청각장애" +가전 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"청각장애" +일상 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"농인" +불편 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"농인" +가전 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
    ]
    
    # 날짜 범위 계산 (2020.01.01 ~ 2022.12.31, 1년 단위)
    date_ranges = calculate_date_ranges(start_date_str='20200101', end_date_str='20221231', max_urls=2500)
    
    # 2023.01.01 이후 데이터는 이미 수집했으므로 주석처리
    # date_ranges_2023 = calculate_date_ranges(start_date_str='20230101', end_date_str='20251114', max_urls=2500)
    
    print(f"\n{'='*60}")
    print("네이버 블로그 크롤링 시작")
    print(f"{'='*60}")
    print(f"\n총 {len(date_ranges)}개의 기간으로 분할:")
    for i, date_range in enumerate(date_ranges, 1):
        print(f"  {i}. {date_range['start']} ~ {date_range['end']} ({date_range['period']})")
    
    print(f"\n총 {len(keywords)}개의 키워드 조합:")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw}")
    
    print(f"\n크롤링을 시작합니다...\n")
    
    # 각 키워드별로 크롤링 실행
    all_keywords_data = []
    
    for keyword_idx, keyword in enumerate(keywords, 1):
        print(f"\n{'='*80}")
        print(f"키워드 {keyword_idx}/{len(keywords)}: {keyword}")
        print(f"{'='*80}\n")
        
        keyword_all_data = []
        
        for date_range in date_ranges:
            print(f"\n{'='*60}")
            print(f"기간: {date_range['start']} ~ {date_range['end']}")
            print(f"{'='*60}")
            
            all_data = crawl_naver_blog(
                keyword=keyword,
                start_date=date_range['start'],
                end_date=date_range['end'],
                max_urls=2500
            )
            if all_data:
                keyword_all_data.extend(all_data)
        
        # 각 키워드별 전체 데이터 통합 저장 (CSV, Excel, Pickle 동일 데이터)
        if keyword_all_data:
            print(f"\n{'='*60}")
            print(f"키워드 '{keyword}' 전체 데이터 저장")
            print(f"{'='*60}")
            save_data(keyword_all_data, keyword, source_type='네이버블로그')
            all_keywords_data.extend(keyword_all_data)
            print(f"\n✅ 키워드 '{keyword}' 크롤링 완료! 총 {len(keyword_all_data)}개의 데이터를 수집했습니다.\n")
    
    print(f"\n✅ 모든 크롤링 완료! 총 {len(all_keywords_data)}개의 데이터를 수집했습니다.")

if __name__ == "__main__":
    main()
