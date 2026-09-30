# --- 데이터 수집(Crawling)을 위한 라이브러리 ---
from selenium import webdriver as wb
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException  # type: ignore
import re  # 정규식 사용
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup as bs  # BeautifulSoup 추가

# --- 데이터 처리 및 분석을 위한 라이브러리 ---
import pandas as pd
import time
from datetime import datetime, timedelta
from urllib.parse import quote, urlparse, parse_qs, unquote
import pickle
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def smart_scroll(driver, scroll_step=1200, pause=0.5, container_selectors=None):
    """
    지식인 검색 결과 페이지에 맞춘 다중 스크롤 트리거
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
    스크롤 정체 시 다시 이벤트를 발생시켜 컨텐츠 로딩을 유도
    """
    try:
        driver.execute_script("window.scrollBy(0, -300);")
        time.sleep(pause * 0.4)
        driver.execute_script("window.scrollBy(0, arguments[0]);", max(400, scroll_step // 2))
        time.sleep(pause * 0.4)
    except Exception:
        pass


def extract_real_kin_url(raw_url):
    """
    검색 결과에서 얻은 href(raw_url)에서 실제 kin.naver.com URL을 뽑아냄
    - search.naver.com의 redirect URL (u=...)도 처리
    """
    if not raw_url:
        return None
    
    url = raw_url.strip()
    
    # 1) search.naver.com ... u= 실제 URL 인코딩 형태
    if 'search.naver.com' in url and 'u=' in url and 'kin.naver.com' not in url:
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if 'u' in qs and qs['u']:
                target = unquote(qs['u'][0])
                url = target
        except Exception:
            pass
    
    # 2) 실제 지식인 URL인지 확인
    if 'kin.naver.com' not in url:
        return None

    # 3) 명백히 이상한 URL(프로필 링크 등) 제거
    if 'search/profileLink' in url or '/profileLink' in url:
        return None
    
    # #, 불필요한 트래킹 파라미터 등 정리
    url = url.split('#')[0].strip()
    return url


def collect_kin_urls(driver, kin_url_pattern):
    """
    현재 페이지에서 지식인 URL을 수집
    노트북 방식(BeautifulSoup) 추가
    """
    temp_urls = set()
    
    # 방법 0: 노트북 방식 - BeautifulSoup 사용 (div.question_area > div:nth-child(3) > a)
    try:
        html = driver.page_source
        soup = bs(html, 'lxml')
        url_tags = soup.select('div.question_area > div:nth-child(3) > a')
        print(f"  🔎 BeautifulSoup으로 찾은 URL 태그 개수: {len(url_tags)}개")
        
        for tag in url_tags:
            try:
                href = tag.get('href')
                if not href:
                    continue
                real_url = extract_real_kin_url(href)
                if not real_url:
                    continue
                # qna/detail, qna/question만 남기기
                if kin_url_pattern.search(real_url):
                    temp_urls.add(real_url)
            except Exception:
                continue
    except Exception:
        pass
    
    # 방법 1: 새 UI (headline2) 기반 추출
    try:
        headline_spans = driver.find_elements(
            By.CSS_SELECTOR,
            "span.sds-comps-text.sds-comps-text-ellipsis-1.sds-comps-text-type-headline2"
        )
        print(f"  🔎 headline2 span 개수: {len(headline_spans)}개")
        
        for span in headline_spans:
            try:
                link_el = span.find_element(By.XPATH, "./ancestor::a[1]")
                href = link_el.get_attribute('href') or ''
                if not href:
                    continue
                
                real_url = extract_real_kin_url(href)
                if not real_url:
                    continue
                
                if kin_url_pattern.search(real_url):
                    temp_urls.add(real_url)
            except Exception:
                continue
    except Exception:
        pass
    
    # 방법 2: 기존 title_link 클래스 기반
    try:
        title_links = driver.find_elements(By.CLASS_NAME, 'title_link')
        for link in title_links:
            try:
                href = link.get_attribute('href')
                if not href:
                    continue
                real_url = extract_real_kin_url(href)
                if not real_url:
                    continue
                if kin_url_pattern.search(real_url):
                    temp_urls.add(real_url)
            except Exception:
                continue
    except Exception:
        pass
    
    # 방법 3: a[href*="kin.naver.com"] 기반 보완
    try:
        kin_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="kin.naver.com"], a[href*="kin.naver.com/qna"]')
        for link in kin_links:
            try:
                href = link.get_attribute('href')
                if not href:
                    continue
                real_url = extract_real_kin_url(href)
                if not real_url:
                    continue
                if kin_url_pattern.search(real_url):
                    temp_urls.add(real_url)
            except Exception:
                continue
    except Exception:
        pass
    
    return temp_urls


def calculate_date_ranges(start_date_str='20230101', end_date_str='20251114', max_urls=2500):
    """
    날짜 범위 계산 - 2년 단위
    """
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    date_ranges = []
    
    current_start = start_date
    while current_start < end_date:
        current_end_2y = current_start + timedelta(days=730)
        if current_end_2y > end_date:
            current_end_2y = end_date
        
        date_ranges.append({
            'start': current_start.strftime('%Y%m%d'),
            'end': current_end_2y.strftime('%Y%m%d'),
            'period': '2years'
        })
        
        current_start = current_end_2y + timedelta(days=1)
        if current_start >= end_date:
            break
    
    return date_ranges


def crawl_naver_kin(keyword, start_date, end_date, max_urls=2500):
    """
    네이버 지식인 크롤링
    """
    full_keyword = keyword
    keyword = parse_keyword_for_display(full_keyword)
    
    print(f"\n{'='*60}")
    print(f"네이버 지식인 크롤링 시작")
    print(f"검색 키워드: {full_keyword}")
    print(f"저장 키워드: {keyword}")
    print(f"기간: {start_date} ~ {end_date}")
    print(f"최대 URL 수: {max_urls}")
    print(f"{'='*60}\n")
    
    encoded_keyword = quote(full_keyword, safe='')
    kin_search_url = f'https://search.naver.com/search.naver?where=kin&query={encoded_keyword}&sm=tab_opt&nso=so%3Ar%2Cp%3Afrom{start_date}to{end_date}'
    
    print(f"생성된 URL: {kin_search_url}\n")
    
    options = wb.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-infobars')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-logging')
    options.add_argument('--disable-background-networking')
    options.add_argument('--disable-background-timer-throttling')
    options.add_argument('--disable-backgrounding-occluded-windows')
    options.add_argument('--disable-breakpad')
    options.add_argument('--disable-component-update')
    options.add_argument('--disable-default-apps')
    options.add_argument('--disable-sync')
    options.add_argument('--disable-background-mode')
    options.add_argument('--disable-features=TranslateUI')
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('prefs', {
        'profile.default_content_setting_values.notifications': 2,
        'profile.default_content_settings.popups': 0,
    })
    options.add_experimental_option('useAutomationExtension', False)
    options.page_load_strategy = 'normal'
    
    try:
        driver = wb.Chrome(options=options)
    except Exception as e:
        print(f"⚠️  Chrome 드라이버 실행 실패, 기본 옵션으로 재시도: {e}")
        driver = wb.Chrome()
    
    driver.maximize_window()
    driver.implicitly_wait(10)
    
    # 🔥 qna/detail 또는 qna/question만 허용
    kin_url_pattern = re.compile(
        r'https?://kin\.naver\.com/qna/(detail|question)\.naver[^"\'\s]*'
    )
    
    try:
        driver.get(kin_search_url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        print("✅ 브라우저 실행 및 페이지 이동 완료!")
        
        print("\n🔍 '상세 검색결과 보기' 버튼 찾는 중...")
        time.sleep(2)
        
        try:
            detail_buttons = driver.find_elements(
                By.CSS_SELECTOR,
                'a.more_link, a[class*="more_link"], a[onclick*="goOtherCR"], a[href*="where=kin"]'
            )
            
            clicked = False
            for btn in detail_buttons:
                try:
                    if btn.is_displayed() and btn.is_enabled():
                        btn_text = btn.text.strip()
                        btn_class = btn.get_attribute('class') or ''
                        if '상세' in btn_text or '상세 검색' in btn_text or 'more_link' in btn_class:
                            driver.execute_script("arguments[0].click();", btn)
                            print("✅ '상세 검색결과 보기' 버튼 클릭 완료!")
                            time.sleep(3)
                            clicked = True
                            break
                except Exception:
                    continue
            
            if not clicked:
                print("⚠️  '상세 검색결과 보기' 버튼을 찾을 수 없습니다. 계속 진행...")
        except Exception as e:
            print(f"⚠️  버튼 클릭 중 오류: {e}. 계속 진행...")
        
        print(f"\n{'='*60}")
        print("스크롤 기반 URL 수집 시작")
        print(f"{'='*60}\n")
        
        # 스크롤 설정
        max_scroll_count = 150
        scroll_pause = 1.5
        no_change_limit = 10
        
        time.sleep(3)
        
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
        except Exception:
            body = None
        
        last_height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
        )
        no_change_count = 0
        actual_scrolls = 0
        
        print("  📜 스크롤 다운 중...")
        for scroll_round in range(max_scroll_count):
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
            
            if scroll_round % 5 == 0:
                try:
                    smart_scroll(driver, scroll_step=1500, pause=0.8)
                    time.sleep(0.5)
                except Exception:
                    pass
            
            new_height = driver.execute_script(
                "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
            )
            
            if new_height == last_height:
                no_change_count += 1
                if no_change_count >= no_change_limit:
                    print(f"  ⚠️  스크롤 높이 변화 없음 ({actual_scrolls}번째 스크롤, {no_change_count}회 연속). 조기 종료.")
                    break
            else:
                no_change_count = 0
                last_height = new_height
            
            if (scroll_round + 1) % 10 == 0:
                print(f"    진행: {scroll_round + 1}/{max_scroll_count}번 스크롤 완료 (현재 높이: {new_height})")
                try:
                    more_buttons = driver.find_elements(By.CSS_SELECTOR, 'a.more, button.more, .more, [class*="more"], [class*="btn_more"]')
                    for btn in more_buttons:
                        try:
                            if btn.is_displayed() and btn.is_enabled():
                                driver.execute_script("arguments[0].click();", btn)
                                time.sleep(1.5)
                                print("  🔄 더보기 버튼 클릭")
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
        
        print(f"  ✅ 스크롤 다운 완료! (총 {actual_scrolls}번 스크롤)")
        
        print("  ⏳ 최종 콘텐츠 로딩 대기 중...")
        time.sleep(3)
        
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            smart_scroll(driver, scroll_step=2000, pause=1.0)
            time.sleep(2)
        except Exception:
            pass
        
        print("  🔍 URL 수집 중...")
        all_seen_urls = collect_kin_urls(driver, kin_url_pattern)
        href_list = list(all_seen_urls)
        
        print(f"\n✅ 총 {len(href_list)}개의 지식인 URL 수집 완료! (총 {actual_scrolls}번 스크롤)")
        
        if len(href_list) < 1000:
            print(f"\n⚠️  URL 수가 1000개 미만입니다 ({len(href_list)}개).")
            print("   1년 단위로 기간을 확장하는 것을 고려하세요.")
        
        if len(href_list) > max_urls:
            href_list = href_list[:max_urls]
            print(f"⚠️  최대 URL 수({max_urls})로 제한: {len(href_list)}개")
        
        print(f"\n{'='*60}")
        print("지식인 질문 데이터 추출 시작...")
        print(f"{'='*60}\n")
        
        all_data = []
        seen_urls = set()
        
        for i, url in enumerate(href_list, 1):
            try:
                if not url or url.strip() == "":
                    print(f"[{i}/{len(href_list)}] ⏭️  유효하지 않은 URL로 건너뜀: {url}")
                    continue
                
                if url in seen_urls:
                    print(f"[{i}/{len(href_list)}] ⏭️  중복 URL로 건너뜀: {url[:60]}...")
                    continue
                
                # qna/detail, qna/question만
                if '/qna/' not in url:
                    print(f"[{i}/{len(href_list)}] ⏭️  qna 페이지가 아니라 건너뜀: {url[:60]}...")
                    continue
                if 'search/profileLink' in url or '/profileLink' in url:
                    print(f"[{i}/{len(href_list)}] ⏭️  프로필 링크 건너뜀: {url[:60]}...")
                    continue
                
                seen_urls.add(url)
                print(f"[{i}/{len(href_list)}] 처리 중: {url[:60]}...")
                driver.get(url)
                time.sleep(2)
                
                # -----------------------------
                # 제목 추출 (노트북 방식 + Selenium)
                # -----------------------------
                title = "N/A"
                
                # 1순위: <title>에서 추출
                try:
                    html = driver.page_source
                    soup = bs(html, 'lxml')
                    title_elem = soup.select_one('title')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        title = title.replace(': 지식iN', '').strip()
                except Exception:
                    pass
                
                # 2순위: Selenium 셀렉터
                if title == "N/A" or not title:
                    title_selectors = [
                        (By.CSS_SELECTOR, '.title'),
                        (By.CSS_SELECTOR, '.question-title'),
                        (By.CSS_SELECTOR, '.c-heading__title'),
                        (By.CSS_SELECTOR, 'h2.title'),
                        (By.TAG_NAME, 'h2'),
                        (By.TAG_NAME, 'h1')
                    ]
                    
                    for selector_type, selector in title_selectors:
                        try:
                            title_elem = driver.find_element(selector_type, selector)
                            t = title_elem.text.strip()
                            if t:
                                title = t
                                break
                        except:
                            continue
                
                # -----------------------------
                # 내용(content) 추출
                # -----------------------------
                content = ""
                
                # 1순위: BeautifulSoup으로 .questionDetail
                try:
                    html = driver.page_source
                    soup = bs(html, 'lxml')
                    question_detail = soup.select_one('.questionDetail')
                    if question_detail:
                        tmp = question_detail.get_text(" ", strip=True)
                        if tmp:
                            content = tmp
                except Exception:
                    pass
                
                # 2순위: .questionDetail 안의 p.se-text-paragraph (Selenium)
                if not content:
                    try:
                        detail = driver.find_element(By.CSS_SELECTOR, '.questionDetail')
                        paragraphs = detail.find_elements(By.CSS_SELECTOR, 'p.se-text-paragraph')
                        texts = []
                        for p in paragraphs:
                            txt = p.text.strip()
                            if txt:
                                texts.append(txt)
                        if texts:
                            content = ' '.join(texts)
                    except Exception:
                        pass
                
                # 3순위: 기존 selector 백업 (질문 본문)
                if not content:
                    try:
                        content_selectors = [
                            (By.CSS_SELECTOR, '.c-heading__content'),
                            (By.CSS_SELECTOR, '.question-content'),
                            (By.CSS_SELECTOR, '.content'),
                            (By.CSS_SELECTOR, '.question_text'),
                            (By.CSS_SELECTOR, '#answer-content')
                        ]
                        
                        for selector_type, selector in content_selectors:
                            try:
                                content_elem = driver.find_element(selector_type, selector)
                                tmp = content_elem.text.strip()
                                tmp = tmp.replace('\n', ' ').replace('\r', ' ')
                                tmp = ' '.join(tmp.split())
                                if tmp:
                                    content = tmp
                                    break
                            except:
                                continue
                    except Exception:
                        pass
                
                # 4순위: BeautifulSoup으로 주요 영역 텍스트 한 번 더 시도
                if not content:
                    try:
                        html = driver.page_source
                        soup = bs(html, 'lxml')
                        for selector in ['.c-heading__content', '.question-content', '.content', '.question_text']:
                            elem = soup.select_one(selector)
                            if elem:
                                tmp = elem.get_text(" ", strip=True)
                                if tmp:
                                    content = ' '.join(tmp.split())
                                    break
                    except Exception:
                        pass
                
                # 5순위: 최후의 보루 - 페이지 주요 영역 전체 텍스트
                if not content:
                    try:
                        html = driver.page_source
                        soup = bs(html, 'lxml')
                        main = (
                            soup.select_one('div#content')
                            or soup.select_one('div#main_content')
                            or soup.body
                        )
                        if main:
                            tmp = main.get_text(" ", strip=True)
                            if tmp:
                                content = ' '.join(tmp.split())
                    except Exception:
                        pass
                
                # content 후처리
                if content:
                    content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    content = ' '.join(content.split())
                
                # -----------------------------
                # 날짜(date) 추출
                # -----------------------------
                date = "N/A"
                
                # 1순위: BeautifulSoup
                try:
                    html = driver.page_source
                    soup = bs(html, 'lxml')
                    try:
                        date_elem = soup.select_one('div.userInfo.userInfo__bullet > span:nth-child(3)')
                        if date_elem:
                            date = date_elem.get_text(strip=True)
                            date = date.replace('작성일', '').strip()
                        else:
                            date_elem = soup.select_one('div.userInfo.userInfo__bullet > span:nth-child(2)')
                            if date_elem:
                                date = date_elem.get_text(strip=True)
                                date = date.replace('작성일', '').strip()
                    except:
                        pass
                except Exception:
                    pass
                
                # 2순위: Selenium
                if date == "N/A" or not date:
                    date_selectors = [
                        (By.CSS_SELECTOR, 'div.userInfo.userInfo__bullet > span:nth-child(3)'),
                        (By.CSS_SELECTOR, 'div.userInfo.userInfo__bullet > span:nth-child(2)'),
                        (By.CSS_SELECTOR, '.c-userinfo__date'),
                        (By.CSS_SELECTOR, '.question-date'),
                        (By.CSS_SELECTOR, '.date'),
                        (By.CSS_SELECTOR, '.c-heading__date')
                    ]
                    
                    for selector_type, selector in date_selectors:
                        try:
                            date_elem = driver.find_element(selector_type, selector)
                            d = date_elem.text.strip()
                            if d and d != "N/A":
                                d = d.replace('작성일', '').strip()
                                if '.' in d:
                                    date_parts = d.split('.')
                                    if len(date_parts) >= 3:
                                        d = f"{date_parts[0]}.{date_parts[1]}.{date_parts[2]}."
                                date = d
                                break
                        except:
                            continue
                
                # -----------------------------
                # 최종 필터링
                # -----------------------------
                # ✨ 요구사항: "내용 좀 짧아도 괜찮고, N/A만 아니면 긁어오기"
                if not content or content.strip() == "":
                    print(f"  ⏭️  content 없음으로 건너뜀: {title[:30]}...")
                    continue
                
                if title == "N/A" or not title or len(title.strip()) == 0:
                    print(f"  ⏭️  title 없음으로 건너뜀")
                    continue
                
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
                continue
        
        print(f"\n✅ 총 {len(all_data)}개의 지식인 질문 데이터 수집 완료!")
        
        return all_data
        
    except Exception as e:
        print(f"❌ 크롤링 오류: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        driver.quit()


def parse_keyword_for_display(full_keyword):
    """
    전체 키워드 문자열에서 메인 키워드와 + 키워드만 추출
    예: '"청각장애" +불편 -알리 -광군' -> '청각장애 + 불편'
    """
    if not full_keyword:
        return ""
    
    main_keyword = ""
    main_match = re.search(r'"([^"]+)"', full_keyword)
    if main_match:
        main_keyword = main_match.group(1)
    
    plus_keywords = []
    plus_pattern = r'\+\s*([^\s-]+)'
    plus_matches = re.findall(plus_pattern, full_keyword)
    plus_keywords = [kw.strip() for kw in plus_matches if kw.strip()]
    
    if main_keyword:
        if plus_keywords:
            return f"{main_keyword} + {' + '.join(plus_keywords)}"
        else:
            return main_keyword
    else:
        parts = full_keyword.split()
        if parts:
            main_keyword = parts[0].strip('"')
            if plus_keywords:
                return f"{main_keyword} + {' + '.join(plus_keywords)}"
            else:
                return main_keyword
    
    return full_keyword


def create_safe_keyword_name(parsed_keyword):
    """
    파싱된 키워드를 파일명에 사용할 수 있도록 안전한 문자열로 변환
    파일명 형식: 크롤링데이터(네이버지식인, 키워드명)
    """
    safe_name = parsed_keyword.replace('"', '')
    safe_name = safe_name.replace(' + ', '_+_').replace(' ', '_')
    invalid_chars = ['<', '>', ':', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    return safe_name[:50]


def save_data(all_data, keyword, source_type='네이버지식인'):
    """
    데이터를 CSV 파일로 저장 (Pickle, Excel 저장 제거)
    파일명 형식: 크롤링데이터(네이버지식인, 키워드명)
    """
    if not all_data:
        print("저장할 데이터가 없습니다.")
        return
    
    df_final = pd.DataFrame(all_data, columns=['keyword', 'title', 'content', 'date', 'url'])
    
    safe_keyword = create_safe_keyword_name(keyword)
    base_filename = f'크롤링데이터({source_type}, {safe_keyword})'
    
    csv_filename = f'{base_filename}(2).csv'
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장 완료: {csv_filename} ({len(df_final)}행)")
    
    # Pickle 저장 제거
    # pkl_filename = f'{base_filename}.pkl'
    # with open(pkl_filename, 'wb') as f:
    #     pickle.dump(df_final, f)
    # print(f"✅ Pickle 저장 완료: {pkl_filename} ({len(df_final)}행)")
    
    # Excel 저장 제거
    # excel_filename = f'{base_filename}.xlsx'
    # try:
    #     df_final.to_excel(excel_filename, index=False, engine='openpyxl')
    #     print(f"✅ Excel 저장 완료: {excel_filename} ({len(df_final)}행)")
    # except Exception as e:
    #     print(f"⚠️  Excel 저장 실패: {e}")
    #     print("   openpyxl 설치 필요: pip install openpyxl")
    
    return csv_filename


def main():
    """
    메인 함수 - 네이버 지식인 크롤링 실행
    """
    keywords = [
        '"청각장애" +불편 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"청각장애" +가전 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"청각장애" +일상 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"농인" +불편 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
        '"농인" +가전 -알리 -광군 -광고 -쿠폰 -보청기 -인공와우 -주민센터 -산재 -신청',
    ]
    
    # 2020.01.01 ~ 2022.12.31 기간 수집
    date_ranges = calculate_date_ranges(start_date_str='20200101', end_date_str='20221231', max_urls=2500)
    
    # 나머지 년도 주석처리
    # date_ranges = calculate_date_ranges(start_date_str='20230101', end_date_str='20251115', max_urls=2500)
    
    print(f"\n{'='*60}")
    print("네이버 지식인 크롤링 시작")
    print(f"{'='*60}")
    print(f"\n총 {len(date_ranges)}개의 기간으로 분할:")
    for i, date_range in enumerate(date_ranges, 1):
        print(f"  {i}. {date_range['start']} ~ {date_range['end']} ({date_range['period']})")
    
    print(f"\n총 {len(keywords)}개의 키워드 조합:")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw}")
    
    print(f"\n크롤링을 시작합니다...\n")
    
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
            
            all_data = crawl_naver_kin(
                keyword=keyword,
                start_date=date_range['start'],
                end_date=date_range['end'],
                max_urls=2500
            )
            if all_data:
                keyword_all_data.extend(all_data)
        
        if keyword_all_data:
            print(f"\n{'='*60}")
            print(f"키워드 '{keyword}' 전체 데이터 저장")
            print(f"{'='*60}")
            display_keyword = keyword_all_data[0]['keyword'] if keyword_all_data else keyword
            save_data(keyword_all_data, display_keyword, source_type='네이버지식인')
            all_keywords_data.extend(keyword_all_data)
            print(f"\n✅ 키워드 '{keyword}' 크롤링 완료! 총 {len(keyword_all_data)}개의 데이터를 수집했습니다.\n")
    
    print(f"\n✅ 모든 크롤링 완료! 총 {len(all_keywords_data)}개의 데이터를 수집했습니다.")


if __name__ == "__main__":
    main()
