from bs4 import BeautifulSoup
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import datetime
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any

# 크롤링 요청 데이터 모델
class CrawlRequest(BaseModel):
    company_code: str
    page: int

# 라우터 설정
router = APIRouter()

class CrawlError(Exception):
    """Custom exception for errors during crawling."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

# Selenium 설정 함수
def get_browser():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

class CrawlError(Exception):

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def crawl_news(company_code: str, page: int) -> List[Dict[str, str]]:
    driver = None  # driver 초기화
    try:
        if(len(company_code) != 6):
            print("종목 코드가 올바른 형식이 아닙니다.")
            raise CrawlError("종목 코드가 올바른 형식이 아닙니다.")
        
        if( page < 1):
            print("페이지 번호는 1과 200 사이여야 합니다.")
            raise CrawlError("페이지 번호는 1과 200 사이여야 합니다.")    

        
        # URL / 요청 헤더 설정
        url = f'https://finance.naver.com/item/news.naver?code={company_code}&page={page}'
        driver = get_browser()
        driver.get(url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'news_frame')))
        driver.switch_to.frame('news_frame')

        source_code = driver.page_source
        html = BeautifulSoup(source_code, "html.parser")

        # 중복 뉴스 제거
        for tr in html.select('tr.relation_lst'):
            tr.decompose()

        # 기사 item의 신문사 / 날짜 / 뉴스 주소 갖고 오기
        infos = html.select('.info')
        dates = html.select('.date')
        aTags = html.select('td.title a')

        links = [a.attrs['href'] for a in aTags]
        articles = []

        for i, full_url in enumerate(links):
            try:
                driver.get(full_url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'article')))
            except Exception as e:
                raise CrawlError(f"기사 페이지 로딩 실패: {full_url}")

            new_page_source = driver.page_source
            soup = BeautifulSoup(new_page_source, 'html.parser')

            for div in soup.select('div.vod_player_wrap._VIDEO_AREA_WRAP'):
                div.decompose()

            for div in soup.select('div.artical-btm'):
                div.decompose()

            for br in soup.find_all("br"):
                br.replace_with("\n")

            article_content = soup.select_one('article').text.strip()
            article_title = soup.select_one('#title_area span').text.strip()

            article = {
                'title': article_title,
                'publisher': infos[i].text.strip() if i < len(infos) else 'Unknown',
                'date': dates[i].text.strip() if i < len(dates) else 'Unknown',
                'link': full_url,
                'content': article_content,
            }

            articles.append(article)

        driver.quit()

    # 모든 크롤링 작업이 끝난 후 브라우저 종료
    except CrawlError as e:
        if driver:
            driver.quit()
        raise e  # CrawlError를 다시 발생시켜서 상위 함수에서 처리
    except Exception as e:
        if driver:
            driver.quit()
        raise CrawlError(f"크롤링 중 에러 발생: {str(e)}")  # 모든 에러를 CrawlError로 감싸서 던짐

    return articles
