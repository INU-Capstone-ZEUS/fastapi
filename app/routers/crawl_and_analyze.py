import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from .crawl_news import crawl_news, CrawlError 
from .analyze_news import analyze_news

router = APIRouter()

# Gemini 분석 요청 
class AnalysisReqDTO(BaseModel):
    evaluation: str
    summary: str

# Gemini 분석 응답
class AnalysisResDTO(BaseModel):
    evaluation: str
    summary: str
    link: str
    title: str


# 전체 요청
class CrawlAndAnalyzeRequest(BaseModel):
    company_code: str
    page: int
    company_name: str

# 전체 응답
class CrawlAndAnalyzeResponse(BaseModel):
    status: str
    total_articles: int 
    analysis: List[AnalysisResDTO]


@router.post("/crawl-and-analyze", response_model=CrawlAndAnalyzeResponse)
async def crawl_and_analyze(request: CrawlAndAnalyzeRequest) -> CrawlAndAnalyzeResponse:
    company_code = request.company_code
    page = request.page
    company_name = request.company_name

    # 크롤링 단계
    try:
        articles = crawl_news(company_code, page)
    except CrawlError as e:  # CrawlError만 처리
        raise HTTPException(status_code=400, detail=f"Crawling error: {e.message}")

    except Exception as e:
        # 예기치 않은 에러 처리
        raise HTTPException(status_code=400, detail=f"Crawling error: {e.message}")

    # 분석 단계
    try:
        analyzed_articles = analyze_news(articles, company_name)
    except Exception as e:
        # Gemini API 요청 제한으로 인해 발생하는 에러 처리
        raise HTTPException(status_code=400, detail=f"Gemini API error: {e.message}")


    # 분석 결과를 DTO로 변환
    analysis_dto = [AnalysisResDTO(**article) for article in analyzed_articles]

    # 응답 데이터를 JSON 파일로 저장
    file_name = f"{company_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump([article.dict() for article in analysis_dto], json_file, ensure_ascii=False, indent=4)

    # 응답 데이터 생성
    return CrawlAndAnalyzeResponse(
        status="success",
        total_articles=len(analysis_dto),
        analysis=analysis_dto
    )

