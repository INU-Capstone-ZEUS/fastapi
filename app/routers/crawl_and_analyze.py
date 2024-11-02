from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List
from .crawl_news import crawl_news, CrawlError 
from .analyze_news import analyze_news
from fastapi import HTTPException

router = APIRouter()

# Gemini 분석 요청 
class AnalysisReqDTO(BaseModel):
    evaluation: str
    reason: str
    summary: str

# Gemini 분석 응답
class AnalysisResDTO(BaseModel):
    evaluation: str
    reason: str
    summary: str
    link: str


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

        # return CrawlAndAnalyzeResponse(
        #     status="failed",
        #     total_articles=0,
        #     analysis=[],
        #     error_message=e.message
        # )
    except Exception as e:
        # 예기치 않은 에러 처리
        raise HTTPException(status_code=400, detail=f"Crawling error: {e.message}")

        # return CrawlAndAnalyzeResponse(
        #     status="failed",
        #     total_articles=0,
        #     analysis=[],
        #     error_message=f"Unexpected error: {str(e)}"
        # )

    # 분석 단계
    try:
        analyzed_articles = analyze_news(articles, company_name)
    except Exception as e:
        # Gemini API 요청 제한으로 인해 발생하는 에러 처리
        raise HTTPException(status_code=400, detail=f"Gemini API error: {e.message}")
        # return CrawlAndAnalyzeResponse(
        #     status="failed",
        #     total_articles=0,
        #     analysis=[],
        #     error_message=f"Gemini API error: {str(e)}"
        # )

    # 분석 결과를 DTO로 변환
    analysis_dto = [AnalysisResDTO(**article) for article in analyzed_articles]

    # 응답 데이터 생성
    return CrawlAndAnalyzeResponse(
        status="success",
        total_articles=len(analysis_dto),
        analysis=analysis_dto
    )

