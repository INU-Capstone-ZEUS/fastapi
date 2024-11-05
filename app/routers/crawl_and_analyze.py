import json
import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from .crawl_news import crawl_news, CrawlError 
from .analyze_news import analyze_news
from .websocket import notify_clients

router = APIRouter()
s3_client = boto3.client("s3")

# 분석 응답 모델 정의
class AnalysisResDTO(BaseModel):
    evaluation: str
    summary: str
    link: str
    title: str

class CrawlAndAnalyzeRequest(BaseModel):
    company_code: str
    page: int
    company_name: str

class CrawlAndAnalyzeResponse(BaseModel):
    status: str
    total_articles: int 
    analysis: List[AnalysisResDTO]

@router.post("/crawl-and-analyze", response_model=CrawlAndAnalyzeResponse)
async def crawl_and_analyze(request: CrawlAndAnalyzeRequest) -> CrawlAndAnalyzeResponse:
    company_code = request.company_code
    page = request.page
    company_name = request.company_name

    try:
        articles = crawl_news(company_code, page)
    except CrawlError as e:
        raise HTTPException(status_code=400, detail=f"Crawling error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crawling error: {e.message}")

    try:
        analyzed_articles = analyze_news(articles, company_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gemini API error: {e.message}")

    analysis_dto = [AnalysisResDTO(**article) for article in analyzed_articles]

    s3_bucket_name = "dev-jeus-bucket"  
    file_name = f"{company_name}.json"

    # 기존 JSON 파일 가져오기
    try:
        response = s3_client.get_object(Bucket=s3_bucket_name, Key=file_name)
        existing_data = json.loads(response['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        # 파일이 없는 경우 빈 리스트로 초기화
        existing_data = []

    # 기존 데이터에서 링크를 기준으로 중복 검사
    existing_links = {item["link"] for item in existing_data}

    # 중복되지 않는 새 데이터를 필터링하여 추가
    new_data = [article.dict() for article in analysis_dto if article.link not in existing_links]

    # 기존 데이터에 새로운 비중복 데이터 추가
    updated_data = existing_data + new_data

    # JSON 데이터를 다시 S3에 업로드
    try:
        s3_client.put_object(
            Bucket=s3_bucket_name,
            Key=file_name,
            Body=json.dumps(updated_data, ensure_ascii=False, indent=4),
            ContentType='application/json'
        )
        
        await notify_clients({company_name})  # 클라이언트 알림
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {e}")

    return CrawlAndAnalyzeResponse(
        status="success",
        total_articles=len(new_data),
        analysis=analysis_dto
    )
