import os
import google.generativeai as genai
from dotenv import load_dotenv
import typing_extensions as typing
from fastapi import APIRouter
from typing import Dict, Any, List
import json
from fastapi import APIRouter, HTTPException

class Analysis(typing.TypedDict):
    evaluation: str
    reason: str
    summary: str

router = APIRouter()

def configure_gemini_api():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('MY_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)


def analyze_article(article_content: str, company_name: str, article_link: str, article_title: str):

    if not article_content:
        return {
            "evaluation": "Error",
            "summary": "분석 실패",
            "link": article_link,  # 링크는 결과에만 포함,
            "title": article_title
        }
    
    prompt = f"""
    다음의 Article을 바탕으로, 이 기사가 ${company_name} 종목에 대해 긍정적인 평가를 내리고 있는지, 부정적인 평가를 내리고 있는지, 또는 종목과 관련이 없는지 판단해줘. [긍정 / 관련 없음 / 부정] 중 하나의 단어로 평가를 내려서 evaluation 필드에 저장해줘. 또한 이 기사에 포함된 종목 관련 정말 중요한 기사의 핵심 내용을 한 번에 알아볼 수 있도록 한 줄로 간략하게 요약해서 'summary'에 저장해줘. 반환 결과는 반드시 JSON 형식이어야 해.
    여기서 'summary' 필드 예시를 들어보자면 다음과 같아.
    
    Article Example:
    {{'evaluation': '긍정, 'summary': '${company_name}가 3분기 실적에서 견조한 매출을 기록 및 '헬시 플레저' 트렌드에 부합하는 제품 출시 및 좋은 성적 도출'}}

    Article:
    {article_content}

    반환 형식은 아래와 같아:
    Analysis = {{'evaluation': str, 'summary': str}}
    Return: Analysis
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)

        # candidates가 비어있는지 먼저 확인
        if not result.candidates or not result.candidates[0].content.parts:
            print(prompt);
            print(result);
            raise ValueError("API 응답이 비어 있습니다.")
        
        # API에서 반환된 텍스트 정리
        result_text = result.candidates[0].content.parts[0].text
        cleaned_response = result_text.replace("```json", "").replace("```", "").strip()
        
        # JSON 형식으로 변환
        analysis_result = json.loads(cleaned_response)
        analysis_result['link'] = article_link  # 링크는 결과에만 포함
        analysis_result['title'] = article_title  # 링크는 결과에만 포함
        
        return analysis_result  # 분석 결과 반환
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {
            "evaluation": "Error",
            "summary": "API 요청에 실패했습니다.",
            "link": article_link,  # 실패 시에도 링크를 반환
            "title": article_title
        }

# 뉴스 기사 분석 함수
def analyze_news(articles: List[Dict[str, str]], company_name: str) -> List[Dict[str, str]]:
    try:
        configure_gemini_api()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API 설정 오류: {str(e)}")
    
    analyzed_articles = []

    for article in articles:
        # 각 기사에 'content' 필드가 없거나 비어 있을 경우 처리
        if 'content' not in article or not article['content']:
            analyzed_articles.append({
                "evaluation": "Error",
                "summary": "기사 내용이 없습니다.",
                "link": article.get('link', 'unknown'),  # 링크 필드 추가
                "title": article.get('title', 'unknown')  # 링크 필드 추가
            })
            continue
        
        # 각 기사를 분석하여 결과를 추가
        analysis = analyze_article(article['content'], company_name, article['link'], article['title'])
        analyzed_articles.append(analysis)

    return analyzed_articles