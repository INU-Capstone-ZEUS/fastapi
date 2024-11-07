# from fastapi import APIRouter, HTTPException
# from fastapi.responses import JSONResponse
# import httpx
# import os

# router = APIRouter()

# KIWOOM_SERVER = os.environ.get('KIWOOM_SERVER', 'http://kiwoom-api:5000')

# @router.get("/login")
# @router.post("/login")
# async def kiwoom_login():
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.post(f"{KIWOOM_SERVER}/login")
#             response.raise_for_status()
#             return JSONResponse(content=response.json())
#         except httpx.HTTPStatusError as e:
#             raise HTTPException(status_code=e.response.status_code, detail="Login failed")
#         except httpx.RequestError:
#             raise HTTPException(status_code=503, detail="Kiwoom API server is unavailable")

# fastapi_server.py (64비트 Python 환경에서 실행)
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import requests

app = FastAPI()
router = APIRouter()

class StockRequest(BaseModel):
    code_name: str
    count: int
    tick_range: int
    mT: Optional[str] = 'm'

class DateRangeRequest(BaseModel):
    code_name: str
    today: str
    recent_day: str
    mT: Optional[str] = None


@router.get("/stock_code/{name}")
async def get_stock_code(name: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:5000/get_stock_code/{name}", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Stock not found")
        else:
            raise HTTPException(status_code=500, detail=f"Error from Flask server: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Flask server: {str(e)}")

@router.post("/search_name_list_by_name")
async def search_name_list_by_name(name: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/search_name_list_by_name", json={"name": name})
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to search name list")

@router.post("/search_name_list_by_code")
async def search_name_list_by_code(code: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/search_name_list_by_code", json={"code": code})
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to search name list")

@router.post("/get_recent_data")
async def get_recent_data(request: StockRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/get_recent_data", json=request.dict())
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to get recent data")

@router.post("/get_day_data")
async def get_day_data(request: StockRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/get_day_data", json=request.dict())
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to get day data")

@router.post("/get_minute_or_tick_data")
async def get_minute_or_tick_data(request: StockRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/get_minute_or_tick_data", json=request.dict())
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to get minute or tick data")

@router.post("/get_update_period_day")
async def get_update_period_day(request: DateRangeRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/get_update_period_day", json=request.dict())
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to get update period day data")

@router.post("/get_update_period_minutes")
async def get_update_period_minutes(request: DateRangeRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/get_update_period_minutes", json=request.dict())
    if response.status_code == 200:
        return response.json()
    raise HTTPException(status_code=500, detail="Failed to get update period minutes data")

app.include_router(router, prefix="/api/vi")
