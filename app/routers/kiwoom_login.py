from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os

router = APIRouter()

KIWOOM_SERVER = os.environ.get('KIWOOM_SERVER', 'http://kiwoom-api:5000')

@router.get("/login")
@router.post("/login")
async def kiwoom_login():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{KIWOOM_SERVER}/login")
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Login failed")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Kiwoom API server is unavailable")