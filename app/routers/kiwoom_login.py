from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
import httpx
import uvicorn

app = FastAPI()
router = APIRouter()

KIWOOM_API_URL = "http://127.0.0.0:5000"

@router.get("/login")
@router.post("/login")
async def kiwoom_login():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{KIWOOM_API_URL}/login")
    if response.status_code == 200:
        return JSONResponse(content=response.json())
    else:
        raise HTTPException(status_code=response.status_code, detail="Login failed")

@router.get("/user_info")
async def kiwoom_user_info():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{KIWOOM_API_URL}/user_info")
    if response.status_code == 200:
        return JSONResponse(content=response.json())
    elif response.status_code == 400:
        raise HTTPException(status_code=400, detail="Not connected to Kiwoom API")
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to get user info from Kiwoom API")

app.include_router(router, prefix="/kiwoom")
