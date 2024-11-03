from fastapi import APIRouter,HTTPException
import httpx


router = APIRouter()

KIWOOM_SERVER = "http://localhost:5000"

@router.get('/kiwoom/login')
async def kiwoom_login():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{KIWOOM_SERVER}/login")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to log in to Kiwoom")
    return response.json()

@router.get('/kiwoom/user_info')
async def kiwoom_user_info():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{KIWOOM_SERVER}/user_info")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve user info from Kiwoom")
    return response.json()