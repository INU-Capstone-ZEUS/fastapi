from fastapi import APIRouter,HTTPException
import httpx
import requests
import time
import threading

# def keep_alive_session():
#     global kiwoom
#     while True:
#         state = kiwoom.GetConnectState()
#         if state == 0:
#             kiwoom.CommConnect(block=True)
#         time.sleep(300)

# @router.post("/login")
# def login():
#     global kiwoom
    
#     kiwoom.CommConnect(block=True)
#     threading.Thread(target=keep_alive_session, daemon=True).start()
#     return {"message": "로그인 완료"}


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