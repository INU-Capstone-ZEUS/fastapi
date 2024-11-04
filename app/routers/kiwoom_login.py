from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import httpx
import socket

router = APIRouter()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

KIWOOM_API_IP = get_local_ip()
KIWOOM_API_PORT = 5000
KIWOOM_API_URL = f"http://{KIWOOM_API_IP}:{KIWOOM_API_PORT}"

@router.get("/login")
@router.post("/login")
async def kiwoom_login():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{KIWOOM_API_URL}/login")
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Login failed")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Kiwoom API server is unavailable")

@router.get("/user_info")
async def kiwoom_user_info():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{KIWOOM_API_URL}/user_info")
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise HTTPException(status_code=400, detail="Not connected to Kiwoom API")
            else:
                raise HTTPException(status_code=e.response.status_code, detail="Failed to get user info from Kiwoom API")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Kiwoom API server is unavailable")