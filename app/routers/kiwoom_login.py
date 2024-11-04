from fastapi import APIRouter
from pykiwoom.kiwoom import Kiwoom
import time
import threading

router = APIRouter()
kiwoom = None

def keep_alive_session():
    global kiwoom
    while True:
        state = kiwoom.GetConnectState()
        if state == 0:
            kiwoom.CommConnect(block=True)
        time.sleep(300)

@router.post("/login")
def login():
    global kiwoom
    kiwoom = Kiwoom()
    kiwoom.CommConnect(block=True)
    threading.Thread(target=keep_alive_session, daemon=True).start()
    return {"message": "로그인 완료"}
