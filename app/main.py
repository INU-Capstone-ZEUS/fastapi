import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import IP_NUM, PORT_NUM
from app.routers.root import router as root_router
from app.routers.crawl_and_analyze import router as crawl_router
from app.routers.model_predict import router as model_router
from app.routers.kiwoom_login import router as kiwoom_router

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 (GET, POST, PUT, DELETE 등)를 허용
    allow_headers=["*"],  # 모든 헤더를 허용
)

# 라우터 포함
app.include_router(root_router)
app.include_router(crawl_router)
app.include_router(model_router)
#app.include_router(kiwoom_router, prefix='/kiwoom', tags=['kiwoom'])

if __name__ == '__main__':
    uvicorn.run(
        "app.main:app", 
        host=IP_NUM, 
        port=int(PORT_NUM), 
        workers=1, 
        ssl_keyfile="/etc/ssl/private/privkey.pem", 
        ssl_certfile="/etc/ssl/certs/fullchain.pem"
    )