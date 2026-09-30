import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import models
from app.database import engine
from app.routers import subtitles # 다른 라우터도 여기에 추가

# 앱 실행 시 DB 테이블 자동 생성 (테이블 없을 경우)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="The One Point API")

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "*" # 개발 중에는 모든 곳에서 허용하는 것이 정신건강에 좋습니다.
]

# CORS 설정 (TV 앱이나 프론트엔드에서 접속 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 보안상 실제 배포시엔 프론트엔드 주소로 변경 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(subtitles.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to The One Point DX Server"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)