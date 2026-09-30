import sys
import os
from app.database import SessionLocal
from app import models

# DB 연결
db = SessionLocal()

def check_current_mode(profile_id):
    # 1. ProfileCaptionSetting 테이블(주문서)을 조회함 
    setting = db.query(models.ProfileCaptionSetting).filter(
        models.ProfileCaptionSetting.profile_id == profile_id
    ).first()

    if setting:
        # 2. 현재 저장된 모드 ID와 정보를 가져옴
        mode = db.query(models.CaptionModeCustomizing).filter(
            models.CaptionModeCustomizing.id == setting.mode_id
        ).first()
        
        print(f"=== {profile_id}번 프로필의 현재 상태 ===")
        print(f"👉 저장된 모드 ID: {setting.mode_id}")
        print(f"👉 모드 이름: {mode.mode_name}")
        print(f"👉 감성 모드 여부: {mode.is_empathy_on}")
        print("================================")
    else:
        print(f"❌ {profile_id}번 프로필의 설정이 없습니다.")

if __name__ == "__main__":
    check_current_mode(profile_id=1) # 1번 유저 확인