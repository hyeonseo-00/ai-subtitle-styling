import sys
import os

# 현재 디렉토리(backend)를 파이썬 경로에 추가하여 app 모듈을 찾을 수 있게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine
from app import models

# 테이블 생성 (없으면 자동 생성)
models.Base.metadata.create_all(bind=engine)

def init_data():
    db = SessionLocal()
    
    try:
        # --- 1. 테스트 계정 생성 (Account) ---
        # user_id는 SQL 파일에 따라 int 타입입니다
        if not db.query(models.Account).filter_by(user_id=1001).first():
            print("Creating Test Account...")
            test_account = models.Account(
                user_id=1001,
                email="test@lg.com"
            )
            db.add(test_account)
            db.commit()
            db.refresh(test_account)
            
            # --- 2. 테스트 프로필 생성 (Profile) ---
            print("Creating Test Profile...")
            test_profile = models.Profile(
                account_id=test_account.id,
                nickname="User1",
                user_type="청각장애인", #  USER-01
                avatar_image="default_avatar.png"
            )
            db.add(test_profile)
            db.commit()
            db.refresh(test_profile)

            # --- 3. 자막 모드 프리셋 생성 (CaptionModeCustomizing) ---
            print("Inserting Default Caption Modes...")
            
            # (1) 없음 모드  SUB-01-01
            mode_none = models.CaptionModeCustomizing(
                profile_id=test_profile.id,
                mode_name="없음",
                is_empathy_on=False,
                # SQL 파일의 CamelCase 컬럼명 반영
                fontSize_toggle=False,
                fontColor_toggle=False,
                speaker=False,
                bgm=False,
                effect=False
            )
            
            # (2) 영화/드라마 모드  SUB-01-02
            mode_drama = models.CaptionModeCustomizing(
                profile_id=test_profile.id,
                mode_name="영화/드라마",
                is_empathy_on=True,
                font_size=24,
                fontSize_toggle=True,
                font_color="#FFFFFF",
                fontColor_toggle=True,
                font_level=2,  # 중간 강도
                color_level=2, # 중간 감정 색상
                speaker=True,  # 화자 구분 ON
                bgm=True,
                effect=False
            )

            # (3) 뉴스 모드  SUB-01-03
            mode_news = models.CaptionModeCustomizing(
                profile_id=test_profile.id,
                mode_name="뉴스",
                is_empathy_on=True,
                font_size=30,       # 크게
                fontSize_toggle=True,
                font_color="#FFFFFF",
                fontColor_toggle=True,
                font_level=1,       # 강도 변화 적음
                color_level=1,      # 색상 변화 적음
                speaker=False,
                bgm=False,
                effect=False
            )

            # (4) 예능 모드  SUB-01-04
            mode_variety = models.CaptionModeCustomizing(
                profile_id=test_profile.id,
                mode_name="예능",
                is_empathy_on=True,
                font_size=28,
                fontSize_toggle=True,
                font_color="#FFD700", # 골드/노랑 계열
                fontColor_toggle=True,
                font_level=3,       # 강도 변화 큼
                color_level=3,      # 색상 변화 큼
                speaker=True,
                bgm=True,
                effect=True         # 효과음 ON
            )

            db.add_all([mode_none, mode_drama, mode_news, mode_variety])
            db.commit()

            # --- 4. 현재 설정 매핑 (ProfileCaptionSetting) ---
            print("Setting Default Mode to 'Drama'...")
            # 방금 넣은 드라마 모드의 ID를 참조
            current_setting = models.ProfileCaptionSetting(
                profile_id=test_profile.id,
                mode_id=mode_drama.id
            )
            db.add(current_setting)
            db.commit()
            
            print("✅ Initial data setup completed successfully!")
        else:
            print("ℹ️ Data already exists. Skipping initialization.")
            
    except Exception as e:
        print(f"❌ Error during init: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_data()