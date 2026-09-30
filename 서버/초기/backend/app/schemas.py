from pydantic import BaseModel
from typing import Optional, List # Optional이 꼭 있어야 합니다!
from datetime import datetime

# 자막 모드 기본 스키마
class CaptionModeBase(BaseModel):
    mode_name: str
    is_empathy_on: bool = False
    
    # [수정] int -> Optional[int] = None 으로 변경
    # DB에서 값이 없어도(None) 에러가 나지 않게 합니다.
    font_size: Optional[int] = 20
    fontSize_toggle: bool = False
    font_color: Optional[str] = "#FFFFFF"
    fontColor_toggle: bool = False
    
    # ▼▼▼ 여기가 에러가 났던 부분입니다! ▼▼▼
    font_level: Optional[int] = 1   # int -> Optional[int]
    color_level: Optional[int] = 1  # int -> Optional[int]
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    speaker: bool = False
    bgm: bool = False
    effect: bool = False

# ... 아래 코드는 그대로 두세요 ...
class CaptionModeUpdate(BaseModel):
    font_size: Optional[int] = None
    fontSize_toggle: Optional[bool] = None
    font_color: Optional[str] = None
    fontColor_toggle: Optional[bool] = None
    font_level: Optional[int] = None
    color_level: Optional[int] = None
    speaker: Optional[bool] = None
    bgm: Optional[bool] = None
    effect: Optional[bool] = None

class CaptionModeResponse(CaptionModeBase):
    id: int
    profile_id: int
    updated_at: Optional[datetime]
    
    class Config:
        orm_mode = True