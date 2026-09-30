from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Account(Base):
    __tablename__ = "account"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True) #
    email = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    profiles = relationship("Profile", back_populates="account")

class Profile(Base):
    __tablename__ = "profile"
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("account.id"))
    nickname = Column(String(50))
    avatar_image = Column(String(255))
    user_type = Column(String(30))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    account = relationship("Account", back_populates="profiles")
    custom_modes = relationship("CaptionModeCustomizing", back_populates="profile")
    current_setting = relationship("ProfileCaptionSetting", uselist=False, back_populates="profile")

class CaptionModeCustomizing(Base):
    __tablename__ = "caption_mode_customizing"
    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("profile.id"))
    mode_name = Column(String(50))
    is_empathy_on = Column(Boolean, default=False)
    
    font_size = Column(Integer)
    fontSize_toggle = Column(Boolean, default=False) # CamelCase 유지
    font_color = Column(String(10))
    fontColor_toggle = Column(Boolean, default=False)
    
    font_level = Column(Integer)
    color_level = Column(Integer)
    
    speaker = Column(Boolean, default=False)
    bgm = Column(Boolean, default=False)
    effect = Column(Boolean, default=False)
    
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    profile = relationship("Profile", back_populates="custom_modes")

class ProfileCaptionSetting(Base):
    __tablename__ = "profile_caption_setting"
    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("profile.id"))
    mode_id = Column(Integer, ForeignKey("caption_mode_customizing.id"))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    profile = relationship("Profile", back_populates="current_setting")