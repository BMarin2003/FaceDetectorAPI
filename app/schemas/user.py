from fastapi import UploadFile
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional

class UserBase(BaseModel):
    id: int
    upaoID: int
    nombres: str
    apellidos: str
    correo: EmailStr
    requisitoriado: bool = False

class UserCreate(UserBase):
    foto: UploadFile

class UserUpdate(BaseModel):
    upaoID: Optional[int]
    nombres: Optional[str]
    apellidos: Optional[str]
    correo: Optional[str]
    requisitoriado: Optional[bool]
    conservar: Optional[bool] = False

class UserInDB(UserBase):
    id: int
    upaoID: int
    foto: Optional[HttpUrl] = None
    KP: Optional[HttpUrl] = None

    class Config:
        from_attributes = True