from fastapi import FastAPI

from app.routes import users

FaceDetectorAPI = FastAPI()

FaceDetectorAPI.include_router(users.router)