from fastapi import APIRouter
from services.bot_service import get_status, start_bot, stop_bot

router = APIRouter()

@router.get("/status")
def bot_status():
    return get_status()

@router.post("/start")
def start():
    return start_bot()

@router.post("/stop")
def stop():
    return stop_bot()
    
