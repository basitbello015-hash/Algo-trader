# routes/dashboard_router.py

from fastapi import APIRouter
from services.dashboard_service import get_dashboard_data

router = APIRouter()

@router.get("/")
def dashboard():
    """
    Returns live dashboard statistics for the frontend.
    Example:
    {
        "profit": 124.5,
        "openTrades": 2,
        "balance": 1050.75,
        "dailyChange": 3.1
    }
    """
    return get_dashboard_data()
