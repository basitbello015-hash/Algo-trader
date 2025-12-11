import asyncio
import os
import time
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Bot controller & State
from bot_fib_scoring import ALLOWED_COINS
from app_state import bc  # Import global bot controller

# Routers
from routes.config_routes import router as config_router
from routes.accounts_routes import router as accounts_router
from routes.bot_routes import router as bot_router
from routes.dashboard_routes import router as dashboard_router
from routes.history_routes import router as history_router
from services.accounts_service import get_accounts
from services.config_service import get_config

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

APP_PORT = int(os.getenv("PORT", "8000"))
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() in ("1", "true", "yes")
PRICE_POLL_INTERVAL = float(os.getenv("PRICE_POLL_INTERVAL", "3.0"))

# -----------------------
# Initialize FastAPI
# -----------------------
app = FastAPI(
    title="MGX Crypto Trading Bot Backend",
    description="Backend API for MGX Crypto Trading Bot",
    version="1.0.0"
)

# -----------------------
# Templates
# -----------------------
templates = Jinja2Templates(directory="templates")

# -----------------------
# CORS (Frontend access)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Price Cache
# -----------------------
price_cache: Dict[str, float] = {}

# -----------------------
# WebSocket Connection Manager
# -----------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except:
                self.disconnect(ws)

ws_manager = ConnectionManager()

# -----------------------
# Root Endpoint (Dashboard)
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Serving dashboard.html as the main entry point
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/accounts", response_class=HTMLResponse)
async def accounts_page(request: Request):
    accounts = get_accounts()
    return templates.TemplateResponse("accounts.html", {"request": request, "accounts": accounts})

@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    config = get_config()
    return templates.TemplateResponse("config.html", {"request": request, "config": config})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        ws_manager.disconnect(websocket)

# -----------------------
# Price Fetch Loop
# -----------------------
async def price_loop():
    import requests

    while True:
        try:
            for sym in ALLOWED_COINS:
                try:
                    r = requests.get(
                        "https://api.bybit.com/v5/market/tickers",
                        params={"category": "spot", "symbol": sym},
                        timeout=10
                    )
                    j = r.json()

                    if j.get("retCode") == 0 and j.get("result", {}).get("list"):
                        price = float(j["result"]["list"][0]["lastPrice"])
                        price_cache[sym] = price

                        # WebSocket online price broadcast
                        await ws_manager.broadcast({
                            "type": "price",
                            "symbol": sym,
                            "price": price,
                            "timestamp": int(time.time())
                        })

                except Exception as e:
                    print(f"[Price Error] {sym}: {e}")

            await asyncio.sleep(PRICE_POLL_INTERVAL)

        except Exception as e:
            print(f"[Critical Price Loop Error] {e}")
            await asyncio.sleep(5)

# -----------------------
# Include All Routers
# -----------------------
app.include_router(config_router, prefix="/api/config", tags=["Config"])
app.include_router(accounts_router, prefix="/api/accounts", tags=["Accounts"])
app.include_router(bot_router, prefix="/api/bot", tags=["Bot"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(history_router, prefix="/api/history", tags=["History"])

# -----------------------
# App Startup Event
# -----------------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(price_loop())
    print("✅ MGX Trading Bot Backend started successfully!")

# -----------------------
# Test write endpoint
# -----------------------
@app.get("/api/test-write")
def test_write():
    try:
        test_file = "accounts_test.json"
        with open(test_file, "w") as f:
            f.write("test ok")

        return {
            "success": True,
            "message": f"Write OK: {os.path.abspath(test_file)}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
