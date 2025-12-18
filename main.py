import asyncio
import os
import time
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

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
# Global State
# -----------------------
bot_started = False  # Prevent multiple bot startups
price_cache: Dict[str, float] = {}

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

APP_PORT = int(os.getenv("PORT", "8000"))
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() in ("1", "true", "yes")
PRICE_POLL_INTERVAL = float(os.getenv("PRICE_POLL_INTERVAL", "3.0"))
KEEP_ALIVE_INTERVAL = 300  # 5 minutes for free tier prevention

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
# WebSocket Connection Manager
# -----------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self.price_subscribers: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        if ws in self.price_subscribers:
            self.price_subscribers.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except:
                self.disconnect(ws)

    async def broadcast_prices(self, message: dict):
        """Send price updates to all price subscribers"""
        for ws in list(self.price_subscribers):
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
        # Send initial price cache if available
        if price_cache:
            await websocket.send_json({
                "type": "price_batch",
                "prices": price_cache,
                "timestamp": int(time.time())
            })
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/prices")
async def price_websocket_endpoint(websocket: WebSocket):
    """WebSocket specifically for price updates"""
    await websocket.accept()
    ws_manager.price_subscribers.append(websocket)
    
    try:
        # Send initial price cache
        if price_cache:
            await websocket.send_json({
                "type": "price_batch",
                "prices": price_cache,
                "timestamp": int(time.time())
            })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Optional: Handle specific price subscription requests
            if data.startswith("subscribe:"):
                symbol = data.split(":")[1]
                # Handle specific symbol subscription
                pass
    except:
        ws_manager.disconnect(websocket)

@app.get("/api/bot/debug")
def bot_debug():
    """Debug endpoint to check bot status and configuration"""
    from app_state import bc
    
    accounts = bc.load_accounts() if hasattr(bc, 'load_accounts') else []
    
    return {
        "bot_running": bc.is_running() if hasattr(bc, 'is_running') else False,
        "accounts_count": len(accounts),
        "accounts": [
            {
                "name": a.get("name"),
                "monitoring": a.get("monitoring"),
                "position": a.get("position"),
                "validated": a.get("validated"),
                "balance": a.get("balance")
            }
            for a in accounts
        ],
        "config": {
            "dry_run": TRADE_SETTINGS.get("dry_run", True),
            "use_market_order": TRADE_SETTINGS.get("use_market_order", False),
            "trade_allocation_pct": TRADE_SETTINGS.get("trade_allocation_pct", 100),
            "max_trades_per_day": TRADE_SETTINGS.get("max_trades_per_day", 30)
        },
        "timestamp": datetime.now().isoformat()
    }
# -----------------------
# Price Fetch Loop
# -----------------------
async def price_loop():
    import requests
    
    print(f"[Price Loop] Started. Polling {len(ALLOWED_COINS)} coins every {PRICE_POLL_INTERVAL}s")
    
    while True:
        try:
            updated_count = 0
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
                        old_price = price_cache.get(sym)
                        price_cache[sym] = price
                        updated_count += 1

                        # Broadcast to WebSocket subscribers
                        await ws_manager.broadcast_prices({
                            "type": "price",
                            "symbol": sym,
                            "price": price,
                            "old_price": old_price,
                            "change": ((price - old_price) / old_price * 100) if old_price else 0,
                            "timestamp": int(time.time())
                        })

                except Exception as e:
                    print(f"[Price Error] {sym}: {e}")
            
            # Log progress occasionally
            if updated_count > 0 and time.time() % 30 < PRICE_POLL_INTERVAL:  # Log every ~30 seconds
                print(f"[Price Loop] Updated {updated_count}/{len(ALLOWED_COINS)} prices")
                
            await asyncio.sleep(PRICE_POLL_INTERVAL)

        except Exception as e:
            print(f"[Critical Price Loop Error] {e}")
            await asyncio.sleep(5)

# -----------------------
# Keep Alive Background Task (for Render free tier)
# -----------------------
async def keep_alive_task():
    """Ping our own service to prevent Render from spinning down"""
    import aiohttp
    
    while True:
        try:
            # Wait a bit before starting keep-alive
            await asyncio.sleep(60)
            
            app_url = os.getenv("RENDER_EXTERNAL_URL", "")
            if app_url:
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{app_url}/ping", timeout=10) as resp:
                            if resp.status == 200:
                                print(f"[Keep Alive] Ping successful at {datetime.now().strftime('%H:%M:%S')}")
                    except Exception as e:
                        print(f"[Keep Alive] Error: {e}")
        except Exception as e:
            print(f"[Keep Alive Task Error] {e}")
        
        await asyncio.sleep(KEEP_ALIVE_INTERVAL)

# -----------------------
# Include All Routers
# -----------------------
app.include_router(config_router, prefix="/api/config", tags=["Config"])
app.include_router(accounts_router, prefix="/api/accounts", tags=["Accounts"])
app.include_router(bot_router, prefix="/api/bot", tags=["Bot"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(history_router, prefix="/api/history", tags=["History"])

# -----------------------
# Utility Endpoints
# -----------------------
@app.get("/ping")
async def ping():
    """Health check endpoint to keep Render instance alive"""
    return {
        "status": "alive", 
        "timestamp": datetime.now().isoformat(),
        "price_cache_size": len(price_cache),
        "bot_started": bot_started
    }

@app.get("/api/prices")
def get_prices():
    """Debug endpoint to check price cache"""
    return {
        "prices": price_cache,
        "count": len(price_cache),
        "coins": ALLOWED_COINS,
        "timestamp": int(time.time()),
        "last_update": max([price_cache.get(sym, 0) for sym in ALLOWED_COINS if sym in price_cache]) if price_cache else None
    }

@app.get("/api/system-status")
async def system_status():
    """Get overall system status"""
    return {
        "bot_started": bot_started,
        "price_feed_running": "price_cache" in globals() and len(price_cache) > 0,
        "websocket_connections": len(ws_manager.active),
        "price_subscribers": len(ws_manager.price_subscribers),
        "server_time": datetime.now().isoformat(),
        "config": {
            "price_poll_interval": PRICE_POLL_INTERVAL,
            "live_mode": LIVE_MODE,
            "port": APP_PORT
        }
    }

# -----------------------
# App Startup Event
# -----------------------
@app.on_event("startup")
async def startup_event():
    """Auto-start bot and price feed on server startup"""
    global bot_started
    
    if bot_started:
        print("Bot already started, skipping...")
        return
        
    try:
        print("=" * 50)
        print("Starting MGX Crypto Trading Bot")
        print(f"Live Mode: {LIVE_MODE}")
        print(f"Price Poll Interval: {PRICE_POLL_INTERVAL}s")
        print("=" * 50)
        
        # 1. Start the bot trading logic
        print("Starting bot controller...")
        bc.start()
        bot_started = True
        print("✓ Bot controller started successfully")
        
        # 2. Start the price fetching loop in the background
        print("Starting price feed...")
        asyncio.create_task(price_loop())
        print("✓ Price feed loop started")
        
        # 3. Start keep-alive task for Render free tier
        print("Starting keep-alive service...")
        asyncio.create_task(keep_alive_task())
        print("✓ Keep-alive service started")
        
        print("=" * 50)
        print("Startup complete! All systems operational.")
        print("=" * 50)
            
    except Exception as e:
        print(f"✗ Failed during startup: {str(e)}")
        import traceback
        traceback.print_exc()

# -----------------------
# App Shutdown Event
# -----------------------
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down bot and services...")
    # Add any cleanup logic here for bot, connections, etc.
    print("✓ Shutdown complete")

# -----------------------
# Test write endpoint
# -----------------------
@app.get("/api/test-write")
def test_write():
    try:
        test_file = "accounts_test.json"
        with open(test_file, "w") as f:
            f.write(f"test ok at {datetime.now().isoformat()}")

        return {
            "success": True,
            "message": f"Write OK: {os.path.abspath(test_file)}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on port {APP_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
