from app_state import bc

def get_status():
    """
    Returns the current status of the bot controller.
    """
    running = bc.is_running()
    
    # Get active strategy/symbol info from config or state
    trades = bc._read_trades()
    active_symbols = list(set(t.get("symbol") for t in trades if t.get("open")))
    
    return {
        "running": running,
        "active_symbols": active_symbols,
        "strategy": "Fibonacci Scoring",
        "uptime": "Running" if running else "Stopped"
    }

def start_bot():
    """
    Signals the bot to start trading.
    """
    if bc.is_running():
         return {"status": "info", "message": "Bot is already running"}
    
    bc.start()
    return {"status": "success", "message": "Bot started successfully"}

def stop_bot():
    """
    Signals the bot to stop trading.
    """
    if not bc.is_running():
        return {"status": "info", "message": "Bot is already stopped"}
        
    bc.stop()
    return {"status": "success", "message": "Bot stopped successfully"}
