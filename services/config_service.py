import json
import os
from bot_fib_scoring import load_config

# Path to config file (under /data or relative to bot logic)
# bot_fib_scoring.py expects config.json in the same directory as itself
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "exchange": "Bybit",
    "strategy": "scalping",
    "stopLoss": -1.0,
    "maxHold": 86400,
    "rsiPeriod": 14,
    "rsiOversold": 35,
    "rsiOverbought": 65,
    "momentumScale": 1.0,
    "tradeAllocation": 100,
    "minTradeAmount": 5.0,
    "scanInterval": 10,
    "useMarketOrder": True,
    "testOnTestnet": False,
    "dryRun": False 
}

def ensure_config_exists():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)

def get_config():
    ensure_config_exists()
    return load_config()

def save_config(data: dict):
    ensure_config_exists()
    try:
        config = load_config()
        config.update(data)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return {"status": "saved", "data": config}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def reset_config():
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return {"status": "reset", "defaults": DEFAULT_CONFIG}
    except Exception as e:
        return {"status": "error", "message": str(e)}
