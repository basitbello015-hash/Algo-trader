"""
Rewritten and cleaned strategy controller for Superb Crypto Bot.

Key features:
- 1-minute timeframe (interval = '1')
- Max hold time = 30 minutes (1800 seconds)
- Wilder RSI, smoothed momentum, EMA trend filter
- Safer Fibonacci pivots (exclude last candle from pivot calc)
- Take profit on Fibonacci extensions (1.272 -> 1.618 -> 2.618)
- Stop loss = 1% below swing low (ensures SL below entry)
- Order response validation, retry wrapper, file locking
- Dry-run support with real market data
- Daily trade limit (30 trades/24h cycle)
- Rate limiting for API calls
- Order book analysis for slippage protection
- Position size validation
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque

# Exchange client used in original repo
from pybit.unified_trading import HTTP

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use config.json if present for override; otherwise defaults below
def load_config() -> Dict[str, Any]:
    cfg_path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

CONFIG = load_config()

# Timeframe: '1' means 1 minute
TIMEFRAME = CONFIG.get("timeframe", "1")

# Keep allowed coins as before
ALLOWED_COINS = CONFIG.get("allowed_coins", [
    "ADAUSDT", "XRPUSDT", "TRXUSDT", "DOGEUSDT", "CHZUSDT", "VETUSDT", "BTTUSDT", "HOTUSDT",
    "XLMUSDT", "ZILUSDT", "IOTAUSDT", "SCUSDT", "DENTUSDT", "KEYUSDT", "WINUSDT", "CVCUSDT",
    "MTLUSDT", "CELRUSDT", "FUNUSDT", "STMXUSDT", "REEFUSDT", "ANKRUSDT", "ONEUSDT", "OGNUSDT",
    "CTSIUSDT", "DGBUSDT", "CKBUSDT", "ARPAUSDT", "MBLUSDT", "TROYUSDT", "PERLUSDT", "DOCKUSDT",
    "RENUSDT", "COTIUSDT", "MDTUSDT", "OXTUSDT", "PHAUSDT", "BANDUSDT", "GTOUSDT", "LOOMUSDT",
    "PONDUSDT", "FETUSDT", "SYSUSDT", "TLMUSDT", "NKNUSDT", "LINAUSDT", "ORNUSDT", "COSUSDT",
    "FLMUSDT", "ALICEUSDT",
])

# Risk rules
RISK_RULES = {
    # stop_loss percent is not used directly for SL price; we use swing low * (1 - 0.01)
    "stop_loss_pct": CONFIG.get("stopLossPct", 1.0),  # percent
    "max_hold": int(CONFIG.get("maxHoldSeconds", 30 * 60)),  # 30 minutes default (1800 s)
    "max_slippage_pct": CONFIG.get("maxSlippagePct", 0.5),  # 0.5% max slippage
    "min_order_book_depth_usd": CONFIG.get("minOrderBookDepthUsd", 1000.0),  # $1000 min depth
}

# Score and strategy settings
SCORE_SETTINGS = {
    "rsi_period": int(CONFIG.get("rsiPeriod", 14)),
    "rsi_oversold_threshold": float(CONFIG.get("rsiOversold", 35.0)),
    "momentum_entry_threshold_pct": float(CONFIG.get("momentumEntryThreshold", 0.1)),
    "momentum_strong_pct": float(CONFIG.get("momentumStrong", 0.5)),
    "momentum_very_strong_pct": float(CONFIG.get("momentumVeryStrong", 1.5)),
    "fib_lookback": int(CONFIG.get("fibLookback", 50)),
    "score_weights": {
        "rsi": int(CONFIG.get("scoreWeightRsi", 1)),
        "momentum": int(CONFIG.get("scoreWeightMomentum", 1)),
        "ema": int(CONFIG.get("scoreWeightEma", 1)),
        "candle": int(CONFIG.get("scoreWeightCandle", 1)),
        "fib_zone": int(CONFIG.get("scoreWeightFibZone", 1)),
    }
}

TRADE_SETTINGS = {
    "trade_allocation_pct": CONFIG.get("tradeAllocation", 100),
    "min_trade_amount": float(CONFIG.get("minTradeAmount", 5.0)),
    "use_market_order": CONFIG.get("useMarketOrder", True),
    "test_on_testnet": CONFIG.get("testOnTestnet", False),
    "scan_interval": int(CONFIG.get("scanInterval", 10)),
    "debug_raw_responses": CONFIG.get("debugRawResponses", False),
    "dry_run": CONFIG.get("dryRun", False),
    "max_trades_per_day": int(CONFIG.get("maxTradesPerDay", 30)),
}

# Rate limiting settings
RATE_LIMIT_SETTINGS = {
    "max_requests_per_second": CONFIG.get("maxRequestsPerSecond", 10),
    "max_requests_per_minute": CONFIG.get("maxRequestsPerMinute", 100),
}

# Adjusted paths to match existing project structure (app/ instead of accounts/)
ACCOUNTS_FILE = os.path.join(BASE_DIR, "app/data/accounts.json")
TRADES_FILE = os.path.join(BASE_DIR, "app/data/trades.json")

# Ensure app directory exists
os.makedirs(os.path.join(BASE_DIR, "app/data"), exist_ok=True)

# Ensure files exist
for f in (ACCOUNTS_FILE, TRADES_FILE):
    if not os.path.exists(f):
        with open(f, "w") as fh:
            fh.write("[]")

# -------------------- Rate Limiter --------------------
class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, max_per_second: int = 10, max_per_minute: int = 100):
        self.max_per_second = max_per_second
        self.max_per_minute = max_per_minute
        self.second_tokens = max_per_second
        self.minute_tokens = max_per_minute
        self.last_second_refill = time.time()
        self.last_minute_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens. Returns True if successful, False if rate limit exceeded.
        Blocks until tokens are available.
        """
        with self.lock:
            current_time = time.time()
            
            # Refill second bucket
            if current_time - self.last_second_refill >= 1.0:
                self.second_tokens = self.max_per_second
                self.last_second_refill = current_time
            
            # Refill minute bucket
            if current_time - self.last_minute_refill >= 60.0:
                self.minute_tokens = self.max_per_minute
                self.last_minute_refill = current_time
            
            # Check if we have enough tokens
            if self.second_tokens >= tokens and self.minute_tokens >= tokens:
                self.second_tokens -= tokens
                self.minute_tokens -= tokens
                return True
            
            # Wait and retry
            wait_time = min(
                1.0 - (current_time - self.last_second_refill),
                60.0 - (current_time - self.last_minute_refill)
            )
            if wait_time > 0:
                time.sleep(wait_time)
                return self.acquire(tokens)
            
            return False

# -------------------- Utilities --------------------
def now_ts() -> int:
    return int(time.time())

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def fmt_elapsed(sec: int) -> str:
    m = sec // 60
    s = sec % 60
    return f"{m}m {s}s"

def safe_json(obj: Any, max_depth: int = 4, _depth: int = 0, _seen: Optional[set] = None) -> Any:
    if _seen is None:
        _seen = set()
    try:
        if id(obj) in _seen or _depth > max_depth:
            return "<recursion>"
        _seen.add(id(obj))
    except Exception:
        pass
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = "<key>"
            out[key] = safe_json(v, max_depth, _depth + 1, _seen)
        return out
    if isinstance(obj, (list, tuple)):
        return [safe_json(v, max_depth, _depth + 1, _seen) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

# -------------------- Indicator helpers --------------------
def calc_ema(values: List[float], period: int) -> Optional[float]:
    if not values or len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for p in values[period:]:
        ema = p * k + ema * (1 - k)
    return ema

def wilder_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if not closes or len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    # initial window
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses += -d
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    # Wilder smoothing
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def smoothed_momentum_pct(closes: List[float], lookback: int = 5, smooth_span: int = 3) -> float:
    if not closes or len(closes) < lookback + 1:
        return 0.0
    raw = []
    for i in range(lookback, len(closes)):
        denom = closes[i - lookback] if closes[i - lookback] != 0 else 1.0
        raw.append(((closes[i] - closes[i - lookback]) / denom) * 100.0)
    if not raw:
        return 0.0
    alpha = 2.0 / (smooth_span + 1.0)
    ema = raw[0]
    for v in raw[1:]:
        ema = v * alpha + ema * (1.0 - alpha)
    return float(ema)

def calc_fib_levels(high: float, low: float) -> Dict[str, float]:
    diff = high - low
    return {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low,
        '1.272_ext': high + 1.272 * diff,
        '1.618_ext': high + 1.618 * diff,
        '2.0_ext': high + 2.0 * diff,
        '2.618_ext': high + 2.618 * diff,
    }

def price_in_zone(price: float, levels: Dict[str, float], lo_key: str = '0.382', hi_key: str = '0.618') -> bool:
    try:
        lo = levels[lo_key]; hi = levels[hi_key]
        return min(lo, hi) <= price <= max(lo, hi)
    except Exception:
        return False

def detect_bullish_candle(candles: List[Dict[str, float]]) -> bool:
    if not candles:
        return False
    last = candles[-1]
    if len(candles) >= 2:
        prev = candles[-2]
        if prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open'] and last['open'] < prev['close']:
            return True
    body = abs(last['close'] - last['open'])
    lower_wick = (last['open'] - last['low']) if last['open'] > last['close'] else (last['close'] - last['low'])
    upper_wick = last['high'] - max(last['open'], last['close'])
    if body > 0 and lower_wick / body >= 2 and upper_wick / body <= 0.5:
        return True
    return False

def pivot_fib_levels_from_confirmed_window(highs: List[float], lows: List[float], lookback: int = 50) -> Dict[str, float]:
    """
    Safer pivot: use lookback window but exclude most recent candle.
    """
    if not highs or not lows:
        return {}
    if len(highs) < lookback:
        lookback = len(highs)
    # exclude last candle
    slice_highs = highs[-lookback - 1:-1] if len(highs) >= lookback + 1 else highs[:-1] if len(highs) > 1 else highs
    slice_lows = lows[-lookback - 1:-1] if len(lows) >= lookback + 1 else lows[:-1] if len(lows) > 1 else lows
    if not slice_highs or not slice_lows:
        slice_highs = highs[-lookback:]
        slice_lows = lows[-lookback:]
    swing_high = max(slice_highs)
    swing_low = min(slice_lows)
    return calc_fib_levels(swing_high, swing_low)

# -------------------- Bot Controller (enhanced) --------------------
class BotController:
    def __init__(self, log_queue: Optional[threading.Queue] = None):
        self.log_queue = log_queue
        self._running = False 
        self._stop = threading.Event()
        self._file_lock = threading.Lock()
        self._threads: List[threading.Thread] = []

        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_per_second=RATE_LIMIT_SETTINGS["max_requests_per_second"],
            max_per_minute=RATE_LIMIT_SETTINGS["max_requests_per_minute"]
        )

        # Daily limit tracking
        self.trades_today = 0
        self.day_start_time = time.time()
        self.MAX_TRADES_DAILY = TRADE_SETTINGS.get("max_trades_per_day", 30)

        # Dry-run simulated balances (symbol -> balance)
        self.dry_run_balances: Dict[str, float] = {}

        # ensure account files exist
        for path, default in ((ACCOUNTS_FILE, []), (TRADES_FILE, [])):
            if not os.path.exists(path):
                try:
                    with open(path, "w") as fh:
                        json.dump(default, fh, indent=2)
                except Exception as e:
                    print(f"Error creating {path}: {e}")

    # ------------------ Logging ------------------
    def log(self, msg: str):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.log_queue:
            try:
                self.log_queue.put(line, block=False)
            except Exception:
                pass

     # ------------------ Daily Limit Check ------------------                      
    def _check_daily_limit(self) -> bool:
        """
        Returns True if the daily trade limit is reached.
        Does NOT stop the bot — just signals that trades should be skipped.
        Resets counter if 24 hours have passed since last reset.
        """
        current_time = time.time()

        # Initialize if missing
        if not hasattr(self, "day_start_time"):
            self.day_start_time = current_time
            self.trades_today = 0

        # Reset counter after 24h
        if current_time - self.day_start_time > 86400:
            self.log("24 hours passed. Resetting daily trade counter.")
            self.day_start_time = current_time
            self.trades_today = 0

        # Ensure MAX_TRADES_DAILY exists
        if not hasattr(self, "MAX_TRADES_DAILY") or self.MAX_TRADES_DAILY <= 0:
            self.MAX_TRADES_DAILY = 10  # default value

        # Return True if limit reached
        limit_reached = self.trades_today >= self.MAX_TRADES_DAILY
        if limit_reached:
            self.log(f"Daily trade limit reached ({self.trades_today}/{self.MAX_TRADES_DAILY})")

        return limit_reached

    # ------------------ Account file helpers (locked) ------------------
    def load_accounts(self) -> List[Dict[str, Any]]:
        try:
            with self._file_lock:
                with open(ACCOUNTS_FILE, "r") as fh:
                    return json.load(fh)
        except Exception as e:
            self.log(f"load_accounts error: {e}")
            return []

    def save_accounts(self, accounts: List[Dict[str, Any]]):
        try:
            with self._file_lock:
                with open(ACCOUNTS_FILE, "w") as fh:
                    json.dump(accounts, fh, indent=2)
        except Exception as e:
            self.log(f"save_accounts error: {e}")
            # Raise exception so API knows it failed (matching previous logic)
            raise RuntimeError(f"Failed to save accounts: {e}")

    def _read_trades(self) -> List[Dict[str, Any]]:
        try:
            with self._file_lock:
                with open(TRADES_FILE, "r") as fh:
                    return json.load(fh)
        except Exception as e:
            self.log(f"_read_trades error: {e}")
            return []

    def _write_trades(self, trades: List[Dict[str, Any]]):
        try:
            with self._file_lock:
                with open(TRADES_FILE, "w") as fh:
                    json.dump(trades, fh, indent=2)
        except Exception as e:
            self.log(f"_write_trades error: {e}")

    def add_trade(self, trade: Dict[str, Any]):
        try:
            trades = self._read_trades()
            trades.append(safe_json(trade, max_depth=6))
            if len(trades) > 500:
                trades = trades[-500:]
            self._write_trades(trades)
        except Exception as e:
            self.log(f"add_trade error: {e}")

    def is_running(self) -> bool:
        """Check if the bot is currently running."""
        return self._running

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        try:
            changed = False
            trades = self._read_trades()
            for t in trades:
                if t.get("id") == trade_id:
                    t.update(safe_json(updates, max_depth=6))
                    changed = True
                    break
            if changed:
                self._write_trades(trades)
            return changed
        except Exception as e:
            self.log(f"update_trade error: {e}")
            return False

    # ------------------ Client wrapper ------------------
    def _get_client(self, account: Dict[str, Any]) -> Optional[HTTP]:
        """
        Return a pybit HTTP client if all required fields exist.
        For dry-run mode, still create client to fetch real market data.
        """
        try:
            # Dashboard fields
            account_id = account.get("id")
            account_name = account.get("name")
            exchange = account.get("exchange")

             # API credentials (dashboard naming)
            key = (
                account.get("api_key")
                or account.get("key")
                or account.get("apiKey")
            )

            secret = (
                account.get("api_secret")
                or account.get("secret")
                or account.get("apiSecret")
                or account.get("secretKey")
            )

            # Validate required fields
            if not account_id:
                self.log("Missing account id")
                return None

            if not account_name:
                self.log("Missing account name")
                return None

            if not exchange:
                self.log("Missing exchange field")
                return None

            if not key or not secret:
                self.log("Missing API credentials")
                return None

            if exchange.lower() != "bybit":
                self.log(f"Unsupported exchange: {exchange}")
                return None

            # Create client (use mainnet even in dry-run to get real data)
            client = HTTP(
                api_key=key,
                api_secret=secret,
                testnet=TRADE_SETTINGS.get("test_on_testnet", False) and not TRADE_SETTINGS.get("dry_run", False)
            )

            self.log(f"Client created for {account_name} (ID: {account_id})")

            return client

        except Exception as e:
            self.log(f"_get_client error: {e}")
            return None

    # ------------------ API retry wrapper with rate limiting ------------------
    def _retry(self, fn: Callable[..., Any], attempts: int = 3, base_delay: float = 0.5, *args, **kwargs):
        last_exc = None
        for i in range(attempts):
            try:
                # Acquire rate limit token
                self.rate_limiter.acquire()
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                delay = base_delay * (i + 1)
                time.sleep(delay)
        raise last_exc

    # ------------------ Order book analysis ------------------
    def _get_order_book(self, client: HTTP, symbol: str, limit: int = 25) -> Optional[Dict[str, Any]]:
        """
        Fetch order book data for slippage analysis.
        """
        try:
            self.rate_limiter.acquire()
            resp = client.get_orderbook(category="spot", symbol=symbol, limit=limit)
            if isinstance(resp, dict) and resp.get("retCode") == 0:
                return resp.get("result", {})
            return None
        except Exception as e:
            self.log(f"_get_order_book error for {symbol}: {e}")
            
