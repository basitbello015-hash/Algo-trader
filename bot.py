"""
Enhanced Strategy Controller - High Probability Setup
Optimized for 70% qualification rate with stricter, more reliable conditions
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
import random

# Exchange client
from pybit.unified_trading import HTTP

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# ==================== UPDATED SETTINGS ====================
# 5-minute timeframe
TIMEFRAME = CONFIG.get("timeframe", "5")

# AI-selected top 10 coins under $1 (update these daily based on your AI picks)
AI_COINS_UNDER_1_DOLLAR = CONFIG.get("ai_coins", [
    "DOGEUSDT",   # $0.15
    "SHIBUSDT",   # $0.00002
    "TRXUSDT",    # $0.11
    "ADAUSDT",    # $0.40
    "VETUSDT",    # $0.03
    "ALGOUSDT",   # $0.20
    "XLMUSDT",    # $0.12
    "ONEUSDT",    # $0.02
    "ANKRUSDT",   # $0.04
    "COTIUSDT",   # $0.10
])
ALLOWED_COINS = AI_COINS_UNDER_1_DOLLAR[:10]  # Ensure exactly 10 coins

# Risk rules
RISK_RULES = {
    "stop_loss_pct": CONFIG.get("stopLossPct", 1.0),  # 1% stop loss from entry
    "take_profit_pct": CONFIG.get("takeProfitPct", 2.5),  # 2.5% take profit (split into 2.0% and 3.0%)
    "max_hold": int(CONFIG.get("maxHoldSeconds", 90 * 60)),  # 90 minutes
    "max_slippage_pct": CONFIG.get("maxSlippagePct", 0.3),  # 0.3% max slippage
    "min_order_book_depth_usd": CONFIG.get("minOrderBookDepthUsd", 5000.0),  # $5000 min depth
}

# Score and strategy settings - OPTIMIZED FOR HIGH PROBABILITY
SCORE_SETTINGS = {
    "rsi_period": int(CONFIG.get("rsiPeriod", 10)),  # Faster RSI for 5m
    "rsi_oversold_threshold": float(CONFIG.get("rsiOversold", 40.0)),  # Higher threshold = more signals
    "rsi_overbought_threshold": float(CONFIG.get("rsiOverbought", 70.0)),
    
    # Momentum settings - relaxed for more signals
    "momentum_lookback": int(CONFIG.get("momentumLookback", 3)),  # 3 candles (15 minutes)
    "momentum_entry_threshold_pct": float(CONFIG.get("momentumEntryThreshold", 0.3)),  # 0.3% minimum
    
    # Volume confirmation - CRITICAL for reliability
    "volume_ratio_threshold": float(CONFIG.get("volumeRatioThreshold", 1.5)),  # 1.5x average volume
    "min_volume_usd": float(CONFIG.get("minVolumeUsd", 50000)),  # $50k minimum volume
    
    # Trend filter - more permissive
    "ema_fast": int(CONFIG.get("emaFast", 9)),  # 9-period EMA
    "ema_slow": int(CONFIG.get("emaSlow", 21)),  # 21-period EMA
    "require_trend_alignment": CONFIG.get("requireTrendAlignment", False),  # Can trade counter-trend
    
    # Scoring weights - prioritize volume and momentum
    "score_weights": {
        "rsi": int(CONFIG.get("scoreWeightRsi", 2)),
        "momentum": int(CONFIG.get("scoreWeightMomentum", 3)),
        "volume": int(CONFIG.get("scoreWeightVolume", 4)),  # Most important
        "ema": int(CONFIG.get("scoreWeightEma", 1)),
        "candle": int(CONFIG.get("scoreWeightCandle", 2)),
    }
}

TRADE_SETTINGS = {
    "trade_allocation_pct": CONFIG.get("tradeAllocation", 25),  # 25% per trade for risk management
    "min_trade_amount": float(CONFIG.get("minTradeAmount", 10.0)),
    "use_market_order": CONFIG.get("useMarketOrder", True),
    "test_on_testnet": CONFIG.get("testOnTestnet", False),
    "scan_interval": int(CONFIG.get("scanInterval", 60)),  # Scan every 60 seconds (5m candles)
    "dry_run": CONFIG.get("dryRun", True),
    "max_trades_per_day": int(CONFIG.get("maxTradesPerDay", 10)),  # 10 trades/day max
    "min_profit_target_pct": float(CONFIG.get("minProfitTarget", 2.0)),  # First TP at 2%
    "max_profit_target_pct": float(CONFIG.get("maxProfitTarget", 3.0)),  # Second TP at 3%
    "partial_profit_pct": float(CONFIG.get("partialProfitPct", 0.5)),  # Take 50% at first TP
}

RATE_LIMIT_SETTINGS = {
    "max_requests_per_second": CONFIG.get("maxRequestsPerSecond", 5),
    "max_requests_per_minute": CONFIG.get("maxRequestsPerMinute", 50),
}

# File paths
ACCOUNTS_FILE = os.path.join(BASE_DIR, "app/data/accounts.json")
TRADES_FILE = os.path.join(BASE_DIR, "app/data/trades.json")
COINS_FILE = os.path.join(BASE_DIR, "app/data/coins_daily.json")  # Store daily AI picks

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "app/data"), exist_ok=True)

# -------------------- ENHANCED INDICATORS --------------------
def calc_smoothed_rsi(closes: List[float], period: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Calculate RSI and its slope for momentum confirmation."""
    if len(closes) < period + 3:
        return None, None
    
    # Calculate RSI values
    rsi_values = []
    for i in range(period, len(closes)):
        gains = []
        losses = []
        for j in range(i - period + 1, i + 1):
            if j == 0:
                continue
            change = closes[j] - closes[j-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if sum(losses) == 0:
            rsi = 100.0
        else:
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_values.append(rsi)
    
    if not rsi_values:
        return None, None
    
    current_rsi = rsi_values[-1]
    
    # Calculate RSI slope (3-period momentum)
    if len(rsi_values) >= 4:
        slope = (rsi_values[-1] - rsi_values[-4]) / 3
    else:
        slope = 0
    
    return current_rsi, slope

def calc_volume_confirmation(volumes: List[float], prices: List[float], 
                           lookback: int = 20) -> Tuple[float, bool]:
    """Calculate volume ratio and confirm if volume is significant."""
    if len(volumes) < lookback + 1:
        return 0.0, False
    
    current_volume = volumes[-1]
    avg_volume = sum(volumes[-lookback:-1]) / (lookback - 1)
    
    # Calculate volume in USD
    current_price = prices[-1]
    current_volume_usd = current_volume * current_price
    
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    
    # Check if volume meets minimum USD threshold
    volume_adequate = current_volume_usd >= SCORE_SETTINGS["min_volume_usd"]
    
    return volume_ratio, volume_adequate

def detect_reversal_candle(candles: List[Dict[str, float]]) -> int:
    """
    Detect bullish reversal patterns.
    Returns score from 0-3 based on pattern strength.
    """
    if len(candles) < 3:
        return 0
    
    current = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3]
    
    score = 0
    
    # 1. Hammer pattern
    body = abs(current['close'] - current['open'])
    lower_wick = min(current['open'], current['close']) - current['low']
    upper_wick = current['high'] - max(current['open'], current['close'])
    
    if lower_wick >= 2 * body and upper_wick <= 0.5 * body:
        score += 2
    
    # 2. Bullish engulfing
    if (prev['close'] < prev['open'] and  # Previous bearish
        current['close'] > current['open'] and  # Current bullish
        current['close'] > prev['open'] and  # Closes above previous open
        current['open'] < prev['close']):    # Opens below previous close
        score += 2
    
    # 3. Morning star (simplified)
    if (prev2['close'] < prev2['open'] and  # First bearish
        abs(prev['close'] - prev['open']) < 0.1 * prev2['body'] and  # Small body
        current['close'] > current['open'] and  # Bullish
        current['close'] > (prev2['close'] + prev2['open']) / 2):  # Above midpoint
        score += 3
    
    # 4. Simple higher low pattern
    if (prev['low'] > prev2['low'] and 
        current['low'] > prev['low'] and
        current['close'] > prev['close']):
        score += 1
    
    return min(score, 3)  # Cap at 3

def calc_entry_score(symbol: str, candles: List[Dict[str, float]], 
                    volumes: List[float]) -> Dict[str, Any]:
    """Calculate entry score with relaxed but smarter conditions."""
    if len(candles) < 50:
        return {"score": 0, "details": {}}
    
    # Extract data
    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    current_price = closes[-1]
    
    # 1. RSI Analysis (40% weight)
    rsi, rsi_slope = calc_smoothed_rsi(closes, SCORE_SETTINGS["rsi_period"])
    rsi_score = 0
    if rsi:
        if rsi < SCORE_SETTINGS["rsi_oversold_threshold"]:
            rsi_score = 3  # Oversold = good entry
        elif rsi < 50 and rsi_slope > 0:
            rsi_score = 2  # Rising from oversold
        elif rsi < 60:
            rsi_score = 1  # Neutral but okay
    
    # 2. Volume Confirmation (30% weight - MOST IMPORTANT)
    volume_ratio, volume_adequate = calc_volume_confirmation(volumes, closes)
    volume_score = 0
    if volume_adequate:
        if volume_ratio >= SCORE_SETTINGS["volume_ratio_threshold"]:
            volume_score = 4  # High volume breakout
        elif volume_ratio >= 1.2:
            volume_score = 2  # Above average volume
    
    # 3. Momentum (20% weight)
    momentum = smoothed_momentum_pct(closes, lookback=SCORE_SETTINGS["momentum_lookback"])
    momentum_score = 0
    if momentum > SCORE_SETTINGS["momentum_entry_threshold_pct"]:
        momentum_score = 2 if momentum > 0.5 else 1
    
    # 4. EMA Trend (10% weight - optional)
    ema_score = 0
    if len(closes) >= SCORE_SETTINGS["ema_slow"]:
        ema_fast = calc_ema(closes, SCORE_SETTINGS["ema_fast"])
        ema_slow = calc_ema(closes, SCORE_SETTINGS["ema_slow"])
        if ema_fast and ema_slow:
            if current_price > ema_fast > ema_slow:
                ema_score = 2  # Strong uptrend
            elif current_price > ema_slow:
                ema_score = 1  # Above slow EMA
    
    # 5. Candle Pattern (10% weight)
    candle_score = detect_reversal_candle(candles)
    
    # Calculate weighted total
    weights = SCORE_SETTINGS["score_weights"]
    total_score = (
        rsi_score * weights["rsi"] +
        volume_score * weights["volume"] +
        momentum_score * weights["momentum"] +
        ema_score * weights["ema"] +
        candle_score * weights["candle"]
    )
    
    max_possible = (
        3 * weights["rsi"] +
        4 * weights["volume"] +
        2 * weights["momentum"] +
        2 * weights["ema"] +
        3 * weights["candle"]
    )
    
    # Normalize to 0-10 scale
    normalized_score = (total_score / max_possible) * 10 if max_possible > 0 else 0
    
    return {
        "score": normalized_score,
        "details": {
            "rsi": rsi,
            "rsi_slope": rsi_slope,
            "volume_ratio": volume_ratio,
            "volume_adequate": volume_adequate,
            "momentum": momentum,
            "candle_pattern": candle_score,
            "price": current_price,
            "symbol": symbol
        }
    }

def should_enter_trade(score_data: Dict[str, Any]) -> bool:
    """Entry decision with 70% qualification rate target."""
    details = score_data.get("details", {})
    
    # MUST-HAVE conditions
    if not details.get("volume_adequate", False):
        return False
    
    # Score threshold - lowered for more signals
    if score_data.get("score", 0) < 4.0:  # Reduced from typical 6.0
        return False
    
    # At least 2 positive signals
    positive_signals = 0
    if details.get("rsi", 50) < 45:
        positive_signals += 1
    if details.get("volume_ratio", 0) >= 1.2:
        positive_signals += 1
    if details.get("momentum", 0) > 0:
        positive_signals += 1
    if details.get("candle_pattern", 0) >= 1:
        positive_signals += 1
    
    return positive_signals >= 2  # Need at least 2 positive signals

# -------------------- UPDATED BOT CONTROLLER --------------------
class EnhancedBotController:
    def __init__(self, log_queue: Optional[threading.Queue] = None):
        self.log_queue = log_queue
        self._running = False 
        self._stop = threading.Event()
        self._file_lock = threading.Lock()
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_per_second=RATE_LIMIT_SETTINGS["max_requests_per_second"],
            max_per_minute=RATE_LIMIT_SETTINGS["max_requests_per_minute"]
        )
        
        # Daily limit tracking
        self.trades_today = 0
        self.day_start_time = time.time()
        self.MAX_TRADES_DAILY = TRADE_SETTINGS["max_trades_per_day"]
        
        # Performance tracking
        self.qualification_rate = 0.0
        self.total_scans = 0
        self.qualified_scans = 0
        
        # Load daily AI coins if file exists
        self._load_daily_coins()
        
        # Ensure files exist
        for f in (ACCOUNTS_FILE, TRADES_FILE):
            if not os.path.exists(f):
                with open(f, "w") as fh:
                    fh.write("[]")
    
    def _load_daily_coins(self):
        """Load daily AI-selected coins from file."""
        try:
            if os.path.exists(COINS_FILE):
                with open(COINS_FILE, "r") as f:
                    data = json.load(f)
                    if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        global ALLOWED_COINS
                        ALLOWED_COINS = data.get("coins", ALLOWED_COINS)[:10]
                        self.log(f"Loaded daily AI coins: {ALLOWED_COINS}")
        except Exception as e:
            self.log(f"Error loading daily coins: {e}")
    
    def update_daily_coins(self, new_coins: List[str]):
        """Update the daily AI coin selection."""
        try:
            data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "coins": new_coins[:10],
                "updated": datetime.now().isoformat()
            }
            with open(COINS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            global ALLOWED_COINS
            ALLOWED_COINS = new_coins[:10]
            self.log(f"Updated daily AI coins: {ALLOWED_COINS}")
        except Exception as e:
            self.log(f"Error updating daily coins: {e}")
    
    def log(self, msg: str):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.log_queue:
            try:
                self.log_queue.put(line, block=False)
            except Exception:
                pass
    
    def _check_daily_limit(self) -> bool:
        """Check if daily trade limit is reached."""
        current_time = time.time()
        
        # Reset after 24 hours
        if current_time - self.day_start_time > 86400:
            self.log(f"24h reset. Previous trades: {self.trades_today}")
            self.trades_today = 0
            self.day_start_time = current_time
        
        # Check limit
        limit_reached = self.trades_today >= self.MAX_TRADES_DAILY
        if limit_reached:
            hours_left = (86400 - (current_time - self.day_start_time)) / 3600
            self.log(f"Daily limit reached: {self.trades_today}/{self.MAX_TRADES_DAILY}. "
                    f"Resets in {hours_left:.1f} hours")
        
        return limit_reached
    
    def _update_qualification_rate(self, qualified: bool):
        """Track qualification rate for optimization."""
        self.total_scans += 1
        if qualified:
            self.qualified_scans += 1
        
        # Update rate every 10 scans
        if self.total_scans % 10 == 0:
            self.qualification_rate = (self.qualified_scans / self.total_scans) * 100
            self.log(f"Qualification rate: {self.qualification_rate:.1f}% "
                    f"({self.qualified_scans}/{self.total_scans})")
            
            # Auto-adjust thresholds if rate is too low/high
            if self.qualification_rate < 60 and self.total_scans > 50:
                # Too few signals, lower thresholds
                self._adjust_thresholds(higher=True)
            elif self.qualification_rate > 80 and self.total_scans > 50:
                # Too many signals, raise thresholds
                self._adjust_thresholds(higher=False)
    
    def _adjust_thresholds(self, higher: bool = True):
        """Auto-adjust thresholds to maintain ~70% qualification rate."""
        factor = 1.05 if higher else 0.95
        
        # Adjust RSI threshold
        current = SCORE_SETTINGS["rsi_oversold_threshold"]
        SCORE_SETTINGS["rsi_oversold_threshold"] = current * factor
        self.log(f"Auto-adjusted RSI threshold to {SCORE_SETTINGS['rsi_oversold_threshold']:.1f}")
    
    def scan_for_opportunities(self, client: HTTP) -> Optional[Dict[str, Any]]:
        """Scan for trading opportunities with high probability criteria."""
        opportunities = []
        
        for symbol in ALLOWED_COINS:
            try:
                # Get candle data
                self.rate_limiter.acquire()
                resp = client.get_kline(
                    category="spot",
                    symbol=symbol,
                    interval=TIMEFRAME,
                    limit=100
                )
                
                if resp.get("retCode") != 0:
                    continue
                
                result = resp.get("result", {})
                candles_data = result.get("list", [])
                
                if len(candles_data) < 50:
                    continue
                
                # Parse candles
                candles = []
                volumes = []
                for c in candles_data:
                    candle = {
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    }
                    candles.append(candle)
                    volumes.append(candle['volume'])
                
                # Calculate score
                score_data = calc_entry_score(symbol, candles, volumes)
                
                # Check if should enter
                if should_enter_trade(score_data):
                    details = score_data["details"]
                    opportunities.append({
                        "symbol": symbol,
                        "score": score_data["score"],
                        "price": details["price"],
                        "rsi": details["rsi"],
                        "volume_ratio": details["volume_ratio"],
                        "momentum": details["momentum"],
                        "timestamp": time.time()
                    })
                    
                    self.log(f"✅ Qualified: {symbol} | Score: {score_data['score']:.1f} | "
                            f"RSI: {details['rsi']:.1f} | Vol: {details['volume_ratio']:.2f}x")
                
                # Update qualification rate
                self._update_qualification_rate(len(opportunities) > 0)
                
            except Exception as e:
                self.log(f"Error scanning {symbol}: {e}")
                continue
        
        # Return best opportunity
        if opportunities:
            # Sort by score and pick top 2
            opportunities.sort(key=lambda x: x["score"], reverse=True)
            best = opportunities[0]
            
            # Additional validation for best opportunity
            if best["score"] >= 5.0 and best["volume_ratio"] >= 1.2:
                self.log(f"🎯 Best opportunit
