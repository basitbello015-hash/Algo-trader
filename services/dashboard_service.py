from app_state import bc
from datetime import datetime

def get_dashboard_data():
    """
    Retrieves current dashboard statistics by aggregating data from BotController.
    """
    
    # 1. Calculate Total Balance (Sum of all validated accounts)
    accounts = bc.load_accounts()
    total_balance = 0.0
    for acc in accounts:
        # Use 'balance' if available, otherwise 0
        bal = acc.get("balance")
        if bal and isinstance(bal, (int, float)):
            total_balance += float(bal)
            
    # 2. Count Active Positions (Open trades)
    trades = bc._read_trades()
    open_trades = [t for t in trades if t.get("open") is True]
    active_trades_count = len(open_trades)
    
    # 3. Calculate Today's Profit
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    today_pnl = 0.0
    
    for t in trades:
        if t.get("open") is False:
            exit_time = t.get("exit_time") # ISO format string
            if exit_time and exit_time.startswith(today_str):
                try:
                    entry = float(t.get("entry_price", 0))
                    exit_p = float(t.get("exit_price", 0))
                    qty = float(t.get("qty", 0))
                    profit = (exit_p - entry) * qty
                    today_pnl += profit
                except Exception:
                    pass

    # 4. Calculate Daily Change %
    daily_change_pct = 0.0
    if total_balance > 0:
        start_balance = total_balance - today_pnl
        if start_balance > 0:
            daily_change_pct = (today_pnl / start_balance) * 100.0

    # 5. Active Bots (Accounts with monitoring=True)
    active_bots_count = sum(1 for a in accounts if a.get("monitoring") is True)

    return {
        "profit": round(today_pnl, 2),
        "openTrades": active_trades_count,
        "balance": round(total_balance, 2),
        "dailyChange": round(daily_change_pct, 2),
        "activeBots": active_bots_count
    }
