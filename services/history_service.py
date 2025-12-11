import time
import uuid
from typing import List, Tuple, Optional
from app_state import bc

def get_trades(limit: int = 50, offset: int = 0, symbol: Optional[str] = None, status: Optional[str] = None) -> Tuple[List[dict], int]:
    """
    Retrieve paginated trade history from the shared trades file.
    """
    all_trades = bc._read_trades()

    # Filtering
    if symbol:
        all_trades = [t for t in all_trades if t.get("symbol") == symbol]
    if status:
        if status.lower() == "open":
            all_trades = [t for t in all_trades if t.get("open") is True]
        elif status.lower() == "closed":
            all_trades = [t for t in all_trades if t.get("open") is False]

    # Sort by entry_time descending (newest first)
    all_trades.sort(key=lambda x: x.get("entry_time") or "", reverse=True)

    total = len(all_trades)
    paginated_trades = all_trades[offset : offset + limit]
    
    return paginated_trades, total

def get_trade_by_id(trade_id: str) -> Optional[dict]:
    """
    Retrieve a single trade by its ID.
    """
    all_trades = bc._read_trades()
    for trade in all_trades:
        if trade.get("id") == trade_id:
            return trade
    return None

def append_trade(trade_data: dict) -> dict:
    """
    Add a new trade record (mostly for testing/manual entry).
    """
    # Generate ID if missing
    if "id" not in trade_data:
        trade_data["id"] = str(uuid.uuid4())
    
    # Add timestamp if missing
    if "entry_time" not in trade_data:
        trade_data["entry_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
    # Ensure 'open' status
    if "open" not in trade_data:
        trade_data["open"] = True

    bc.add_trade(trade_data)
    return trade_data
