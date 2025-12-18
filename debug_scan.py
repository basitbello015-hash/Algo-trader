#!/usr/bin/env python3
"""
Debug script to see why the bot isn't placing trades.
This runs one complete scan cycle and shows detailed diagnostics.
"""
import sys
from app_state import bc
from bot_fib_scoring import ALLOWED_COINS

print("=" * 80)
print("TRADING BOT DIAGNOSTIC SCAN")
print("=" * 80)
print()

# Load accounts
accounts = bc.load_accounts()
if not accounts:
    print("❌ ERROR: No accounts found in app/data/accounts.json")
    sys.exit(1)

print(f"✓ Found {len(accounts)} account(s)")
print()

for idx, acct in enumerate(accounts, 1):
    print(f"--- Account {idx}: {acct.get('name')} ---")
    print(f"ID: {acct.get('id')}")
    print(f"Exchange: {acct.get('exchange')}")
    print(f"Monitoring: {acct.get('monitoring')}")
    print(f"Position: {acct.get('position')}")
    print()
    
    # Check if monitoring is enabled
    if not acct.get('monitoring'):
        print("⚠️  ISSUE: Account monitoring is disabled!")
        print("   FIX: Set 'monitoring': true in accounts.json")
        print()
        continue
    
    # Check if position is already open
    if acct.get('position') == 'open':
        print("⚠️  ISSUE: Account already has an open position")
        print(f"   Open trade ID: {acct.get('open_trade_id')}")
        print("   The bot won't open new trades until this one closes")
        print()
        continue
    
    # Check daily limit
    if bc._check_daily_limit():
        print(f"⚠️  ISSUE: Daily trade limit reached ({bc.trades_today}/{bc.MAX_TRADES_DAILY})")
        print()
        continue
    
    # Get client
    print("Connecting to exchange...")
    client = bc._get_client(acct)
    if not client:
        print("❌ ERROR: Failed to create exchange client")
        print(f"   Error: {acct.get('last_validation_error', 'Unknown')}")
        print()
        continue
    print("✓ Connected to Bybit")
    print()
    
    # Validate account
    print("Validating account...")
    ok, bal, err = bc.validate_account(acct)
    if not ok:
        print(f"❌ ERROR: Account validation failed")
        print(f"   Error: {err}")
        print()
        continue
    print(f"✓ Account validated, balance: ${bal:.2f}")
    print()
    
    # Check allocation
    from bot_fib_scoring import TRADE_SETTINGS
    alloc_pct = float(TRADE_SETTINGS.get('trade_allocation_pct', 100))
    if alloc_pct > 1.0:
        alloc_pct = alloc_pct / 100.0
    usd_alloc = float(bal) * alloc_pct
    min_trade = TRADE_SETTINGS.get('min_trade_amount', 5.0)
    
    print(f"Trade allocation: {alloc_pct*100:.0f}% = ${usd_alloc:.2f}")
    print(f"Minimum trade amount: ${min_trade:.2f}")
    
    if usd_alloc < min_trade:
        print(f"❌ ERROR: Allocation ${usd_alloc:.2f} below minimum ${min_trade:.2f}")
        print()
        continue
    print("✓ Allocation sufficient")
    print()
    
    # Score symbols
    print(f"Scoring {len(ALLOWED_COINS)} symbols...")
    print("(This may take 1-2 minutes due to rate limiting)")
    print()
    
    candidates = []
    scored_count = 0
    
    for symbol in ALLOWED_COINS[:10]:  # Test first 10 only for speed
        try:
            sc, diag = bc.score_symbol(client, symbol)
            scored_count += 1
            
            if sc > 0:
                print(f"  {symbol}: score={sc}")
                if diag.get('error'):
                    print(f"    Error: {diag['error']}")
                else:
                    print(f"    RSI={diag.get('rsi', 'N/A'):.1f}, "
                          f"Momentum={diag.get('momentum_pct', 0):.2f}%, "
                          f"Trend={diag.get('trend', 'N/A')}, "
                          f"InFibZone={diag.get('in_fib_zone', False)}")
            
            if sc >= 3:
                candidates.append((symbol, sc, diag))
                
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
            continue
    
    print()
    print(f"Scored {scored_count} symbols")
    print(f"Found {len(candidates)} candidates with score ≥ 3")
    print()
    
    if not candidates:
        print("❌ NO TRADING OPPORTUNITIES FOUND")
        print()
        print("REASONS:")
        print("  1. Market conditions don't meet entry criteria")
        print("  2. Score threshold (≥3) is too strict")
        print("  3. RSI oversold threshold (≤35) rarely triggered")
        print()
        print("SOLUTIONS:")
        print("  A. Lower score threshold from 3 to 2 (line 791 in bot_fib_scoring.py)")
        print("  B. Increase RSI threshold from 35 to 45 in config.json")
        print("  C. Wait for better market conditions (oversold coins)")
        print()
        continue
    
    # Show best candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_symbol, best_score, best_diag = candidates[0]
    
    print(f"✓ BEST CANDIDATE: {best_symbol} (score={best_score})")
    print(f"  Current price: ${best_diag.get('current_price', 0):.6f}")
    print(f"  RSI: {best_diag.get('rsi', 0):.1f}")
    print(f"  Momentum: {best_diag.get('momentum_pct', 0):.2f}%")
    print(f"  Trend: {best_diag.get('trend', 'N/A')}")
    print(f"  In Fib Zone: {best_diag.get('in_fib_zone', False)}")
    print()
    
    print("✅ BOT SHOULD PLACE A TRADE FOR THIS SYMBOL")
    print()

print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
