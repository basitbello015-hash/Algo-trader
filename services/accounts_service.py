import os
import json
import uuid
from typing import List, Dict
from app_state import bc

# Path to the accounts file your service THINKS is used
ACCOUNTS_FILE_PATH = "app/data/accounts.json"

# Ensure accounts directory exists on startup
os.makedirs(os.path.dirname(ACCOUNTS_FILE_PATH), exist_ok=True)

def debug_print(*args):
    print("\n🔍 DEBUG:", *args)

def read_raw_file():
    """Read raw file content for debugging."""
    try:
        with open(ACCOUNTS_FILE_PATH, "r") as f:
            return f.read()
    except:
        return "<unable to read file>"

def get_accounts() -> List[Dict]:
    """Retrieve all saved accounts using BotController."""
    acc = bc.load_accounts()
    debug_print("GET ACCOUNTS → bc.load_accounts() returned:", acc)
    return acc


def add_account(data: dict) -> Dict:
    debug_print("ADDING ACCOUNT → incoming data:", data)

    # Normalize keys - FIXED: api_Key -> apiKey
    normalized = {
        "id": data.get("id") or str(uuid.uuid4()),
        "name": data.get("name"),
        "exchange": data.get("exchange"),
        "api_key": data.get("apiKey") or data.get("api_key") or data.get("key"),
        "api_secret": data.get("secretKey") or data.get("api_secret") or data.get("secret") or data.get("secret_key"),
        "monitoring": data.get("monitoring", True),
        "position": data.get("position", "closed"),
    }

    data = normalized

    # Generate ID if missing
    if "id" not in data:
        data["id"] = str(uuid.uuid4())

    data.setdefault("monitoring", True)
    data.setdefault("position", "closed")

    # Ensure file exists
    if not os.path.exists(ACCOUNTS_FILE_PATH):
        debug_print("accounts.json does NOT exist — creating it now.")
        with open(ACCOUNTS_FILE_PATH, "w") as f:
            f.write("[]")

    debug_print("SERVICE ACCOUNTS FILE PATH:", os.path.abspath(ACCOUNTS_FILE_PATH))
    debug_print("RAW FILE CONTENT BEFORE SAVE:", read_raw_file())

    with bc._file_lock:
        before = bc.load_accounts()
        debug_print("BC BEFORE ADD → bc.load_accounts():", before)

        accounts = before.copy()
        accounts.append(data)

        debug_print("BC SAVE CALL → saving this list:", accounts)
        bc.save_accounts(accounts)

        after = bc.load_accounts()
        debug_print("BC AFTER SAVE → bc.load_accounts():", after)

        debug_print("RAW FILE CONTENT AFTER SAVE:", read_raw_file())

        # Verification step
        ids = [a.get("id") for a in after]
        debug_print("VERIFICATION IDS FOUND:", ids)

        if data["id"] not in ids:
            debug_print("❌ ERROR: ID not found after saving! Something is wrong.")
            raise Exception(
                "BotController is NOT saving to the same file that accounts_service expects."
            )

    # Auto-validate the account after adding
    try:
        ok, balance, err = bc.validate_account(data)
        if ok:
            debug_print("✅ ACCOUNT VALIDATED SUCCESSFULLY with balance:", balance)
            # Update the saved account with validation info
            with bc._file_lock:
                all_accounts = bc.load_accounts()
                for i, a in enumerate(all_accounts):
                    if a.get("id") == data["id"]:
                        all_accounts[i]["validated"] = True
                        all_accounts[i]["balance"] = balance
                        all_accounts[i]["last_validation_error"] = None
                        break
                bc.save_accounts(all_accounts)
        else:
            debug_print("⚠️ ACCOUNT VALIDATION FAILED:", err)
    except Exception as e:
        debug_print("⚠️ AUTO-VALIDATION ERROR:", str(e))

    debug_print("✅ ACCOUNT ADDED SUCCESSFULLY:", data)
    return {"status": "added", "account": data}


def delete_account(account_id: str) -> Dict:
    with bc._file_lock:
        accounts = bc.load_accounts()
        debug_print("DELETE ACCOUNT → Before:", accounts)

        filtered = [a for a in accounts if a.get("id") != account_id]

        if len(filtered) < len(accounts):
            bc.save_accounts(filtered)
            debug_print("DELETE ACCOUNT → After:", filtered)
            return {"status": "deleted", "id": account_id}

        debug_print("DELETE ACCOUNT → Not Found:", account_id)

    return {"status": "not_found", "id": account_id}


def test_account(account_id: str) -> Dict:
    accounts = bc.load_accounts()
    debug_print("TEST ACCOUNT → All accounts:", accounts)

    account = next((a for a in accounts if a.get("id") == account_id), None)

    if not account:
        debug_print("TEST ACCOUNT → Account not found:", account_id)
        return {"id": account_id, "connection": "failed", "reason": "Account not found"}

    ok, balance, err = bc.validate_account(account)

    if ok:
        with bc._file_lock:
            all_accounts = bc.load_accounts()
            for i, a in enumerate(all_accounts):
                if a.get("id") == account_id:
                    all_accounts[i]["validated"] = True
                    all_accounts[i]["balance"] = balance
                    all_accounts[i]["last_validation_error"] = None
                    break

            bc.save_accounts(all_accounts)

        return {"id": account_id, "connection": "success", "balance": balance}

    else:
        return {"id": account_id, "connection": "failed", "reason": err}
