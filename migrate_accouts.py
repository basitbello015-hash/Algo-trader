"""
Migration script to fix api_Key -> api_key field name in existing accounts.
Run this once on your Render deployment to fix existing accounts.
"""
import json
import os

ACCOUNTS_FILE = "app/data/accounts.json"

def migrate_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        print("No accounts file found. Nothing to migrate.")
        return
    
    try:
        with open(ACCOUNTS_FILE, "r") as f:
            accounts = json.load(f)
        
        print(f"Found {len(accounts)} accounts to check")
        
        migrated = False
        for account in accounts:
            # Fix api_Key -> api_key
            if "api_Key" in account:
                account["api_key"] = account.pop("api_Key")
                migrated = True
                print(f"✅ Migrated account: {account.get('name')}")
            
            # Ensure all required fields exist
            account.setdefault("monitoring", True)
            account.setdefault("position", "closed")
            account.setdefault("validated", False)
            account.setdefault("balance", None)
            account.setdefault("last_validation_error", None)
        
        if migrated:
            # Backup original file
            backup_file = ACCOUNTS_FILE + ".backup"
            with open(backup_file, "w") as f:
                json.dump(accounts, f, indent=2)
            print(f"📦 Backup saved to: {backup_file}")
            
            # Save migrated accounts
            with open(ACCOUNTS_FILE, "w") as f:
                json.dump(accounts, f, indent=2)
            print(f"✅ Migration complete! Fixed {len(accounts)} accounts")
        else:
            print("✅ No migration needed. All accounts are up to date.")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_accounts()
