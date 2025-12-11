from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.accounts_service import get_accounts, add_account, delete_account, test_account

# Create router
router = APIRouter()

# -----------------------
# Pydantic model for validation
# -----------------------
class AccountModel(BaseModel):
    name: str
    exchange: str
    apiKey: str
    secretKey: str

# -----------------------
# Routes
# -----------------------

@router.get("/")
def list_accounts():
    """Get all saved trading accounts."""
    try:
        return get_accounts()
    except Exception as e:
        print(f"❌ Error listing accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
def create_account(data: AccountModel):
    """Create a new trading account."""
    try:
        print(f"📩 Adding account: {data.name} ({data.exchange})")
        result = add_account(data.dict())
        return {"success": True, "message": "Account added successfully", "account": result.get("account")}
    except Exception as e:
        print(f"❌ Error creating account: {e}")
        # Return success=False so frontend can show the error message
        return {"success": False, "message": str(e)}

@router.delete("/{account_id}")
def remove_account(account_id: str):
    """Delete an account by ID."""
    try:
        result = delete_account(account_id)
        return {"success": True, "message": f"Account {account_id} deleted", "id": result["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{account_id}/test")
def test_connection(account_id: str):
    """Test if an account connection works."""
    try:
        result = test_account(account_id)
        return {"success": True, "message": "Connection successful", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
