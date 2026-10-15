import os
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import AdminLoginRequest, TokenResponse

router = APIRouter()

# Default admin credentials for FYP demonstration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "psl2026")


@router.post("/login", response_model=TokenResponse)
def admin_login(creds: AdminLoginRequest):
    if creds.username == ADMIN_USERNAME and creds.password == ADMIN_PASSWORD:
        return {
            "access_token": "psl-fyp-demo-admin-token-jwt-secure",
            "token_type": "bearer",
            "username": creds.username
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials. For demonstration use: admin / psl2026"
    )
