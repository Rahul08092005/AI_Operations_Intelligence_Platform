from fastapi import APIRouter

from app.config import get_settings
from app.database.session import check_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Report API and database status. Always returns HTTP 200 so the
    frontend can display a 'degraded' state instead of a network error."""
    settings = get_settings()
    db_ok = check_db()
    return {
        "status": "healthy" if db_ok else "degraded",
        "api": "up",
        "database": "connected" if db_ok else "disconnected",
        "version": settings.app_version,
        "environment": settings.environment,
    }
