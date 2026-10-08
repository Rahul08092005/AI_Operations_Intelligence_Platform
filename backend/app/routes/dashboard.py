from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/status")
def status() -> dict:
    return {"module": "dashboard", "status": "not_implemented"}
