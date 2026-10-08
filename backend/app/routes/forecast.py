from fastapi import APIRouter

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/status")
def status() -> dict:
    return {"module": "forecast", "status": "not_implemented"}
