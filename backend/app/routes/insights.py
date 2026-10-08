from fastapi import APIRouter

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/status")
def status() -> dict:
    return {"module": "insights", "status": "not_implemented"}
