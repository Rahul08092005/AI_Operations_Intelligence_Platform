from fastapi import APIRouter

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.get("/status")
def status() -> dict:
    return {"module": "anomaly", "status": "not_implemented"}
