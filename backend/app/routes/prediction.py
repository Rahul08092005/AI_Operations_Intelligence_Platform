from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.get("/status")
def status() -> dict:
    return {"module": "prediction", "status": "not_implemented"}
