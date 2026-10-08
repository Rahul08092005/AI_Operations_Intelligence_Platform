from fastapi import APIRouter

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status")
def status() -> dict:
    return {"module": "data", "status": "not_implemented"}
