from fastapi import APIRouter

router = APIRouter(prefix="/explain", tags=["reasoning"])


@router.get("/status")
def status() -> dict:
    return {"module": "reasoning", "status": "not_implemented"}
