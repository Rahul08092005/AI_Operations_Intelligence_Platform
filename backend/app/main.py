"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database.session import check_db
from app.routes import (
    anomaly,
    dashboard,
    data,
    forecast,
    health,
    insights,
    prediction,
    reasoning,
)

logger = logging.getLogger("aiops")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.environment)
    logger.info("Database reachable: %s", check_db())
    # Day 7: load trained models once here (anomaly_model.pkl, failure_model.pkl)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered predictive operations and business intelligence API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
for module in (data, forecast, anomaly, prediction, reasoning, insights, dashboard):
    app.include_router(module.router)
