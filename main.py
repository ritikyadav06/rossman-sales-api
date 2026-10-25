
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Literal

import joblib
import numpy as mp
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field

from features import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rossman-api")

ART = "artifacts"
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

  state["model"] = joblib.load(f"{ART}/model.joblib")
  state["stores_raw"] = pd.read_csv(f"{ART}/stores.csv")
  state["store_day_avg"] = pd.read_csv(f"{ART}/store_day_avg")
  state["columns"] = list(state["model"].feature_names_in_)
  with open(f"{ART}/model_card.json") as f:
    state["card"] = json.load(f)
  log.info("Logging model expecting %d columns", len(state["columns"]))
  yield
  state.clear()

app = FastAPI(
    title="Rossman sales predictor",
    description="Daily sales forecast (EUR) per rossman store",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
  start = time.perf_counter()
  response = await call_next(request)
  log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code,
           (time.perf_counter() - start) * 1000)
  return response

class PredictRequest(BaseModel):
  Store: Annotated[int, Field(..., ge=1, le=1115)]
  Date: date
  Open: Annotated[int, Field(..., ge=0, le=1, description="0 = closed, 1 = open")]
  Promo: Annotated[int, Field(..., ge=0, le=1)]
  StateHoliday: Literal['0', 'a', 'b', 'c'] = '0'
  SchoolHoliday: Annotated[int, Field(0, ge=0, le=1)]

  model_config = {"json_scheme_extra": {"examples": [
      {"Store": 1, "Date": "2015-05-04", "Open": 1,
       "Promo": 1, "StateHoliday": '0', "SchoolHoliday": 0}]}}


class Prediction(BaseModel):
  Store: int
  Date: date
  Open: int
  predicted_sales_eur: float
  version: str

class BatchRequest(BaseModel):
  items: Annotated[list[PredictRequest], Field(..., min=1, max=1000)]

class BatchResponse(BaseModel):
  predictions: list[Prediction]


async def run_predictions(items: list[PredictRequest]) -> list[Prediction]:
  known = set(state['stores_raw']['Store'])
  unknown = sorted({i.Store for i in items} - known)
  if unknown:
    raise HTTPException(status_code=404, detail=f"Unknown store id(s): {unknown}")

  df = pd.DataFrame([i.model_dump() for i in items()])
  preds = np.zeros(len(df))

  open_mask = (df['Open'] == 1).to_numpy()
  if open_mask.any():
    X = build_features(df[open_mask], state['stores_raw'], state['store_day_avg'], state['columns'])

    preds[open_mask] = np.clip(state['model'].predict(X), 0, None)

  version = str(state['card'].get('model_version', 'unknown'))

  return Prediction(Store=i.Store, Date=i.Date, Open=i.Open, predicted_sales_eur=round(float(p), 2), version=version)

INDEX = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def root():
  return FileResponse(INDEX)

@app.get("/health")
def health():
  return {"status": "ok", "model_loaded": "model" in state}

@app.get("/model-info")
def model_info():
  return state['card']

@app.post("/predict", response_model=Prediction)
async def predict(req: PredictRequest):
  return run_predictions([req])[0]

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
  return BatchResponse(predictions=run_predictions(req.items))
