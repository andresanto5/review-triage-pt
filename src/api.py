"""Local inference using the frozen baseline; never loads a user-supplied model."""
from contextlib import asynccontextmanager
import hashlib
import math
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from train import ROOT, normalize

MAX_BODY_BYTES = 65536


class Review(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    title: str = Field(default='', max_length=500, examples=['Produto com defeito'])
    text: str = Field(min_length=1, max_length=10000, examples=['O produto chegou quebrado e não funciona.'])

    @model_validator(mode='after')
    def usable_text(self):
        if len(normalize(self.title + ' ' + self.text)) < 10:
            raise ValueError('Combined normalized text must contain at least 10 characters')
        if not self.text.strip():
            raise ValueError('Review body cannot be blank')
        return self


class Prediction(BaseModel):
    review_required: bool
    negative_score: float
    threshold: float
    model_sha256: str
    target: str = 'rating_1_or_2_proxy'
    score_calibrated: bool = False


def create_app(model_path: Path = ROOT / 'artifacts/baseline.joblib'):
    @asynccontextmanager
    async def lifespan(app):
        # joblib is executable serialization: use only our trusted local artifact.
        bundle = joblib.load(model_path)
        threshold = float(bundle['threshold'])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid model threshold')
        if list(bundle['pipeline'].classes_) != [0, 1]:
            raise ValueError('Unexpected model classes')
        app.state.bundle = bundle
        app.state.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        yield

    app = FastAPI(title='Review Triage PT', version='0.1.0', lifespan=lifespan,
                  description='Demonstração local: encaminha avaliações para revisão humana usando um proxy de notas 1/2. O escore não é uma probabilidade calibrada. Modelo treinado com dados históricos de 2018.')

    @app.middleware('http')
    async def bound_input(request: Request, call_next):
        if request.method == 'POST' and request.url.path == '/predict':
            if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
                return JSONResponse(status_code=415, content={'detail': 'Use application/json'})
            # Count actual streamed bytes; do not trust Content-Length.
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY_BYTES:
                    return JSONResponse(status_code=413, content={'detail': 'Request too large'})
                chunks.append(chunk)
            request._body = b''.join(chunks)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        # Default validation errors can echo raw reviews; return a fixed message.
        return JSONResponse(status_code=422, content={'detail': 'Invalid review: use title (up to 500 characters) and text (1–10000), with at least 10 normalized characters in total. No extra fields.'})

    @app.get('/health')
    def health():
        return {'status': 'ready', 'model_sha256': app.state.model_sha256}

    @app.post('/predict', response_model=Prediction)
    def predict(review: Review):
        bundle = app.state.bundle
        score = float(bundle['pipeline'].predict_proba([normalize(review.title + ' ' + review.text)])[0, 1])
        return Prediction(review_required=score >= bundle['threshold'], negative_score=score,
                          threshold=bundle['threshold'], model_sha256=app.state.model_sha256)

    return app


app = create_app()
