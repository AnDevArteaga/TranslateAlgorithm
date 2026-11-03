from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.engine import Engine, TranslateError

app = FastAPI(
    title="Traductor de lenguas indígenas",
    version="1.0.0",
    description=(
        "Traduce entre español y wayuunaiki. Si la forma está en el diccionario "
        "devuelve sus sentidos. Si no está, propone una forma con el modelo y la marca "
        "como aproximación. Emberá queda anunciado hasta que exista un corpus propio."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine()


class TranslateRequest(BaseModel):
    text: str = Field(max_length=400)
    source: str
    target: str


@app.exception_handler(TranslateError)
def translate_error(_, exc: TranslateError):
    return JSONResponse(status_code=exc.status, content={"error": exc.message})


@app.get("/health")
def health():
    return {"status": "ok", "pairs": engine.dictionary["pairs"]}


@app.get("/languages")
def languages():
    return {
        "languages": [
            {"code": "es", "name": "Español", "ready": True},
            {"code": "guc", "name": "Wayuunaiki", "ready": True},
            {"code": "embera", "name": "Emberá", "ready": False},
        ]
    }


@app.post("/translate")
def translate(body: TranslateRequest):
    return engine.translate(body.text, body.source, body.target)
