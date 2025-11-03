# api.py (in ADS-B-Disaster-Management)
import os
from typing import Any, Dict, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from main import setup_environment, initialize_vectorstore
from src.retriever import FlightRetriever
from src.rag_pipeline import ADSBRAGPipeline

app = FastAPI(title="ADS-B RAG API")

# Allow local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    query: str

# ---- Init once at startup
groq_api_key = setup_environment()
embedding_store = initialize_vectorstore()
retriever = FlightRetriever(embedding_store)
rag = ADSBRAGPipeline(
    groq_api_key=groq_api_key,
    retriever=retriever,
    model="openai/gpt-oss-20b",  # ensure this model string is valid for your provider
)

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "ADS-B RAG API is running",
        "docs": "/docs",
        "post_ask_example": 'curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d \'{"query":"hello"}\''
    }

def _normalize_result(result: Any) -> Dict[str, Any]:
    """Coerce various return types to a stable dict your UI can consume."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "dict"):
        return result.dict()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "text"):
        return {"answer": result.text}
    return {"answer": str(result)}

@app.post("/ask")
def ask(req: AskReq) -> Dict[str, Any]:
    try:
        if hasattr(rag, "run_query"):
            result = rag.run_query(req.query)
        elif callable(rag):                 # if ADSBRAGPipeline implements __call__(...)
            result = rag(req.query)
        elif hasattr(rag, "predict"):       # legacy naming
            result = rag.predict(req.query)
        else:
            raise AttributeError(
                "ADSBRAGPipeline exposes none of: run_query, __call__, predict. "
                "Add one of these methods to your pipeline."
            )

        return _normalize_result(result)

    except Exception as e:
        # Return clean 500 with detail
        raise HTTPException(status_code=500, detail=f"/ask failed: {e}")
    
@app.get("/flights")
def flights() -> List[Dict]:
    """
    Return current flight points for the map.
    Replace this stub with your real ADS-B feed when ready.
    lat/lon in degrees, altitude in feet, speed in knots.
    """
    return [
        {"callsign": "AI203", "lat": 28.556, "lon": 77.100, "altitude": 36000, "speed": 460},
        {"callsign": "UK879", "lat": 19.089, "lon": 72.865, "altitude": 12000, "speed": 320},
        {"callsign": "QA011", "lat": 12.968, "lon": 77.596, "altitude": 24000, "speed": 410},
    ]

