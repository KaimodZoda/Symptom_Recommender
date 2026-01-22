from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import pandas as pd
import uvicorn
import os

#http://127.0.0.1:8000/docs

# Import engine logic from engine.py
from engine import SymptomStandardizer, SymptomRecommender

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (Startup & Shutdown)"""
    global recommender
    print("Starting system and loading data...")
    
    std = SymptomStandardizer(mapping_file="symptom_mapping.pkl")
    
    try:
        rules_path = "association_rules.pkl"
        
        if os.path.exists(rules_path):
            rules_df = pd.read_pickle(rules_path)
            recommender = SymptomRecommender(rules_df, std)
            print("Model and data loaded successfully")
        else:
            # Create Mock Data for initial testing
            mock_rules = pd.DataFrame([
                {"antecedents": frozenset(["Yes_ปวดหัว"]), "consequents": frozenset(["Yes_ปวดตา"]), "confidence": 0.8, "lift": 2.5},
                {"antecedents": frozenset(["Yes_มีไข้"]), "consequents": frozenset(["Yes_หนาวสั่น"]), "confidence": 0.7, "lift": 3.0}
            ])
            recommender = SymptomRecommender(mock_rules, std)
            print("Warning: Real file not found, system running with Mock Data")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        recommender = None
    
    yield
    print("Shutting down system...")

app = FastAPI(
    title="Medical Symptom Recommendation API",
    version="1.3.2",
    lifespan=lifespan
)

recommender = None

# --- Data Models ---

class RecommendationRequest(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "female"})
    age: int = Field(..., json_schema_extra={"example": 40})
    current_symptoms: List[str] = Field(..., json_schema_extra={"example": ["ไอ"]})
    top_n: Optional[int] = Field(6)

class SymptomMetric(BaseModel):
    symptom: str
    score: float
    confidence: float
    lift: float
    source: str

class RecommendationResponse(BaseModel):
    status: str
    count: int
    recommendations: List[SymptomMetric]

# --- Endpoints ---

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model failed to load")
    
    try:
        # Get data from Engine
        raw_results = recommender.recommend_next_symptoms(
            gender=req.gender,
            age=req.age,
            current_yes_symptoms=req.current_symptoms,
            top_n=req.top_n
        )
        
        # Fix Validation Error by flattening data structure
        # raw_results is now a list of tuples: [(symptom_name, metrics_dict), ...]
        formatted_results = []
        for symptom_name, metrics in raw_results:
            formatted_results.append({
                "symptom": symptom_name,
                "score": metrics.get("score", 0.0),
                "confidence": metrics.get("conf", 0.0),
                "lift": metrics.get("lift", 0.0),
                "source": metrics.get("method", "unknown")
            })
        
        return {
            "status": "success",
            "count": len(formatted_results),
            "recommendations": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)