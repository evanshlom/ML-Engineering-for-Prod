"""FastAPI application for serving NICU kangaroo care predictions."""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from datetime import datetime
import time
from typing import Dict, Optional
import logging
from contextlib import asynccontextmanager
import structlog

from .schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthCheckResponse, HealthStatus,
    ModelInfo, ErrorResponse, ExplainabilityRequest, ExplainabilityResponse
)
from ..models.architecture import NICUReadinessClassifier, ModelWithExplainability
from ..utils.config import load_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global variables for model and metadata
MODEL: Optional[ModelWithExplainability] = None
MODEL_INFO: Optional[Dict] = None
START_TIME = datetime.utcnow()
TOTAL_PREDICTIONS = 0
LAST_PREDICTION_TIME: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global MODEL, MODEL_INFO
    
    logger.info("Starting NICU Kangaroo Care API")
    
    # Load model
    try:
        config = load_config()
        base_model = NICUReadinessClassifier(**config['model'])
        MODEL = ModelWithExplainability(base_model)
        MODEL.eval()
        
        # Load model metadata
        MODEL_INFO = {
            "version": config.get('model_version', '1.0.0'),
            "training_date": config.get('training_date', datetime.utcnow()),
            "metrics": config.get('metrics', {})
        }
        
        logger.info("Model loaded successfully", model_version=MODEL_INFO['version'])
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down NICU Kangaroo Care API")


# Create FastAPI app
app = FastAPI(
    title="NICU Kangaroo Care Readiness API",
    description="ML API for predicting infant readiness for kangaroo care in NICU",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            error_type="ValidationError",
            detail="Invalid input values provided"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error("Unexpected error", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_type="InternalError",
            detail="An unexpected error occurred"
        ).model_dump()
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API and model health."""
    uptime = (datetime.utcnow() - START_TIME).total_seconds()
    
    status = HealthStatus.HEALTHY
    checks = {
        "model": MODEL is not None,
        "memory": True,  # Could add actual memory check
        "disk": True,    # Could add actual disk check
        "database": True # Placeholder for future DB integration
    }
    
    if not all(checks.values()):
        status = HealthStatus.UNHEALTHY if not checks["model"] else HealthStatus.DEGRADED
    
    return HealthCheckResponse(
        status=status,
        model_loaded=MODEL is not None,
        model_version=MODEL_INFO.get('version') if MODEL_INFO else None,
        uptime_seconds=uptime,
        last_prediction_time=LAST_PREDICTION_TIME,
        total_predictions=TOTAL_PREDICTIONS,
        checks=checks
    )


# Model info endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and metrics."""
    if not MODEL or not MODEL_INFO:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        model_version=MODEL_INFO['version'],
        training_date=MODEL_INFO['training_date'],
        feature_names=[
            'heart_rate',
            'oxygen_saturation',
            'respiratory_rate',
            'weight_grams',
            'temperature_celsius'
        ],
        metrics=MODEL_INFO['metrics'],
        hyperparameters=MODEL_INFO.get('hyperparameters', {})
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    global TOTAL_PREDICTIONS, LAST_PREDICTION_TIME
    
    if not MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        # Convert features to tensor
        features_dict = request.features.model_dump()
        features_array = np.array([
            features_dict['heart_rate'],
            features_dict['oxygen_saturation'],
            features_dict['respiratory_rate'],
            features_dict['weight_grams'],
            features_dict['temperature_celsius']
        ]).reshape(1, -1)
        
        features_tensor = torch.FloatTensor(features_array)
        
        # Make prediction
        with torch.no_grad():
            logits = MODEL(features_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = int(probability >= 0.5)
        
        # Get feature importance
        importance = MODEL.get_feature_importance(features_tensor)
        
        # Update metrics
        TOTAL_PREDICTIONS += 1
        LAST_PREDICTION_TIME = datetime.utcnow()
        
        # Determine confidence
        confidence = 'high' if probability > 0.85 or probability < 0.15 else \
                    'medium' if probability > 0.7 or probability < 0.3 else 'low'
        
        response = PredictionResponse(
            suitable_for_kangaroo_care=bool(prediction),
            probability=probability,
            confidence=confidence,
            feature_importance=importance,
            model_version=MODEL_INFO['version'],
            patient_id=request.patient_id,
            request_id=request.request_id
        )
        
        duration = time.time() - start_time
        
        logger.info(
            "Prediction completed",
            prediction=prediction,
            probability=probability,
            duration_ms=duration * 1000,
            patient_id=request.patient_id
        )
        
        return response
    
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if not MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    predictions = []
    
    for instance in request.instances:
        pred_request = PredictionRequest(features=instance)
        pred_response = await predict(pred_request)
        predictions.append(pred_response)
    
    suitable_count = sum(1 for p in predictions if p.suitable_for_kangaroo_care)
    avg_probability = np.mean([p.probability for p in predictions])
    processing_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        suitable_count=suitable_count,
        average_probability=avg_probability,
        model_version=MODEL_INFO['version'],
        processing_time_ms=processing_time,
        request_id=request.request_id
    )


# Explainability endpoint
@app.post("/explain", response_model=ExplainabilityResponse)
async def explain_prediction(request: ExplainabilityRequest):
    """Get detailed explanation for a prediction."""
    if not MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # First get the prediction
    pred_request = PredictionRequest(features=request.features)
    prediction = await predict(pred_request)
    
    # Generate interpretation
    features_dict = request.features.model_dump()
    interpretation_parts = []
    
    # Analyze each feature's contribution
    for feature, importance in prediction.feature_importance.items():
        value = features_dict[feature]
        
        if importance > 0.2:  # Significant contribution
            if feature == 'oxygen_saturation' and value >= 95:
                interpretation_parts.append(f"High oxygen saturation ({value}%) strongly supports readiness")
            elif feature == 'weight_grams' and value > 1500:
                interpretation_parts.append(f"Adequate weight ({value}g) indicates stability")
            elif feature == 'heart_rate' and 110 <= value <= 130:
                interpretation_parts.append(f"Stable heart rate ({value} bpm) is favorable")
    
    interpretation = ". ".join(interpretation_parts) if interpretation_parts else \
                    "Multiple factors contribute to this assessment"
    
    return ExplainabilityResponse(
        prediction=prediction,
        explanation_type=request.explanation_type,
        feature_importance=prediction.feature_importance,
        interpretation=interpretation
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)