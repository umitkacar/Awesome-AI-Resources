# 🐍 AI Python Backend Development

**Last Updated:** 2025-06-19

## Overview
Complete guide for building scalable Python backends for AI/ML services, from REST APIs to real-time inference systems.

## 🎯 Architecture Patterns

### Microservices Architecture
```python
# Service structure
ai_backend/
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── prediction.py
│   │   ├── training.py
│   │   └── health.py
│   └── middleware/
│       ├── auth.py
│       └── logging.py
├── core/
│   ├── config.py
│   ├── models.py
│   └── schemas.py
├── ml/
│   ├── predictor.py
│   ├── preprocessor.py
│   └── model_loader.py
├── utils/
│   ├── cache.py
│   └── metrics.py
└── main.py
```

## 🚀 FastAPI Implementation

### Basic AI Service
```python
# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="AI Prediction Service",
    description="Scalable ML model serving API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = "latest"
    
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    processing_time: float

# ML Model Manager
class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_version = "v1.0"
        
    async def load_model(self, version: str):
        if version not in self.models:
            # Simulate model loading
            import joblib
            self.models[version] = joblib.load(f"models/{version}/model.pkl")
        return self.models[version]
    
    async def predict(self, features: List[float], version: str = "latest"):
        import time
        start_time = time.time()
        
        model_version = self.current_version if version == "latest" else version
        model = await self.load_model(model_version)
        
        # Preprocess
        input_array = np.array(features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(input_array)[0]
        confidence = model.predict_proba(input_array).max()
        
        processing_time = time.time() - start_time
        
        return {
            "prediction": float(prediction),
            "confidence": float(confidence),
            "model_version": model_version,
            "processing_time": processing_time
        }

# Initialize model manager
model_manager = ModelManager()

# Routes
@app.get("/")
async def root():
    return {"message": "AI Backend Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": len(model_manager.models) > 0,
        "current_version": model_manager.current_version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = await model_manager.predict(
            request.features, 
            request.model_version
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
class BatchPredictionRequest(BaseModel):
    samples: List[List[float]]
    model_version: Optional[str] = "latest"

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    predictions = []
    for sample in request.samples:
        result = await model_manager.predict(sample, request.model_version)
        predictions.append(result)
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 🔄 Asynchronous Processing

### Celery for Background Tasks
```python
# tasks.py
from celery import Celery
import numpy as np
from typing import List

# Initialize Celery
celery_app = Celery(
    'ai_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'tasks.train_model': {'queue': 'training'},
        'tasks.batch_inference': {'queue': 'inference'}
    }
)

@celery_app.task(bind=True, max_retries=3)
def train_model(self, dataset_id: str, hyperparameters: dict):
    try:
        # Load dataset
        dataset = load_dataset(dataset_id)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**hyperparameters)
        model.fit(dataset['X'], dataset['y'])
        
        # Save model
        model_path = save_model(model, dataset_id)
        
        return {
            'status': 'completed',
            'model_path': model_path,
            'metrics': evaluate_model(model, dataset)
        }
    except Exception as exc:
        # Retry on failure
        raise self.retry(exc=exc, countdown=60)

@celery_app.task
def batch_inference(sample_ids: List[str], model_version: str):
    results = []
    model = load_model(model_version)
    
    for sample_id in sample_ids:
        data = fetch_sample(sample_id)
        prediction = model.predict(data)
        results.append({
            'sample_id': sample_id,
            'prediction': prediction.tolist()
        })
    
    return results
```

### Async Endpoints with Background Tasks
```python
# Async training endpoint
@app.post("/train")
async def train_model_endpoint(
    background_tasks: BackgroundTasks,
    dataset_id: str,
    hyperparameters: dict
):
    # Queue training task
    task = train_model.delay(dataset_id, hyperparameters)
    
    # Add cleanup task
    background_tasks.add_task(cleanup_temp_files, dataset_id)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Model training started"
    }

@app.get("/train/{task_id}")
async def get_training_status(task_id: str):
    task = train_model.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'error': str(task.info)
        }
    
    return response
```

## 🔐 Authentication & Security

### JWT Authentication
```python
# auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# Protected endpoint
@app.post("/predict/secure")
async def secure_predict(
    request: PredictionRequest,
    current_user: str = Depends(get_current_user)
):
    # Log user activity
    log_prediction_request(current_user, request)
    
    # Process prediction
    result = await model_manager.predict(
        request.features, 
        request.model_version
    )
    return result
```

## 📊 Caching & Performance

### Redis Caching
```python
# cache.py
import redis
import json
import hashlib
from typing import Optional, Any
from functools import wraps

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

def cache_prediction(expiration: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = f"prediction:{hashlib.md5(
                json.dumps([args, kwargs], sort_keys=True).encode()
            ).hexdigest()}"
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache_prediction(expiration=300)
async def cached_predict(features: List[float], model_version: str):
    return await model_manager.predict(features, model_version)
```

### Database Integration
```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database

DATABASE_URL = "postgresql://user:password@localhost/aidb"

database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    features = Column(String)
    prediction = Column(Float)
    confidence = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Async database operations
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/predict/logged")
async def predict_with_logging(request: PredictionRequest):
    # Make prediction
    result = await model_manager.predict(
        request.features,
        request.model_version
    )
    
    # Log to database
    query = Prediction.__table__.insert().values(
        features=json.dumps(request.features),
        prediction=result['prediction'],
        confidence=result['confidence'],
        model_version=result['model_version']
    )
    await database.execute(query)
    
    return result
```

## 🚦 Load Balancing & Scaling

### Gunicorn Configuration
```python
# gunicorn_config.py
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'ai_backend'

# Server mechanics
daemon = False
pidfile = '/tmp/ai_backend.pid'
user = None
group = None
tmp_upload_dir = None
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 aiservice && chown -R aiservice:aiservice /app
USER aiservice

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "main:app", "-c", "gunicorn_config.py"]
```

## 🔍 Monitoring & Logging

### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Prediction processing time',
    ['model_version']
)

active_models = Gauge(
    'active_models',
    'Number of loaded models'
)

@app.get("/metrics")
async def get_metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Instrument predictions
@prediction_duration.time()
async def instrumented_predict(features, model_version):
    try:
        result = await model_manager.predict(features, model_version)
        prediction_counter.labels(
            model_version=model_version,
            status='success'
        ).inc()
        return result
    except Exception as e:
        prediction_counter.labels(
            model_version=model_version,
            status='error'
        ).inc()
        raise
```

## 🎯 Best Practices

### Error Handling
```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )
```

### API Versioning
```python
from fastapi import APIRouter

# Version 1 router
v1_router = APIRouter(prefix="/api/v1")

@v1_router.post("/predict")
async def predict_v1(request: PredictionRequest):
    # V1 logic
    pass

# Version 2 router
v2_router = APIRouter(prefix="/api/v2")

@v2_router.post("/predict")
async def predict_v2(request: PredictionRequest):
    # V2 logic with new features
    pass

app.include_router(v1_router)
app.include_router(v2_router)
```

## 🔗 Integration Examples

### gRPC Service
```python
# grpc_server.py
import grpc
from concurrent import futures
import prediction_pb2
import prediction_pb2_grpc

class PredictionService(prediction_pb2_grpc.PredictionServiceServicer):
    def Predict(self, request, context):
        # Process prediction
        result = model_manager.predict(
            list(request.features),
            request.model_version
        )
        
        return prediction_pb2.PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence']
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionService(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

---

*Building robust and scalable Python backends for AI/ML applications* 🐍🚀