"""
Indian Cattle and Buffalo Breed & Age Recognition System
Main FastAPI Application
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

from .routers import prediction
from .database.db import create_tables

# Create FastAPI app
app = FastAPI(
    title="Indian Cattle & Buffalo Recognition System",
    description="AI-powered breed identification and age estimation for Indian cattle and buffaloes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(prediction.router)

# Initialize database
create_tables()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main frontend page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Serve the admin panel page."""
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Cattle Recognition API"}


@app.get("/api/info")
async def api_info():
    """Get API information."""
    return {
        "name": "Indian Cattle & Buffalo Recognition System",
        "version": "1.0.0",
        "description": "AI-powered system for identifying breeds and estimating age of Indian cattle and buffaloes",
        "supported_breeds": {
            "cattle": [
                "Gir", "Sahiwal", "Red Sindhi", "Ongole", "Tharparkar",
                "Kankrej", "Hariana", "Rathi", "Krishna Valley", "Deoni"
            ],
            "buffalo": [
                "Murrah", "Jaffarabadi", "Surti", "Mehsana", "Bhadawari",
                "Nili-Ravi", "Pandharpuri", "Nagpuri", "Toda"
            ]
        },
        "age_groups": [
            "Calf (0-6 months)",
            "Young (6-18 months)",
            "Adult (18 months - 6 years)",
            "Old (6+ years)"
        ],
        "endpoints": {
            "predict": "POST /api/predict",
            "history": "GET /api/predictions",
            "statistics": "GET /api/statistics",
            "breeds": "GET /api/breeds",
            "documentation": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
