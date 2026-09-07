from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="NeuroScan AI - Brain Tumor Segmentation Engine",
    description="FastAPI Backend for Automated Brain Tumor MRI Segmentation & Quantitative Analysis",
    version="1.0.0"
)

# Enable CORS for frontend development servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "NeuroScan AI Segmentation Backend is active.",
        "documentation": "/docs",
        "health": "/health",
        "segment_endpoint": "/segment"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
