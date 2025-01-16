from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

app = FastAPI(title="Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root() -> Dict[str, str]:
    return {"message": "This is backend"}

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)