from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.predict import predict_bytes, load_checkpoint, get_device
from pathlib import Path


app = FastAPI(title='Intel Scene Classification API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Will Replace with Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Intel Scene Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    device = get_device()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    checkpoint_path = BASE_DIR / "models" / "best_checkpoint_intel.pth"
    
    model, checkpoint = load_checkpoint(checkpoint_path=checkpoint_path,device=device)
    
    classes = checkpoint.get('class_names')
    
    results = predict_bytes(model=model, image_bytes=image_bytes, device=device, classes=classes, top_k=3)
    return {"predictions": results}