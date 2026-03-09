from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
import sys

# Define FastAPI app
app = FastAPI(
    title="Galaxy Classification API",
    description="API to predict the probabilities of galaxy morphologies from input images.",
    version="1.0.0"
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = None
# Original Kaggle Galaxy Zoo columns (Class1.1 through Class11.6 - 37 in total)
TARGET_COLS = [
    'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 
    'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 
    'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 
    'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 
    'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 
    'Class11.4', 'Class11.5', 'Class11.6'
]

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), '../../models/baseline_resnet50_best.pth')
    
    # Initialize architecture
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    # We use Sigmoid for probability regression [0, 1] across all 37 classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, len(TARGET_COLS)),
        torch.nn.Sigmoid() 
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}. API will run with uninitialized weights.", file=sys.stderr)
        
    model = model.to(device)
    model.eval()

# Preprocessing transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "Galaxy Classification API is running."}

@app.post("/predict/")
async def predict_galaxy(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
         return JSONResponse(status_code=400, content={"message": "File provided is not an image."})
         
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = outputs[0].cpu().numpy().tolist()
            
        # Format response
        result = {TARGET_COLS[i]: round(probs[i], 4) for i in range(len(TARGET_COLS))}
        
        # Find the highest probability class
        predicted_class = max(result, key=result.get)
        
        return {
            "prediction_probabilities": result,
            "top_prediction": predicted_class
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
