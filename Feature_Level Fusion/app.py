from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import io
import os
from sklearn.metrics import f1_score
import torch.nn.functional as F
import time
import tensorflow as tf
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_vit_cpu.pth")

# Create models directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper function to verify model exists
def verify_model_exists():
    if not os.path.exists(MODEL_PATH):
        # Check in Unstructured data folder
        notebook_model = os.path.join(
            os.path.dirname(__file__), 
            "..", "Unstructured data", 
            "best_vit_cpu.pth"
        )
        if os.path.exists(notebook_model):
            # Auto copy if found
            os.makedirs(MODEL_DIR, exist_ok=True)
            import shutil
            shutil.copy2(notebook_model, MODEL_PATH)
            print(f"Model copied from {notebook_model} to {MODEL_PATH}")
        else:
            raise FileNotFoundError(
                f"Model file not found in either:\n"
                f"- {MODEL_PATH}\n"
                f"- {notebook_model}\n"
                f"Please copy your trained model to one of these locations."
            )

# Load model
def load_model():
    verify_model_exists()
    print(f"Loading model from {MODEL_PATH}...")
    # Cập nhật tên model để khớp với phiên bản mới của timm
    model = timm.create_model(
        "vit_base_patch16_224.augreg_in21k",  # Tên model mới
        pretrained=False, 
        num_classes=4
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Labels
CLASSES = [
    "Benign",
    "[Malignant] early Pre-B",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B"
]

# Add preprocessing constants
MEAN = [0.485, 0.456, 0.406]  # ImageNet means
STD = [0.229, 0.224, 0.225]   # ImageNet stds
IMG_SIZE = 224

def clahe_rgb(img, clip=2.0, tile=8):
    """Apply CLAHE to RGB image"""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception as e:
        print(f"CLAHE error: {str(e)}")
        return img

# Image preprocessing
def preprocess_image(image_bytes):
    """Enhanced preprocessing pipeline matching training"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for opencv processing
        img = np.array(image)
        
        # 1. Apply CLAHE enhancement
        img = clahe_rgb(img)
        
        # 2. Resize with proper interpolation
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # 3. Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(MEAN)) / np.array(STD)
        
        # 4. Convert to tensor (CHW format)
        img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
        return img
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise

model = load_model()

# Load .h5 model for blood indices
BLOOD_MODEL_PATH = os.path.join(MODEL_DIR, "blood_cancer_model.h5")
def load_blood_model():
    return tf.keras.models.load_model(BLOOD_MODEL_PATH, compile=False)
blood_model = load_blood_model()

# Blood indices columns (order must match model training)
BLOOD_FEATURES = [
    "Age", "Cr", "WBC", "LYMp", "MIDp", "NEUTp", "LYMn", "MIDn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDWSD", "RDWCV", "PLT", "MPV", "PDW", "PCT", "PLCR"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return {
                "success": False,
                "error": "Chỉ hỗ trợ file ảnh (PNG, JPG)"
            }

        # Read and preprocess image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            return {
                "success": False, 
                "error": "File rỗng"
            }

        img_tensor = preprocess_image(image_bytes)
        
        # Run prediction
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img_tensor.to(DEVICE))
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # Get main prediction
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            
            # Get all probabilities
            class_probs = {
                CLASSES[i]: float(p) 
                for i, p in enumerate(probs)
            }
        
        return {
            "success": True,
            "class": CLASSES[pred_class],
            "confidence": confidence,
            "class_probabilities": class_probs,
            "processing_time": f"{(time.time() - start_time)*1000:.0f}ms"
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/predict_blood")
async def predict_blood(data: dict):
    try:
        # Chuẩn hóa giá trị Gender: Nam=1, Nữ=0 (nếu là string)
        if "Gender" in data:
            if isinstance(data["Gender"], str):
                if data["Gender"].strip().lower() in ["nam", "male", "1"]:
                    data["Gender"] = 1
                elif data["Gender"].strip().lower() in ["nữ", "nu", "female", "0"]:
                    data["Gender"] = 0
                else:
                    data["Gender"] = None
        # Validate input
        features = [data.get(f, None) for f in BLOOD_FEATURES]
        if None in features:
            return {"success": False, "error": "Missing features: " + str([BLOOD_FEATURES[i] for i, v in enumerate(features) if v is None])}
        X = pd.DataFrame([features], columns=BLOOD_FEATURES)
        X_np = X.values.astype(np.float32)
        X_np = X_np.reshape((X_np.shape[0], 1, X_np.shape[1]))
        pred_blood = blood_model.predict(X_np)[0][0]
        pred_class = int(round(pred_blood))
        return {
            "success": True,
            "predicted_class": pred_class,
            "probability": float(pred_blood)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict_combined")
async def predict_combined(file: UploadFile = File(...), data: dict = None):
    try:
        # 1. Predict from image
        image_bytes = await file.read()
        img_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            outputs = model(img_tensor.to(DEVICE))
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_class_img = int(np.argmax(probs))
            confidence_img = float(probs[pred_class_img])
        # 2. Predict from blood indices
        features = [data.get(f, None) for f in BLOOD_FEATURES]
        if None in features:
            return {"success": False, "error": "Missing blood features: " + str([BLOOD_FEATURES[i] for i, v in enumerate(features) if v is None])}
        X = pd.DataFrame([features], columns=BLOOD_FEATURES)
        X_np = X.values.astype(np.float32)
        X_np = X_np.reshape((X_np.shape[0], 1, X_np.shape[1]))
        pred_blood = blood_model.predict(X_np)[0][0]
        pred_class_blood = int(round(pred_blood))
        # 3. Combine (simple rule: nếu 1 trong 2 dự đoán là ung thư thì trả về ung thư, hoặc weighted average)
        combined_score = (confidence_img + pred_blood) / 2
        combined_class = 1 if combined_score > 0.5 else 0
        return {
            "success": True,
            "image_class": CLASSES[pred_class_img],
            "image_confidence": confidence_img,
            "blood_predicted_class": pred_class_blood,
            "blood_probability": float(pred_blood),
            "combined_score": float(combined_score),
            "combined_class": combined_class
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    try:
        print("Loading model...")
        model = load_model()
        print("Model loaded successfully!")
        
        print("\nStarting FastAPI server...")
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
