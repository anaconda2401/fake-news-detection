import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import functions from existing inference script
from inference import load_models, clean_text

app = FastAPI(title="Fake News Models API")

# Allow Node.js backend to call this directly via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models in memory
ensemble, tfidf, tokenizer, distilbert = None, None, None, None

class NewsRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global ensemble, tfidf, tokenizer, distilbert
    print("Initializing Models into memory...")
    ensemble, tfidf, tokenizer, distilbert = load_models()
    print("Models Initialized!")

@app.post("/predict")
async def predict(req: NewsRequest):
    text = req.text
    # Clean the input text exactly how the model was trained
    cleaned = clean_text(text) if "clean_text" in globals() else text

    # Handle missing clean_text if it was in preprocessing.py:
    # Let's import clean_text from preprocessing.py instead if needed.
    from preprocessing import clean_text as pp_clean
    cleaned = pp_clean(text)
    
    # 1. Ensemble Prediction
    vec_text = tfidf.transform([cleaned])
    ens_pred = ensemble.predict(vec_text)[0]
    
    # Need probabilities instead of just 1 / 0 if possible
    # Assuming standard scikit-learn models have predict_proba
    try:
        ens_proba = ensemble.predict_proba(vec_text)[0]
        ens_confidence = float(max(ens_proba))
    except Exception:
        ens_confidence = 1.0 # fallback

    ens_label = 1 if ens_pred == 1 else 0
    
    # 2. DistilBERT Prediction
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = distilbert(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        dl_pred = torch.argmax(logits, dim=1).item()
        dl_confidence = float(probs[0][dl_pred])
        
    dl_label = int(dl_pred)

    return {
        "ensemble": {
            "prediction": ens_label,
            "label": "Fake News" if ens_label == 1 else "Real News",
            "confidence": ens_confidence
        },
        "distilbert": {
            "prediction": dl_label,
            "label": "Fake News" if dl_label == 1 else "Real News",
            "confidence": dl_confidence
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
