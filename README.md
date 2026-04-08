# TruthLens - Fake News Detection System

A full-stack Fake News Detection system combining Machine Learning (TF-IDF + Ensemble) and Deep Learning (DistilBERT) to classify news as Real or Fake with confidence scores.

---

## Project Overview

This project includes:
- Ensemble ML model (TF-IDF + Logistic Regression + XGBoost)
- DistilBERT transformer model
- FastAPI backend for model inference
- Node.js + Express frontend server
- Web interface for real-time predictions

---

## Architecture

User (Browser) 
➔ Node.js Server (Express) 
➔ FastAPI (Python API) 
➔ Models:
   - Ensemble (TF-IDF + ML)
   - DistilBERT (Transformer)

---

## Project Structure

```text
fake-news-detection/
├── backend/
│   ├── main.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── models/
├── frontend/
│   ├── index.js
│   ├── package.json
│   └── public/
│       ├── index.html
│       ├── script.js
│       └── style.css
├── training/
│   ├── main.py
│   ├── ensemble_model.py
│   ├── distilbert_model.py
│   ├── preprocessing.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone repository
```bash
git clone [https://github.com/anaconda2401/fake-news-detection.git](https://github.com/anaconda2401/fake-news-detection.git)
cd fake-news-detection
```

### Setup Python environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Setup frontend
```bash
cd frontend
npm install
```

---

## Running the Project

### Start FastAPI backend
```bash
cd backend
python main.py
```
*Runs on: http://127.0.0.1:5000*

### Start Node.js server
```bash
cd frontend
npm start
```
*Runs on: http://localhost:3000*

### Open in browser
Navigate to: http://localhost:3000

---

## How It Works

1. **Text preprocessing**
   - Removes noise (URLs, special characters)
   - Prevents dataset leakage (Reuters tags etc.)
   - Normalizes text
2. **Ensemble model**
   - TF-IDF (word + character features)
   - Logistic Regression + XGBoost
   - Voting classifier
3. **DistilBERT model**
   - Transformer-based NLP
   - Context-aware predictions
   - Handles class imbalance with weighted loss

---

## Output Format

```json
{
  "ensemble": {
    "prediction": 1,
    "label": "Fake News",
    "confidence": 0.92
  },
  "distilbert": {
    "prediction": 0,
    "label": "Real News",
    "confidence": 0.88
  }
}
```

---

## Models Used

**Ensemble:**
- TF-IDF
- Logistic Regression
- XGBoost
- Voting Classifier

**DistilBERT:**
- distilbert-base-uncased
- Fine-tuned on dataset

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Log Loss
- Confusion Matrix

---

## Training

To retrain the models:
```bash
cd training
python main.py
```

---

## Highlights

- Hybrid ML + DL system
- Data leakage prevention
- Class imbalance handling
- Real-time API system
- Clean web interface

---

## Future Improvements

- Multilingual support
- Mobile integration
- Cloud deployment
- Live news API
- Advanced transformer models
