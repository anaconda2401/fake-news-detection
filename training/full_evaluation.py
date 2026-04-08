import os
import torch
import joblib
import pandas as pd
import numpy as np
import warnings
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, log_loss,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from preprocessing import load_and_preprocess_data

# Suppress warnings for a clean terminal output
warnings.filterwarnings('ignore')

# We need this class to format the data for DistilBERT
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def print_master_metrics(y_true, y_pred, y_prob, model_name):
    """Calculates and prints an exhaustive list of advanced metrics."""
    
    # 1. Standard Metrics
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 2. Advanced Probability Metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    loss = log_loss(y_true, y_prob)
    
    # 3. Detailed Class Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"\n{'='*60}")
    print(f" 📊 {model_name.upper()} - COMPREHENSIVE DIAGNOSTICS ")
    print(f"{'='*60}")
    print(f"► OVERALL ACCURACY   : {acc * 100:.2f}%  (Total correct predictions)")
    print(f"► MATTHEWS CORR (MCC): {mcc:.4f}   (Correlation between true and predicted)")
    print(f"► ROC-AUC SCORE      : {roc_auc:.4f}   (Ability to distinguish Real vs Fake)")
    print(f"► LOG LOSS           : {loss:.4f}   (Confidence penalty - lower is better)")
    print(f"{'-'*60}")
    print(f"► FAKE NEWS F1-SCORE : {f1:.4f}   (Balance of precision and recall)")
    print(f"► FAKE NEWS RECALL   : {recall:.4f}   (Percentage of actual fake news caught)")
    print(f"► FAKE NEWS PRECISION: {precision:.4f}   (Percentage of fake guesses that were right)")
    print(f"{'-'*60}")
    
    # Text-based Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("► CONFUSION MATRIX:")
    print(f"                     Predicted REAL(0)   Predicted FAKE(1)")
    print(f"   Actual REAL(0)  | {cm[0][0]:<17} | {cm[0][1]}")
    print(f"   Actual FAKE(1)  | {cm[1][0]:<17} | {cm[1][1]}")
    print(f"{'='*60}\n")


def main():
    print("=== Loading Mega-Dataset for Evaluation ===")
    df = load_and_preprocess_data(
        bharat_path='dataset/bharatfakenewskosh.xlsx', 
        simple_csv_path='dataset/news_dataset.csv', 
        test_mode=False 
    )
    
    # Recreate the exact 20% test split from your training phase
    _, X_test, _, y_test = train_test_split(
        df['clean_content'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Convert formats for easy metric calculation
    texts = X_test.tolist()
    y_true = y_test.tolist()
    
    # ==========================================
    # 1. EVALUATE ENSEMBLE MODEL
    # ==========================================
    print("\nLoading Ensemble Model from disk...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'models')
    
    ensemble = joblib.load(os.path.join(models_dir, 'saved_ensemble_model.pkl'))
    tfidf = joblib.load(os.path.join(models_dir, 'saved_tfidf_vectorizer.pkl'))
    
    print("Running Ensemble Inference...")
    X_test_tfidf = tfidf.transform(texts)
    ens_pred = ensemble.predict(X_test_tfidf)
    ens_prob = ensemble.predict_proba(X_test_tfidf)[:, 1] # Get probability of "Fake" class
    
    print_master_metrics(y_true, ens_pred, ens_prob, "Classical Ensemble (TF-IDF)")

    # ==========================================
    # 2. EVALUATE DISTILBERT
    # ==========================================
    print("\nLoading DistilBERT from disk (this takes a moment)...")
    distilbert_dir = os.path.join(models_dir, 'saved_distilbert')
    tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)
    model = DistilBertForSequenceClassification.from_pretrained(distilbert_dir)
    
    print("Tokenizing test data...")
    test_encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    test_dataset = FakeNewsDataset(test_encodings, y_true)
    
    # We use Trainer just for incredibly fast, batched prediction on the GPU
    trainer = Trainer(model=model)
    
    print("Running DistilBERT Inference (Scanning ~7000 articles)...")
    predictions = trainer.predict(test_dataset)
    
    # Convert raw logits to probabilities using Softmax
    import torch.nn.functional as F
    logits = torch.tensor(predictions.predictions)
    probs = F.softmax(logits, dim=-1).numpy()
    
    bert_pred = predictions.predictions.argmax(-1)
    bert_prob = probs[:, 1] # Get probability of "Fake" class
    
    print_master_metrics(y_true, bert_pred, bert_prob, "DistilBERT Contextual Engine")

if __name__ == "__main__":
    main()