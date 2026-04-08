import os
import torch
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from preprocessing import clean_text
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_models():
    print("Loading saved models from disk...")
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load Machine Learning Ensemble
    ensemble = joblib.load(os.path.join(models_dir, 'saved_ensemble_model.pkl'))
    tfidf = joblib.load(os.path.join(models_dir, 'saved_tfidf_vectorizer.pkl'))
    
    # Load Deep Learning Model
    distilbert_path = os.path.join(models_dir, 'saved_distilbert')
    tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
    distilbert = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
    
    return ensemble, tfidf, tokenizer, distilbert

def predict_news(text, ensemble, tfidf, tokenizer, distilbert):
    # Clean the input text exactly how the model was trained
    cleaned = clean_text(text)
    
    # 1. Ensemble Prediction
    vec_text = tfidf.transform([cleaned])
    ens_pred = ensemble.predict(vec_text)[0]
    ens_label = "🚨 FAKE NEWS" if ens_pred == 1 else "✅ REAL NEWS"
    
    # 2. DistilBERT Prediction
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = distilbert(**inputs)
        dl_pred = torch.argmax(outputs.logits, dim=1).item()
    dl_label = "🚨 FAKE NEWS" if dl_pred == 1 else "✅ REAL NEWS"
    
    # Print Results
    print(f"\n{'-'*30}")
    print(f"Ensemble Model (TF-IDF): {ens_label}")
    print(f"DistilBERT (Contextual): {dl_label}")
    print(f"{'-'*30}\n")

if __name__ == "__main__":
    ensemble, tfidf, tokenizer, distilbert = load_models()
    print("\n✅ Models loaded successfully!")
    print("Type a news headline or paragraph below to test them.")
    print("(Type 'quit' or 'exit' to stop)")
    
    while True:
        user_input = input("\nEnter news text: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if user_input.strip() == "":
            continue
            
        predict_news(user_input, ensemble, tfidf, tokenizer, distilbert)