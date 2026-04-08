from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion
from xgboost import XGBClassifier
import joblib
import os
import numpy as np
from evaluation import print_classification_report, plot_confusion_matrix

def train_ensemble_model(df):
    print("\n[Ensemble] Splitting data securely...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("[Ensemble] Building Focused NLP Feature Extractors (55k features)...")
    # 1. Tightly controlled Word Analyzer
    word_vectorizer = TfidfVectorizer(
        max_features=40000, 
        ngram_range=(1, 2), 
        sublinear_tf=True,  
        stop_words='english'
    )
    
    # 2. Cleaner Character Analyzer
    char_vectorizer = TfidfVectorizer(
        max_features=15000, 
        ngram_range=(3, 5), 
        analyzer='char_wb', 
        sublinear_tf=True
    )
    
    advanced_tfidf = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    print("[Ensemble] Vectorizing data...")
    X_train_tfidf = advanced_tfidf.fit_transform(train_df['clean_content'])
    X_val_tfidf = advanced_tfidf.transform(val_df['clean_content'])
    
    # Calculate scale weights
    num_real = np.sum(train_df['label'] == 0)
    num_fake = np.sum(train_df['label'] == 1)
    scale_weight = float(num_real) / num_fake
    
    print("[Ensemble] Training Streamlined Voting Classifier (LR + XGBoost only)...")
    
    # Tuned Logistic Regression using 'liblinear' for better sparse text handling
    model_lr = LogisticRegression(
        C=5, 
        max_iter=3000, 
        class_weight='balanced', 
        solver='liblinear',
        random_state=42
    )
    
    # Simplified, smarter XGBoost
    model_xgb = XGBClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.07, 
        scale_pos_weight=scale_weight, 
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42, 
        n_jobs=-1
    )
    
    # 2-Model Voting Classifier
    ensemble = VotingClassifier(
        estimators=[('lr', model_lr), ('xgb', model_xgb)], 
        voting='soft',
        weights=[2.0, 3.0] # Give XGBoost a slight edge
    )
    
    ensemble.fit(X_train_tfidf, train_df['label'])
    
    print("[Ensemble] Evaluating on Untouched Validation Data (Threshold: 0.55)...")
    
    # CUSTOM THRESHOLD LOGIC
    # Get the raw probabilities instead of the hard guesses
    y_prob = ensemble.predict_proba(X_val_tfidf)[:, 1]
    
    # Only guess 'Fake' (1) if the model is > 55% confident
    y_pred = (y_prob >= 0.55).astype(int)
    
    print_classification_report(val_df['label'], y_pred, "Optimized_Ensemble_Model")
    plot_confusion_matrix(val_df['label'], y_pred, "Optimized_Ensemble_Model")
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(ensemble, os.path.join(models_dir, 'saved_ensemble_model.pkl'))
    joblib.dump(advanced_tfidf, os.path.join(models_dir, 'saved_tfidf_vectorizer.pkl'))
    
    return ensemble, advanced_tfidf