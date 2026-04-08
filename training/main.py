from preprocessing import load_and_preprocess_data
from ensemble_model import train_ensemble_model
from distilbert_model import train_distilbert_model

def main():
    print("=== Indian Fake News Mega-Pipeline ===")
    
    # Passing BOTH dataset paths here
    df = load_and_preprocess_data(
        bharat_path='dataset/bharatfakenewskosh.xlsx', 
        simple_csv_path='dataset/news_dataset.csv', 
        test_mode=False 
    )
    
    print("\n=== Phase 1: Ensemble Model (TF-IDF + ML) ===")
    ensemble_model, vectorizer = train_ensemble_model(df)
    
    print("\n=== Phase 2: DistilBERT Model (Deep Learning) ===")
#    distilbert_model, tokenizer = train_distilbert_model(df)
    
    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()