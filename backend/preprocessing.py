import pandas as pd
import re

def clean_text(text):
    text = str(text)
    
    # 1. Strip dataset-specific data leaks (The "Cheat Codes")
    text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text, flags=re.IGNORECASE) # Removes "WASHINGTON (Reuters) - "
    text = re.sub(r'reuters', '', text, flags=re.IGNORECASE)                # Blankets removes the word Reuters
    text = re.sub(r'Photo by.*?Getty Images\.?', '', text, flags=re.IGNORECASE) # Removes Getty Image credits
    text = re.sub(r'pic\.twitter\.com\/\w+', '', text, flags=re.IGNORECASE) # Removes Twitter image links
    text = re.sub(r'@\w+', '', text)                                        # Removes Twitter handles
    
    # 2. General NLP Cleaning
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Removes standard URLs
    text = re.sub(r'[^a-z0-9\s]', '', text)           # Removes punctuation
    text = re.sub(r'\s+', ' ', text).strip()          # Removes extra spaces
    
    return text

def load_and_preprocess_data(true_path, fake_path, test_mode=False):
    print("Loading datasets...")
    nrows = 1000 if test_mode else None 
    
    df_true = pd.read_csv(true_path, nrows=nrows)
    df_fake = pd.read_csv(fake_path, nrows=nrows)
    
    df_true['label'] = 0
    df_fake['label'] = 1
    
    df = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)
    df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    
    print("Applying aggressive text cleaning to prevent overfitting...")
    df['clean_content'] = df['content'].apply(clean_text)
    
    return df[['clean_content', 'label']]