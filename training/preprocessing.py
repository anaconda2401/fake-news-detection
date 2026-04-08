import pandas as pd
import re
import os

def clean_text_for_bert(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(bharat_path, simple_csv_path, test_mode=False):
    print("Loading datasets...")
    nrows = 500 if test_mode else None 
    combined_data = []

    if os.path.exists(bharat_path):
        if bharat_path.endswith('.xlsx'):
            df_bharat = pd.read_excel(bharat_path, nrows=nrows)
        else:
            df_bharat = pd.read_csv(bharat_path, nrows=nrows)
            
        df_bharat = df_bharat.dropna(subset=['Eng_Trans_News_Body', 'Label'])
        df_bharat['label'] = df_bharat['Label'].apply(lambda x: 1 if str(x).strip().upper() == 'FALSE' else 0)
        df_bharat['content'] = df_bharat['Eng_Trans_News_Body']
        combined_data.append(df_bharat[['content', 'label']])

    if os.path.exists(simple_csv_path):
        df_simple = pd.read_csv(simple_csv_path, nrows=nrows)
        df_simple = df_simple.dropna(subset=['text', 'label'])
        df_simple['label'] = df_simple['label'].apply(lambda x: 1 if 'FAKE' in str(x).strip().upper() else 0)
        df_simple['content'] = df_simple['text']
        combined_data.append(df_simple[['content', 'label']])

    print("Merging and cleaning datasets...")
    df = pd.concat(combined_data, ignore_index=True)
    df['clean_content'] = df['content'].apply(clean_text_for_bert)
    df = df[df['clean_content'].str.strip() != '']
    
    print(f"Total Raw Dataset Size: {len(df)} rows.")
    return df[['clean_content', 'label']]