import torch
from torch import nn
import os
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- ROBUST CUSTOM TRAINER (CRASH-PROOF FOR HUGGING FACE) ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # **kwargs is absolutely necessary here to prevent HF errors
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Apply the mathematically calculated penalty weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

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

def train_distilbert_model(df):
    print("\n[DistilBERT] Phase 1: Splitting data natively (No Oversampling)...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("[DistilBERT] Phase 2: Calculating Mathematical Class Weights...")
    # Calculate the exact mathematical penalty for missing Fake News
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_df['label']
    )
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
    print(f"[DistilBERT] Penalty Multipliers - Real: {class_weights_array[0]:.2f}x, Fake: {class_weights_array[1]:.2f}x")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print(f"[DistilBERT] Phase 3: Tokenizing ({len(train_df)} native rows at length 256)...")
    train_encodings = tokenizer(train_df['clean_content'].tolist(), truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_df['clean_content'].tolist(), truncation=True, padding=True, max_length=256)
    
    train_dataset = FakeNewsDataset(train_encodings, train_df['label'].tolist())
    val_dataset = FakeNewsDataset(val_encodings, val_df['label'].tolist())
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=200,
        weight_decay=0.01,
        fp16=True, 
        eval_strategy="epoch",  
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none" 
    )
    
    print("[DistilBERT] Phase 4: Training with Custom Weighted Loss Function...")
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    print("Saving Best DistilBERT model...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'models', 'saved_distilbert')
    os.makedirs(models_dir, exist_ok=True)
    model.save_pretrained(models_dir)
    tokenizer.save_pretrained(models_dir)
    
    return model, tokenizer