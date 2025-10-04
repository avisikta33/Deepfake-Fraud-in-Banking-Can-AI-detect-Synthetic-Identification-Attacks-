"""
src/bert_model.py
BERT-based Text Analysis for Synthetic Identity Detection
Analyzes KYC documents, application text, and identity narratives
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

class SyntheticIdentityDataset(Dataset):
    """Dataset for synthetic identity text detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTSyntheticIdentityDetector:
    """BERT model for detecting synthetic identities through text analysis"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        print(f"✓ BERT model loaded on {self.device}")
    
    def prepare_dataloaders(self, train_texts, train_labels, 
                           val_texts, val_labels, batch_size=16):
        """Prepare DataLoaders for training"""
        
        train_dataset = SyntheticIdentityDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = SyntheticIdentityDataset(
            val_texts, val_labels, self.tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / total_samples
        
        return avg_loss, accuracy.item()
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_samples += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / total_samples
        
        return avg_loss, accuracy.item(), all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=4, learning_rate=2e-5):
        """Train the BERT model"""
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), 
                          'models/bert_text_analyzer/best_model.pt')
                print(f'✓ Model saved with accuracy: {val_acc:.4f}')
        
        return history
    
    def predict(self, texts, batch_size=16):
        """Make predictions on new text"""
        self.model.eval()
        
        dataset = SyntheticIdentityDataset(
            texts, [0] * len(texts), self.tokenizer  # Dummy labels
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def extract_features(self, text):
        """Extract BERT embeddings for text"""
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return cls_embedding

# Text Analysis Utilities
class TextFeatureExtractor:
    """Extract statistical features from text for fraud detection"""
    
    @staticmethod
    def extract_linguistic_features(text):
        """Extract linguistic features that may indicate synthetic content"""
        features = {}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()])
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Character-level features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if len(text) > 0 else 0
        
        # Consistency indicators
        features['repetition_score'] = TextFeatureExtractor.calculate_repetition(text)
        features['coherence_score'] = TextFeatureExtractor.calculate_coherence(text)
        
        return features
    
    @staticmethod
    def calculate_repetition(text):
        """Calculate text repetition score"""
        words = text.lower().split()
        if len(words) < 2:
            return 0
        unique_words = len(set(words))
        return 1 - (unique_words / len(words))
    
    @staticmethod
    def calculate_coherence(text):
        """Simple coherence score based on sentence structure"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Check if sentences have varying lengths (more coherent)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        variance = np.var(lengths)
        return min(1.0, variance / 10.0)

# Example usage
if __name__ == "__main__":
    # Initialize BERT detector
    detector = BERTSyntheticIdentityDetector()
    
    # Example texts (would come from KYC documents)
    sample_texts = [
        "I am John Doe, born in New York on January 1, 1990. I work as a software engineer.",
        "My name John Doe I born New York work engineer good job money"  # Synthetic/poor quality
    ]
    
    print("✓ BERT model initialized")
    print("✓ Text feature extraction ready")
    print(f"✓ Device: {detector.device}")