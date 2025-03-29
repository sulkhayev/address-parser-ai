import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
import joblib
import pickle
import spacy
import re
from tqdm import tqdm
import argparse
import os

# Optional BERT support - only imported if use_bert=True
try:
    import torch
    from transformers import BertTokenizer, BertModel
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class AddressParserTrainer:
    """
    Advanced training pipeline for the Azerbaijani address parser
    """
    def __init__(self, use_bert=False):
        self.use_bert = use_bert
        if use_bert and not BERT_AVAILABLE:
            print("Warning: BERT dependencies not available. Falling back to traditional models.")
            self.use_bert = False
            
        self.components = ['city', 'region', 'settlement', 'street', 'building', 'apartment']
        self.models = {}
        self.vectorizer = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.label_encoders = {}
        self.label_decoders = {}
        
        if self.use_bert:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            # Note: For Azerbaijani, you might want to use a multilingual BERT model
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    def prepare_data(self, csv_file):
        """
        Load and prepare data from CSV file
        
        Args:
            csv_file: Path to CSV with address data
            
        Returns:
            DataFrame with prepared data
        """
        # Load data
        df = pd.read_csv(csv_file)
        
        # Fill NA values
        df = df.fillna('NONE')  # Special value for missing components
        
        return df
    
    def train_test_split(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(df, test_size=test_size, random_state=random_state)
    
    def train_traditional_models(self, train_df):
        """
        Train traditional ML models for each address component
        
        Args:
            train_df: Training data DataFrame
        """
        # For each component, we'll train a separate classifier
        for component in self.components:
            print(f"\nTraining model for: {component}")
            
            # Create feature pipeline
            pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            # Train the model
            pipeline.fit(train_df['address'], train_df[component])
            
            # Store the model
            self.models[component] = pipeline
            
            print(f"Model for {component} trained successfully.")
    
    def train_bert_model(self, train_df, epochs=5, batch_size=16, learning_rate=2e-5):
        """
        Train a BERT-based model for address parsing
        
        Args:
            train_df: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        if not self.use_bert:
            print("BERT model not enabled. Use use_bert=True in initialization.")
            return
        
        # Create a dataset class for BERT
        class AddressDataset(Dataset):
            def __init__(self, dataframe, tokenizer, components):
                self.data = dataframe
                self.tokenizer = tokenizer
                self.components = components
                
                # Create label encoders for each component
                self.label_encoders = {}
                self.label_decoders = {}
                
                for component in components:
                    unique_values = dataframe[component].unique()
                    self.label_encoders[component] = {val: i for i, val in enumerate(unique_values)}
                    self.label_decoders[component] = {i: val for i, val in enumerate(unique_values)}
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                address = self.data.iloc[idx]['address']
                
                # Tokenize the address
                encoding = self.tokenizer(address, padding='max_length', truncation=True, 
                                         max_length=128, return_tensors='pt')
                
                # Get labels
                labels = []
                for component in self.components:
                    value = self.data.iloc[idx][component]
                    label_id = self.label_encoders[component].get(value, 0)  # 0 for unknown
                    labels.append(label_id)
                
                # Convert to tensor
                labels = torch.tensor(labels, dtype=torch.long)
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': labels
                }
        
        # Create a BERT model for address classification
        class BertAddressParser(nn.Module):
            def __init__(self, num_labels_per_component):
                super(BertAddressParser, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
                
                # Separate classifier heads for each component
                self.classifiers = nn.ModuleList([
                    nn.Linear(self.bert.config.hidden_size, num_labels) 
                    for num_labels in num_labels_per_component
                ])
                
            def forward(self, input_ids, attention_mask):
                # Get BERT embeddings
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state
                
                # Use [CLS] token representation for classification
                cls_output = sequence_output[:, 0, :]
                
                # Apply classifier heads
                logits = [classifier(cls_output) for classifier in self.classifiers]
                
                return logits
        
        # Create the dataset
        dataset = AddressDataset(train_df, self.bert_tokenizer, self.components)
        
        # Store label encoders and decoders for later use
        self.label_encoders = dataset.label_encoders
        self.label_decoders = dataset.label_decoders
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize the model
        num_labels_per_component = [len(dataset.label_encoders[comp]) for comp in self.components]
        self.bert_model = BertAddressParser(num_labels_per_component).to(self.device)
        
        # Define optimizer and loss function
        optimizer = optim.AdamW(self.bert_model.parameters(), lr=learning_rate)
        loss_fns = [nn.CrossEntropyLoss() for _ in self.components]
        
        # Training loop
        self.bert_model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.bert_model(input_ids, attention_mask)
                
                # Calculate loss for each component
                loss = sum(loss_fn(logit, labels[:, i]) 
                           for i, (logit, loss_fn) in enumerate(zip(logits, loss_fns)))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_size * (progress_bar.n + 1))})
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
        
        print("BERT model training completed.")
    
    def evaluate_traditional_models(self, test_df):
        """
        Evaluate traditional ML models on test data
        
        Args:
            test_df: Test data DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        for component in self.components:
            if component not in self.models:
                print(f"No model found for {component}")
                continue
                
            model = self.models[component]
            
            # Make predictions
            predictions = model.predict(test_df['address'])
            
            # Compute metrics
            report = classification_report(test_df[component], predictions, output_dict=True)
            conf_matrix = confusion_matrix(test_df[component], predictions)
            
            results[component] = {
                'report': report,
                'confusion_matrix': conf_matrix,
                'accuracy': accuracy_score(test_df[component], predictions)
            }
            
            print(f"\nResults for {component}:")
            print(f"Accuracy: {results[component]['accuracy']:.4f}")
            
        return results
    
    def evaluate_bert_model(self, test_df, batch_size=16):
        """
        Evaluate BERT model on test data
        
        Args:
            test_df: Test data DataFrame
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.use_bert or not self.bert_model:
            print("BERT model not trained.")
            return None
        
        # Create dataset
        class TestDataset(Dataset):
            def __init__(self, dataframe, tokenizer, components, label_encoders):
                self.data = dataframe
                self.tokenizer = tokenizer
                self.components = components
                self.label_encoders = label_encoders
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                address = self.data.iloc[idx]['address']
                
                # Tokenize
                encoding = self.tokenizer(address, padding='max_length', truncation=True, 
                                         max_length=128, return_tensors='pt')
                
                # Get labels
                labels = []
                for component in self.components:
                    value = self.data.iloc[idx][component]
                    label_id = self.label_encoders[component].get(value, 0)
                    labels.append(label_id)
                
                labels = torch.tensor(labels, dtype=torch.long)
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': labels,
                    'idx': idx
                }
        
        # Create dataset and dataloader
        test_dataset = TestDataset(
            test_df, self.bert_tokenizer, self.components, self.label_encoders
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Evaluation
        self.bert_model.eval()
        
        all_predictions = [[] for _ in self.components]
        all_labels = [[] for _ in self.components]
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.bert_model(input_ids, attention_mask)
                
                # Get predictions
                for i, logit in enumerate(logits):
                    predictions = torch.argmax(logit, dim=1)
                    all_predictions[i].extend(predictions.cpu().numpy())
                    all_labels[i].extend(labels[:, i].cpu().numpy())
        
        # Calculate metrics
        results = {}
        for i, component in enumerate(self.components):
            # Convert numeric labels back to original values
            pred_values = [self.label_decoders[component][pred] for pred in all_predictions[i]]
            true_values = [self.label_decoders[component][label] for label in all_labels[i]]
            
            # Compute metrics
            report = classification_report(true_values, pred_values, output_dict=True)
            accuracy = accuracy_score(true_values, pred_values)
            
            results[component] = {
                'report': report,
                'accuracy': accuracy
            }
            
            print(f"\nResults for {component}:")
            print(f"Accuracy: {results[component]['accuracy']:.4f}")
        
        return results
    
    def save_models(self, output_dir):
        """
        Save trained models to disk
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save traditional models
        for component, model in self.models.items():
            filename = os.path.join(output_dir, f"{component}_model.pkl")
            joblib.dump(model, filename)
            print(f"Saved model for {component} to {filename}")
        
        # Save BERT model if available
        if self.use_bert and self.bert_model:
            bert_dir = os.path.join(output_dir, "bert_model")
            os.makedirs(bert_dir, exist_ok=True)
            
            # Save model
            torch.save(self.bert_model.state_dict(), os.path.join(bert_dir, "model.pt"))
            
            # Save label encoders/decoders
            with open(os.path.join(bert_dir, "label_encoders.pkl"), "wb") as f:
                pickle.dump(self.label_encoders, f)
            
            with open(os.path.join(bert_dir, "label_decoders.pkl"), "wb") as f:
                pickle.dump(self.label_decoders, f)
            
            print(f"Saved BERT model to {bert_dir}")
    
    def load_models(self, input_dir):
        """
        Load trained models from disk
        
        Args:
            input_dir: Directory with saved models
        """
        import os
        
        # Load traditional models
        for component in self.components:
            filename = os.path.join(input_dir, f"{component}_model.pkl")
            if os.path.exists(filename):
                self.models[component] = joblib.load(filename)
                print(f"Loaded model for {component} from {filename}")
        
        # Load BERT model if enabled
        if self.use_bert:
            bert_dir = os.path.join(input_dir, "bert_model")
            model_path = os.path.join(bert_dir, "model.pt")
            
            if os.path.exists(model_path):
                # Load label encoders/decoders
                with open(os.path.join(bert_dir, "label_encoders.pkl"), "rb") as f:
                    self.label_encoders = pickle.load(f)
                
                with open(os.path.join(bert_dir, "label_decoders.pkl"), "rb") as f:
                    self.label_decoders = pickle.load(f)
                
                # Initialize model
                num_labels_per_component = [len(encoders) for encoders in self.label_encoders.values()]
                
                # Define BertAddressParser class (same as in train_bert_model)
                class BertAddressParser(nn.Module):
                    def __init__(self, num_labels_per_component):
                        super(BertAddressParser, self).__init__()
                        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
                        
                        # Separate classifier heads for each component
                        self.classifiers = nn.ModuleList([
                            nn.Linear(self.bert.config.hidden_size, num_labels) 
                            for num_labels in num_labels_per_component
                        ])
                        
                    def forward(self, input_ids, attention_mask):
                        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                        sequence_output = outputs.last_hidden_state
                        cls_output = sequence_output[:, 0, :]
                        logits = [classifier(cls_output) for classifier in self.classifiers]
                        return logits
                
                self.bert_model = BertAddressParser(num_labels_per_component).to(self.device)
                
                # Load weights
                self.bert_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded BERT model from {bert_dir}")
    
    def predict(self, address, use_bert=False):
        """
        Parse an address using trained models
        
        Args:
            address: Raw address string
            use_bert: Whether to use BERT model (if available)
            
        Returns:
            Dictionary with parsed components
        """
        # Preprocess address
        processed_address = address.upper()
        processed_address = re.sub(r'\s+', ' ', processed_address).strip()
        
        result = {comp: None for comp in self.components}
        
        if use_bert and self.bert_model:
            # Use BERT model for prediction
            encoding = self.bert_tokenizer(
                processed_address, 
                padding='max_length', 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            self.bert_model.eval()
            with torch.no_grad():
                logits = self.bert_model(input_ids, attention_mask)
                
                for i, (logit, component) in enumerate(zip(logits, self.components)):
                    prediction = torch.argmax(logit, dim=1).item()
                    result[component] = self.label_decoders[component].get(prediction, None)
        else:
            # Use traditional models
            for component, model in self.models.items():
                prediction = model.predict([processed_address])[0]
                result[component] = prediction if prediction != "NONE" else None
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Train address parser models')
    parser.add_argument('--data', type=str, required=True, help='CSV file with training data')
    parser.add_argument('--output', type=str, default='address_models', help='Output directory for models')
    parser.add_argument('--use-bert', action='store_true', help='Use BERT model (requires GPU)')
    parser.add_argument('--no-bert', action='store_true', help='Disable BERT model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0-1)')
    parser.add_argument('--epochs', type=int, default=5, help='BERT training epochs')
    
    args = parser.parse_args()
    
    # Determine whether to use BERT
    use_bert = False
    if args.use_bert and not args.no_bert:
        if not BERT_AVAILABLE:
            print("Warning: BERT dependencies not available. Falling back to traditional models.")
        else:
            use_bert = True
    
    # Initialize trainer
    trainer = AddressParserTrainer(use_bert=use_bert)
    
    # Load and prepare data
    data = trainer.prepare_data(args.data)
    print(f"Loaded {len(data)} address samples")
    
    # Split into train and test sets
    train_df, test_df = trainer.train_test_split(data, test_size=args.test_size)
    print(f"Training set: {len(train_df)} samples")
    print(f"Testing set: {len(test_df)} samples")
    
    # Train traditional models
    trainer.train_traditional_models(train_df)
    
    # Train BERT model if enabled
    if use_bert:
        trainer.train_bert_model(train_df, epochs=args.epochs)
    
    # Evaluate models
    print("\n--- Evaluating Traditional Models ---")
    trad_results = trainer.evaluate_traditional_models(test_df)
    
    if use_bert:
        print("\n--- Evaluating BERT Model ---")
        bert_results = trainer.evaluate_bert_model(test_df)
    
    # Save models
    trainer.save_models(args.output)
    
    # Test prediction
    test_address = "R.BAĞIROV KÜÇ EV 28/1 XƏTAİ R"
    traditional_parsed = trainer.predict(test_address, use_bert=False)
    
    print("\n=== Example Traditional Model Parsing ===")
    print(f"Address: {test_address}")
    for component, value in traditional_parsed.items():
        print(f"{component}: {value or 'None'}")
    
    if use_bert:
        print("\n=== Example BERT Model Parsing ===")
        bert_parsed = trainer.predict(test_address, use_bert=True)
        for component, value in bert_parsed.items():
            print(f"{component}: {value or 'None'}")

if __name__ == "__main__":
    main()