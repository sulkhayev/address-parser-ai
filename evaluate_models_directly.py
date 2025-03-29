#!/usr/bin/env python
# Script to directly evaluate trained address parser models

import os
import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Optional imports for BERT
try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

class BertAddressParser(nn.Module):
    """BERT model for address parsing"""
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

def load_models(model_dir, use_bert=False):
    """
    Load trained models from directory
    
    Args:
        model_dir: Directory with trained models
        use_bert: Whether to load BERT model
        
    Returns:
        Dictionary of models and related data
    """
    models = {}
    components = ['city', 'region', 'settlement', 'street', 'building', 'apartment']
    
    # Load traditional models
    for component in components:
        model_path = os.path.join(model_dir, f"{component}_model.pkl")
        if os.path.exists(model_path):
            models[component] = joblib.load(model_path)
            print(f"Loaded model for {component}")
    
    bert_model = None
    bert_tokenizer = None
    label_decoders = None
    
    # Load BERT model if enabled and available
    if use_bert and BERT_AVAILABLE:
        bert_dir = os.path.join(model_dir, "bert_model")
        model_path = os.path.join(bert_dir, "model.pt")
        
        if os.path.exists(model_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Load tokenizer
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            
            # Load label decoders
            with open(os.path.join(bert_dir, "label_decoders.pkl"), "rb") as f:
                import pickle
                label_decoders = pickle.load(f)
            
            # Load label encoders
            with open(os.path.join(bert_dir, "label_encoders.pkl"), "rb") as f:
                import pickle
                label_encoders = pickle.load(f)
            
            # Initialize model
            num_labels_per_component = [len(encoders) for encoders in label_encoders.values()]
            bert_model = BertAddressParser(num_labels_per_component).to(device)
            
            # Load weights
            bert_model.load_state_dict(torch.load(model_path, map_location=device))
            bert_model.eval()
            print("Loaded BERT model")
    
    return {
        'traditional_models': models,
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
        'label_decoders': label_decoders,
        'components': components
    }

def parse_with_traditional_models(address, models):
    """
    Parse address using traditional models
    
    Args:
        address: Address string
        models: Dictionary with trained models
        
    Returns:
        Dictionary with parsed components
    """
    result = {}
    
    for component, model in models.items():
        try:
            prediction = model.predict([address])[0]
            # Convert "NONE" back to None
            result[component] = None if prediction == "NONE" else prediction
        except Exception as e:
            print(f"Error predicting {component}: {e}")
            result[component] = None
    
    return result

def parse_with_bert(address, bert_model, bert_tokenizer, label_decoders, components, device):
    """
    Parse address using BERT model
    
    Args:
        address: Address string
        bert_model: Trained BERT model
        bert_tokenizer: BERT tokenizer
        label_decoders: Dictionary with label decoders
        components: List of components to parse
        device: Torch device
        
    Returns:
        Dictionary with parsed components
    """
    result = {}
    
    # Tokenize
    encoding = bert_tokenizer(
        address, 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = bert_model(input_ids, attention_mask)
        
        for i, (logit, component) in enumerate(zip(logits, components)):
            prediction = torch.argmax(logit, dim=1).item()
            result[component] = label_decoders[component].get(prediction, None)
    
    return result

def evaluate_models(test_data, model_dir, use_bert=False, output_dir=None):
    """
    Evaluate model performance
    
    Args:
        test_data: DataFrame with test data
        model_dir: Directory with trained models
        use_bert: Whether to use BERT model
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load models
    model_data = load_models(model_dir, use_bert)
    
    traditional_models = model_data['traditional_models']
    bert_model = model_data['bert_model']
    bert_tokenizer = model_data['bert_tokenizer']
    label_decoders = model_data['label_decoders']
    components = model_data['components']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    addresses = test_data['address'].tolist()
    
    # Prepare result storage
    traditional_predictions = {comp: [] for comp in components}
    bert_predictions = {comp: [] for comp in components}
    ground_truth = {comp: [] for comp in components}
    
    # Process each address
    for i, address in enumerate(addresses):
        print(f"Evaluating address {i+1}/{len(addresses)}: {address}")
        
        # Get ground truth
        for comp in components:
            val = test_data.iloc[i][comp]
            ground_truth[comp].append(None if pd.isna(val) else val)
        
        # Get traditional model predictions
        if traditional_models:
            result = parse_with_traditional_models(address, traditional_models)
            for comp in components:
                if comp in result:
                    traditional_predictions[comp].append(result[comp])
                else:
                    traditional_predictions[comp].append(None)
        
        # Get BERT model predictions
        if use_bert and bert_model:
            result = parse_with_bert(address, bert_model, bert_tokenizer, label_decoders, components, device)
            for comp in components:
                if comp in result:
                    bert_predictions[comp].append(result[comp])
                else:
                    bert_predictions[comp].append(None)
    
    # Calculate metrics
    metrics = {
        'traditional': {},
        'bert': {} if use_bert and bert_model else None,
        'overall': {}
    }
    
    # Process traditional models
    trad_overall_correct = 0
    trad_total_components = 0
    
    for comp in components:
        if comp in traditional_models:
            # Convert None to 'None' string for sklearn metrics
            y_true = [str(val) if val is not None else 'None' for val in ground_truth[comp]]
            y_pred = [str(val) if val is not None else 'None' for val in traditional_predictions[comp]]
            
            # Calculate accuracy
            acc = accuracy_score(y_true, y_pred)
            
            # Calculate detailed metrics
            report = classification_report(y_true, y_pred, output_dict=True)
            
            metrics['traditional'][comp] = {
                'accuracy': acc,
                'report': report
            }
            
            # Count correct predictions
            correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
            trad_overall_correct += correct
            trad_total_components += len(y_true)
    
    if trad_total_components > 0:
        metrics['traditional']['overall'] = {
            'accuracy': trad_overall_correct / trad_total_components,
            'total_addresses': len(addresses),
            'total_components': trad_total_components,
            'correct_components': trad_overall_correct
        }
    
    # Process BERT model
    if use_bert and bert_model:
        bert_overall_correct = 0
        bert_total_components = 0
        
        for comp in components:
            # Convert None to 'None' string for sklearn metrics
            y_true = [str(val) if val is not None else 'None' for val in ground_truth[comp]]
            y_pred = [str(val) if val is not None else 'None' for val in bert_predictions[comp]]
            
            # Calculate accuracy
            acc = accuracy_score(y_true, y_pred)
            
            # Calculate detailed metrics
            report = classification_report(y_true, y_pred, output_dict=True)
            
            metrics['bert'][comp] = {
                'accuracy': acc,
                'report': report
            }
            
            # Count correct predictions
            correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
            bert_overall_correct += correct
            bert_total_components += len(y_true)
        
        if bert_total_components > 0:
            metrics['bert']['overall'] = {
                'accuracy': bert_overall_correct / bert_total_components,
                'total_addresses': len(addresses),
                'total_components': bert_total_components,
                'correct_components': bert_overall_correct
            }
    
    # Save visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save traditional model metrics
        if metrics['traditional']:
            # Accuracy bar chart
            plt.figure(figsize=(12, 6))
            accuracies = [metrics['traditional'][comp]['accuracy'] for comp in components if comp in metrics['traditional']]
            plt.bar(components, accuracies)
            plt.title('Traditional Models Accuracy by Component')
            plt.xlabel('Component')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, 'traditional_accuracy.png'))
            
            # Confusion matrices for each component
            for comp in components:
                if comp in metrics['traditional']:
                    plt.figure(figsize=(10, 8))
                    
                    # Convert to numerical for confusion matrix
                    y_true = [str(val) if val is not None else 'None' for val in ground_truth[comp]]
                    y_pred = [str(val) if val is not None else 'None' for val in traditional_predictions[comp]]
                    
                    # Get unique labels
                    labels = sorted(list(set(y_true) | set(y_pred)))
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    
                    # Plot
                    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
                    plt.title(f'Confusion Matrix for {comp}')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'traditional_cm_{comp}.png'))
        
        # Save BERT model metrics
        if metrics['bert']:
            # Accuracy bar chart
            plt.figure(figsize=(12, 6))
            accuracies = [metrics['bert'][comp]['accuracy'] for comp in components if comp in metrics['bert']]
            plt.bar(components, accuracies)
            plt.title('BERT Model Accuracy by Component')
            plt.xlabel('Component')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, 'bert_accuracy.png'))
            
            # Compare traditional and BERT if both are available
            if metrics['traditional']:
                plt.figure(figsize=(14, 7))
                trad_acc = [metrics['traditional'][comp]['accuracy'] if comp in metrics['traditional'] else 0 for comp in components]
                bert_acc = [metrics['bert'][comp]['accuracy'] if comp in metrics['bert'] else 0 for comp in components]
                
                x = np.arange(len(components))
                width = 0.35
                
                plt.bar(x - width/2, trad_acc, width, label='Traditional')
                plt.bar(x + width/2, bert_acc, width, label='BERT')
                
                plt.xlabel('Component')
                plt.ylabel('Accuracy')
                plt.title('Model Comparison: Traditional vs BERT')
                plt.xticks(x, components)
                plt.legend()
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics in a readable format"""
    print("\n=== EVALUATION RESULTS ===")
    
    # Traditional models
    if metrics['traditional']:
        print("\nTraditional Models:")
        overall = metrics['traditional']['overall']
        print(f"Overall Accuracy: {overall['accuracy'] * 100:.2f}%")
        print(f"Correct Components: {overall['correct_components']}/{overall['total_components']}")
        
        print("\nComponent-wise Accuracy:")
        for comp in [c for c in metrics['traditional'].keys() if c != 'overall']:
            print(f"  {comp.capitalize()}: {metrics['traditional'][comp]['accuracy'] * 100:.2f}%")
    
    # BERT model
    if metrics['bert']:
        print("\nBERT Model:")
        overall = metrics['bert']['overall']
        print(f"Overall Accuracy: {overall['accuracy'] * 100:.2f}%")
        print(f"Correct Components: {overall['correct_components']}/{overall['total_components']}")
        
        print("\nComponent-wise Accuracy:")
        for comp in [c for c in metrics['bert'].keys() if c != 'overall']:
            print(f"  {comp.capitalize()}: {metrics['bert'][comp]['accuracy'] * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Address Parser Models')
    parser.add_argument('--test-data', type=str, required=True,
                        help='CSV file with test data')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory with trained models')
    parser.add_argument('--use-bert', action='store_true',
                        help='Use BERT model if available')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load test data
    test_data = pd.read_csv(args.test_data)
    
    # Evaluate models
    metrics = evaluate_models(
        test_data, 
        args.model_dir, 
        args.use_bert,
        args.output_dir
    )
    
    # Print results
    print_metrics(metrics)

if __name__ == "__main__":
    main()