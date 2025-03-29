#!/usr/bin/env python
# Script to evaluate address parser accuracy against ground truth data

import requests
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def parse_address(address, api_url, use_bert=False):
    """
    Parse an address using the API
    
    Args:
        address: Address string to parse
        api_url: Base URL of the API
        use_bert: Whether to use BERT model
        
    Returns:
        Parsed address components
    """
    endpoint = f"{api_url}/parse"
    payload = {
        "address": address,
        "use_bert": use_bert
    }
    
    response = requests.post(endpoint, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def evaluate_accuracy(ground_truth_file, api_url, use_bert=False, output_file=None):
    """
    Evaluate model accuracy using a CSV file with ground truth data
    
    Args:
        ground_truth_file: CSV file with ground truth data
        api_url: Base URL of the API
        use_bert: Whether to use BERT model
        output_file: Optional file to save detailed results
    
    Returns:
        Dictionary with accuracy metrics
    """
    # Load ground truth data
    ground_truth = pd.read_csv(ground_truth_file)
    
    # Fill NaN values with None for consistency
    ground_truth = ground_truth.where(pd.notnull(ground_truth), None)
    
    # Components to evaluate
    components = ['city', 'region', 'settlement', 'street', 'building', 'apartment']
    
    # Store results
    all_results = []
    component_predictions = {comp: [] for comp in components}
    component_ground_truth = {comp: [] for comp in components}
    
    # Process each address
    for idx, row in ground_truth.iterrows():
        address = row['address']
        
        # Skip rows with missing address
        if pd.isna(address):
            continue
            
        print(f"Processing address {idx+1}/{len(ground_truth)}: {address}")
        
        # Get predictions from API
        prediction = parse_address(address, api_url, use_bert)
        
        if prediction:
            # Store result for detailed analysis
            result = {
                'address': address,
                'predictions': prediction,
                'ground_truth': {comp: row[comp] for comp in components},
                'correct': {}
            }
            
            # Check correctness for each component
            for comp in components:
                # Handle None and 'NONE' as equivalent
                pred_val = prediction[comp]
                true_val = row[comp]
                
                # Convert 'NONE' string to None for comparison
                if isinstance(pred_val, str) and pred_val.upper() == 'NONE':
                    pred_val = None
                if isinstance(true_val, str) and true_val.upper() == 'NONE':
                    true_val = None
                
                # Store for metrics calculation
                component_predictions[comp].append(pred_val)
                component_ground_truth[comp].append(true_val)
                
                # Check if correct
                is_correct = pred_val == true_val
                result['correct'][comp] = is_correct
            
            all_results.append(result)
    
    # Calculate metrics
    metrics = {}
    
    for comp in components:
        # Convert None to 'None' string for sklearn metrics
        y_true = [str(val) if val is not None else 'None' for val in component_ground_truth[comp]]
        y_pred = [str(val) if val is not None else 'None' for val in component_predictions[comp]]
        
        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Calculate detailed metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics[comp] = {
            'accuracy': acc,
            'report': report
        }
    
    # Calculate overall accuracy
    overall_correct = 0
    total_components = 0
    
    for result in all_results:
        for comp in components:
            if comp in result['correct']:
                total_components += 1
                if result['correct'][comp]:
                    overall_correct += 1
    
    overall_accuracy = overall_correct / total_components if total_components > 0 else 0
    
    metrics['overall'] = {
        'accuracy': overall_accuracy,
        'total_addresses': len(all_results),
        'total_components': total_components,
        'correct_components': overall_correct
    }
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'detailed_results': all_results
            }, f, ensure_ascii=False, indent=2)
    
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics in a readable format"""
    print("\n=== EVALUATION RESULTS ===")
    print(f"Overall Accuracy: {metrics['overall']['accuracy'] * 100:.2f}%")
    print(f"Total Addresses: {metrics['overall']['total_addresses']}")
    print(f"Correct Components: {metrics['overall']['correct_components']}/{metrics['overall']['total_components']}")
    
    print("\nComponent-wise Accuracy:")
    for comp in [c for c in metrics.keys() if c != 'overall']:
        print(f"  {comp.capitalize()}: {metrics[comp]['accuracy'] * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Address Parser Accuracy')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='CSV file with ground truth data')
    parser.add_argument('--api-url', type=str, default='http://127.0.0.1:5000',
                        help='API base URL')
    parser.add_argument('--use-bert', action='store_true',
                        help='Use BERT model if available')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_accuracy(
        args.ground_truth, 
        args.api_url, 
        args.use_bert,
        args.output
    )
    
    # Print results
    print_metrics(metrics)

if __name__ == "__main__":
    main()