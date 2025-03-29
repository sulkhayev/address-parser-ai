#!/usr/bin/env python
# Example script to demonstrate the Address Parser API usage

import requests
import json
import argparse

def parse_address(address, api_url, use_bert=False):
    """
    Parse a single address using the API
    
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

def batch_parse(addresses, api_url, use_bert=False):
    """
    Parse multiple addresses in batch using the API
    
    Args:
        addresses: List of address strings
        api_url: Base URL of the API
        use_bert: Whether to use BERT model
        
    Returns:
        List of parsed addresses
    """
    endpoint = f"{api_url}/batch"
    payload = {
        "addresses": addresses,
        "use_bert": use_bert
    }
    
    response = requests.post(endpoint, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def print_parsed_address(parsed):
    """Pretty print parsed address components"""
    print("\nParsed Components:")
    print("-----------------")
    for component, value in parsed.items():
        print(f"{component.capitalize()}: {value or '-'}")

def print_batch_results(results):
    """Pretty print batch parsing results"""
    print("\nBatch Results:")
    print("-------------")
    for i, item in enumerate(results):
        print(f"\n{i+1}. Original: {item['original']}")
        for component, value in item['parsed'].items():
            print(f"   {component.capitalize()}: {value or '-'}")

def main():
    parser = argparse.ArgumentParser(description='Example Address Parser API Usage')
    parser.add_argument('--api-url', type=str, default='http://127.0.0.1:5000', 
                        help='API base URL')
    parser.add_argument('--address', type=str, 
                        help='Single address to parse')
    parser.add_argument('--input-file', type=str, 
                        help='Input file with addresses (one per line)')
    parser.add_argument('--output-file', type=str, 
                        help='Output JSON file for results')
    parser.add_argument('--use-bert', action='store_true',
                        help='Use BERT model if available')
    
    args = parser.parse_args()
    
    if args.address:
        # Parse a single address
        print(f"Parsing address: {args.address}")
        result = parse_address(args.address, args.api_url, args.use_bert)
        if result:
            print_parsed_address(result)
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\nResult saved to {args.output_file}")
    
    elif args.input_file:
        # Parse addresses from file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            addresses = [line.strip() for line in f if line.strip()]
        
        print(f"Parsing {len(addresses)} addresses from {args.input_file}")
        results = batch_parse(addresses, args.api_url, args.use_bert)
        
        if results:
            print_batch_results(results)
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nResults saved to {args.output_file}")
    
    else:
        print("Please provide either --address or --input-file")

if __name__ == "__main__":
    main()