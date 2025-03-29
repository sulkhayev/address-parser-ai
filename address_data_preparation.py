import pandas as pd
import re
import json
from tqdm import tqdm
import os
import argparse

class AddressDataPreparation:
    """
    Utility for preparing Azerbaijani address data for machine learning model training.
    This class helps with:
    1. Initial rule-based parsing to create a starting point for manual labeling
    2. Saving and loading partially labeled data
    3. Interactive labeling/correction interface
    """
    
    def __init__(self):
        # Patterns for rule-based extraction
        self.patterns = {
            'region': r'\b(XƏTAİ|NİZAMİ|NƏSİMİ|YASAMAL|SABUNÇU|BİNƏQƏDİ|SABİRABAD|AQSTAFA|NƏRİMANOV|SURAXANI|SƏBAİL|XƏZƏR|ABŞERON|FÜZÜLİ|UCAR|BƏRDƏ|MASALLİ|YARDIMLI|GÖYÇAY|ŞABRAN|NAX|NEFTCALA)\b(?:\s+R(?:AY)?\.?)?',
            'settlement': r'\b([A-ZƏİĞŞÇÖÜ\s]+)\s+QƏS(?:\.|-|\b)',
            'street': r'\b([A-ZƏİĞŞÇÖÜ\.\s]+(?:KÜ[CÇ]|PR|K|KÜŞƏ?)\.?)\b',
            'building': r'\b(?:EV|BİNA)\s+([0-9]+(?:\/[0-9]+)?(?:[A-Z\"]+)?)',
            'apartment': r'\bM(?:ƏN)?\.?\s+([0-9]+(?:\/[0-9]+)?(?:[A-Za-z]+)?)'
        }
        
        self.components = ['city', 'region', 'settlement', 'street', 'building', 'apartment']
        
        # Known cities in Azerbaijan
        self.cities = ['BAKI', 'SUMQAYIT', 'GƏNCƏ', 'MİNGƏÇEVİR', 'NAFTALAN', 'ŞUŞA',
                      'NAXÇIVAN', 'LƏNKƏRAN', 'ŞƏKİ', 'YEVLAX', 'ŞIRVAN']
        
        # Known regions
        self.regions = ['XƏTAİ', 'NİZAMİ', 'NƏSİMİ', 'YASAMAL', 'SABUNÇU', 'BİNƏQƏDİ', 'SABİRABAD','AQSTAFA',
                       'NƏRİMANOV', 'SURAXANI', 'SƏBAİL', 'XƏZƏR', 'ABŞERON', 'FÜZÜLİ',
                       'UCAR', 'BƏRDƏ', 'MASALLİ', 'YARDIMLI', 'GÖYÇAY', 'ŞABRAN', 'NAX', 'NEFTCALA']
        
        # Special street indicators
        self.street_indicators = ['KÜÇ', 'PR', 'K', 'KÜŞƏ', 'KÇ']
    
    def preprocess_address(self, address):
        """Clean and normalize address string"""
        # Convert to uppercase
        address = address.upper()
        
        # Normalize spaces
        address = re.sub(r'\s+', ' ', address)
        
        return address.strip()
        
    def initial_parse(self, address):
        """Rule-based parsing for initial labeling"""
        address = self.preprocess_address(address)
        result = {comp: None for comp in self.components}
        
        # Try to identify city
        for city in self.cities:
            if re.search(r'\b' + re.escape(city) + r'\b', address):
                result['city'] = city
                break
        
        # Apply patterns
        for component, pattern in self.patterns.items():
            match = re.search(pattern, address, re.IGNORECASE)
            if match:
                # Get the capture group or full match
                value = match.group(1) if match.groups() else match.group(0)
                # Clean up values
                if component == 'region':
                    value = re.sub(r'\s+R(?:AY)?\.?$', '', value)
                elif component == 'settlement':
                    value = re.sub(r'\s+QƏS(?:\.|-|\b)$', '', value)
                
                result[component] = value.strip()
        
        # Special case for MKR (microdistricts)
        mkr_match = re.search(r'(\d+)\s*MKR', address)
        if mkr_match:
            result['region'] = mkr_match.group(0).strip()
            
        # Special case for direct building numbers without EV
        if not result['building']:
            # Try to find standalone numbers that might be building numbers
            building_match = re.search(r'\b(\d+(?:\/\d+)?(?:[A-Z]+)?)\b', address)
            if building_match:
                result['building'] = building_match.group(1)
                
        return result
    
    def create_initial_dataset(self, addresses, output_file='address_initial.json'):
        """
        Create initial dataset with rule-based parsing for manual correction
        
        Args:
            addresses: List of address strings
            output_file: File to save the initial parsing results
        """
        data = []
        
        for addr in tqdm(addresses, desc="Initial parsing"):
            parsed = self.initial_parse(addr)
            data.append({
                'original': addr,
                'processed': self.preprocess_address(addr),
                'components': parsed,
                'verified': False  # Needs manual verification
            })
        
        # Save to JSON for manual correction
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Initial dataset saved to {output_file}")
        return data
    
    def manual_correction_interface(self, dataset_file, output_file='address_verified.json'):
        """
        Interactive interface for manually correcting the parsed addresses
        
        Args:
            dataset_file: JSON file with initial parsed data
            output_file: File to save the corrected data
        """
        # Load the dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        verified_count = sum(1 for item in data if item['verified'])
        print(f"Dataset has {len(data)} addresses, {verified_count} already verified.")
        
        try:
            for i, item in enumerate(data):
                if item['verified']:
                    continue
                
                print(f"\nAddress {i+1}/{len(data)}: {item['original']}")
                print("Current parsing:")
                for comp, value in item['components'].items():
                    print(f"  {comp}: {value}")
                
                print("\nCorrections (leave empty to keep current value):")
                for comp in self.components:
                    new_value = input(f"  {comp} [{item['components'][comp] or ''}]: ")
                    if new_value:
                        item['components'][comp] = new_value
                
                item['verified'] = True
                
                # Save progress after each verification
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # Option to exit
                if i < len(data) - 1:
                    cont = input("\nContinue to next address? (y/n): ")
                    if cont.lower() != 'y':
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
        
        verified_count = sum(1 for item in data if item['verified'])
        print(f"Verification completed for {verified_count}/{len(data)} addresses.")
        print(f"Dataset saved to {output_file}")
    
    def convert_to_training_format(self, dataset_file, output_file='training_data.csv'):
        """
        Convert the verified dataset to a format suitable for ML training
        
        Args:
            dataset_file: JSON file with verified data
            output_file: CSV file for the training data
        """
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Keep only verified items
        verified_data = [item for item in data if item['verified']]
        
        # Convert to dataframe
        rows = []
        for item in verified_data:
            row = {'address': item['processed']}
            row.update(item['components'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Training data saved to {output_file}")
        return df

def main():
    parser = argparse.ArgumentParser(description='Address data preparation utilities')
    parser.add_argument('--input', type=str, help='Input file with raw addresses')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--verify', type=str, help='Verify and correct address file')
    parser.add_argument('--convert', type=str, help='Convert verified data to training format')
    
    args = parser.parse_args()
    
    prep = AddressDataPreparation()
    
    if args.input and args.output:
        # Read raw addresses
        with open(args.input, 'r', encoding='utf-8') as f:
            addresses = [line.strip() for line in f if line.strip()]
        
        # Create initial dataset
        prep.create_initial_dataset(addresses, args.output)
    
    if args.verify and args.output:
        # Manually verify and correct
        prep.manual_correction_interface(args.verify, args.output)
    
    if args.convert and args.output:
        # Convert to training format
        prep.convert_to_training_format(args.convert, args.output)
    
    if not (args.input or args.verify or args.convert):
        # Example demo if no arguments provided
        test_address = "R.BAĞIROV KÜÇ EV 28/1 XƏTAİ R"
        parsed = prep.initial_parse(test_address)
        
        print(f"\nExample parsing for: {test_address}")
        for component, value in parsed.items():
            print(f"{component}: {value or 'None'}")

if __name__ == "__main__":
    main()