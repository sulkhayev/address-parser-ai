import os
import pandas as pd
import joblib
import pickle
import re
from flask import Flask, request, jsonify, render_template
import argparse

# Optional BERT imports - only used if use_bert=True
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


class AddressParser:
    """Address parser that uses trained models"""
    def __init__(self, model_dir, use_bert=False):
        self.model_dir = model_dir
        self.use_bert = use_bert and BERT_AVAILABLE
        self.components = ['city', 'region', 'settlement', 'street', 'building', 'apartment']
        self.models = {}
        self.bert_model = None
        self.bert_tokenizer = None
        self.label_decoders = None
        self.device = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load traditional and BERT models"""
        # Load traditional models
        for component in self.components:
            model_path = os.path.join(self.model_dir, f"{component}_model.pkl")
            if os.path.exists(model_path):
                self.models[component] = joblib.load(model_path)
                print(f"Loaded model for {component}")
            else:
                print(f"Warning: Model for {component} not found at {model_path}")
        
        # Load BERT model if enabled
        if self.use_bert:
            bert_dir = os.path.join(self.model_dir, "bert_model")
            model_path = os.path.join(bert_dir, "model.pt")
            
            if os.path.exists(model_path):
                # Set device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {self.device}")
                
                # Load tokenizer
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
                
                # Load label decoders
                with open(os.path.join(bert_dir, "label_decoders.pkl"), "rb") as f:
                    self.label_decoders = pickle.load(f)
                
                # Load label encoders (needed for model initialization)
                with open(os.path.join(bert_dir, "label_encoders.pkl"), "rb") as f:
                    label_encoders = pickle.load(f)
                
                # Initialize model
                num_labels_per_component = [len(encoders) for encoders in label_encoders.values()]
                self.bert_model = BertAddressParser(num_labels_per_component).to(self.device)
                
                # Load weights
                self.bert_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.bert_model.eval()
                print("Loaded BERT model")
            else:
                print(f"Warning: BERT model not found at {model_path}")
    
    def parse(self, address, use_bert=None):
        """
        Parse an address using trained models
        
        Args:
            address: Raw address string
            use_bert: Whether to use BERT model (overrides instance setting)
            
        Returns:
            Dictionary with parsed components
        """
        # Default to instance setting if not specified
        if use_bert is None:
            use_bert = self.use_bert
        
        # Preprocess address
        processed_address = address.upper()
        processed_address = re.sub(r'\s+', ' ', processed_address).strip()
        
        result = {comp: None for comp in self.components}
        
        if use_bert and self.bert_model and self.bert_tokenizer:
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
            
            with torch.no_grad():
                logits = self.bert_model(input_ids, attention_mask)
                
                for i, (logit, component) in enumerate(zip(logits, self.components)):
                    prediction = torch.argmax(logit, dim=1).item()
                    result[component] = self.label_decoders[component].get(prediction, None)
        else:
            # Use traditional models
            for component, model in self.models.items():
                try:
                    prediction = model.predict([processed_address])[0]
                    # Convert "NONE" back to None
                    result[component] = None if prediction == "NONE" else prediction
                except Exception as e:
                    print(f"Error predicting {component}: {e}")
        
        return result
    
    def estimate_confidence(self, address, parsed_result):
        """
        Estimate confidence of parsing without ground truth
        
        Args:
            address: Original address string
            parsed_result: Dictionary with parsed components
            
        Returns:
            Dictionary with confidence estimates
        """
        confidence = {
            'components': {},
            'overall': 0.0
        }
        
        # Preprocess address for analysis
        processed_address = address.upper()
        processed_address = re.sub(r'\s+', ' ', processed_address).strip()
        
        # Component-specific confidence rules
        
        # 1. Settlement confidence
        if parsed_result['settlement']:
            # Check if settlement followed by QƏS
            settlement_pattern = rf"{re.escape(parsed_result['settlement'])}\s+QƏS"
            if re.search(settlement_pattern, processed_address):
                confidence['components']['settlement'] = 0.95
            else:
                confidence['components']['settlement'] = 0.7
        else:
            # If no settlement found, check if there's a QƏS in the address
            if 'QƏS' in processed_address:
                confidence['components']['settlement'] = 0.3
            else:
                confidence['components']['settlement'] = 0.9  # High confidence in null settlement
        
        # 2. Street confidence
        if parsed_result['street']:
            # Check for street indicators
            street_indicators = ['KÜÇ', 'PR', 'K', 'KÜŞƏ', 'KÜÇƏ', 'KÇ', 'DÖN', 'KÜÇƏSI', 'DÖNGƏ', 'KÜÇƏSİ']
            has_indicator = any(indicator in parsed_result['street'] for indicator in street_indicators)
            
            if has_indicator:
                confidence['components']['street'] = 0.9
            else:
                confidence['components']['street'] = 0.6
        else:
            confidence['components']['street'] = 0.4  # Low confidence in null street
        
        # 3. Building confidence
        if parsed_result['building']:
            # Check for building number indicators
            building_pattern = r'EV\s+' + re.escape(str(parsed_result['building']))
            if re.search(building_pattern, processed_address):
                confidence['components']['building'] = 0.95
            else:
                confidence['components']['building'] = 0.7
        else:
            confidence['components']['building'] = 0.4  # Low confidence in null building
        
        # 4. Apartment confidence
        if parsed_result['apartment']:
            # Check for apartment indicators
            apartment_pattern = r'M(?:ƏN)?\s+' + re.escape(str(parsed_result['apartment']))
            if re.search(apartment_pattern, processed_address):
                confidence['components']['apartment'] = 0.95
            else:
                confidence['components']['apartment'] = 0.6
        else:
            # If no apartment but there's an M or MƏN in the address
            if re.search(r'M(?:ƏN)?\s+\d+', processed_address):
                confidence['components']['apartment'] = 0.3
            else:
                confidence['components']['apartment'] = 0.85  # High confidence in null apartment
        
        # 5. Region confidence
        if parsed_result['region']:
            # Check for common regions
            common_regions = ['BAKI','GƏNCƏ','SUMQAYIT','MİNGƏÇEVİR','NAXÇIVAN','ŞƏKİ','YEVLAX','ŞİRVAN','LƏNKƏRAN','ŞƏMKİR','XANKƏNDİ','NAFTALAN','QƏBƏLƏ','ŞAMAXI','QUBA','ZAQATALA','GÖYÇAY','TOVUZ','BALAKƏN','AĞDAM','AĞDAŞ','AĞSTAFA','ASTARA','BƏRDƏ','BEYLƏQAN','BİLƏSUVAR','CƏLİLABAD','DAŞKƏSƏN','ƏLİ BAYRAMLI','FÜZULİ','GORANBOY','GÖYGÖL','HACIQABUL','İMİŞLİ','İSMAYILLI','KÜRDƏMİR','LERİK','MASALLI','NEFTÇALA','OĞUZ','QAX','QAZAX','QOBUSTAN','QUSAR','SAATLI','SABİRABAD','SALYAN','SİYƏZƏN','TƏRTƏR','UCAR','XAÇMAZ','XIZI','XOCALI','YARDIMLI']
        
            if parsed_result['region'] in common_regions:
                confidence['components']['region'] = 0.9
            else:
                confidence['components']['region'] = 0.7
                
            # Check for R indicator
            region_pattern = rf"{re.escape(parsed_result['region'])}\s+R"
            if re.search(region_pattern, processed_address):
                confidence['components']['region'] = 0.95
        else:
            # If no region found, check if there's an R in the address
            if ' R' in processed_address:
                confidence['components']['region'] = 0.3
            else:
                confidence['components']['region'] = 0.8  # Moderate confidence in null region
        
        # 6. City confidence
        if parsed_result['city']:
            # Check for common cities
            common_cities = ['BAKI', 'SUMQAYIT', 'GƏNCƏ', 'MİNGƏÇEVİR', 'ŞUŞA', 'NAXÇIVAN']
            if parsed_result['city'] in common_cities:
                confidence['components']['city'] = 0.9
            else:
                confidence['components']['city'] = 0.7
        else:
            confidence['components']['city'] = 0.8  # Moderate confidence in null city
        
        # Calculate overall confidence as average
        confidence_values = list(confidence['components'].values())
        confidence['overall'] = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        return confidence
    
    def apply_post_processing(self, address, parsed_result):
        """
        Apply post-processing rules to improve parsing
        
        Args:
            address: Original address string
            parsed_result: Initial parsing result
            
        Returns:
            Improved parsing result
        """
        # Create a copy to avoid modifying the original
        result = parsed_result.copy()
        
        # Preprocess address for analysis
        processed_address = address.upper()
        processed_address = re.sub(r'\s+', ' ', processed_address).strip()
        
        # Fix settlement and street issues
        if result['settlement'] and 'QƏS' in processed_address:
            # Extract settlement more accurately
            settlement_match = re.search(r'([A-ZƏİĞŞÇÖÜ]+)\s+QƏS', processed_address)
            if settlement_match:
                result['settlement'] = settlement_match.group(1)
                
                # Fix street - remove settlement from street
                if result['street'] and result['street'].startswith(result['settlement']):
                    # Remove settlement and QƏS from the street
                    street_pattern = r'([A-ZƏİĞŞÇÖÜ\.]+\s+(?:KÜ[CÇ]|PR|K|KÜŞƏ?))'
                    street_match = re.search(street_pattern, processed_address.replace(settlement_match.group(0), ''))
                    if street_match:
                        result['street'] = street_match.group(1).strip()
        
        # Identify and correct common street format issues
        if result['street']:
            # Remove settlement from street if still present
            if result['settlement'] and result['street'].startswith(f"{result['settlement']} QƏS"):
                result['street'] = result['street'].replace(f"{result['settlement']} QƏS", "").strip()
            
            # Handle cases where street contains comma-separated building number
            building_in_street = re.search(r'(.*),\s*(\d+)', result['street'])
            if building_in_street and not result['building']:
                result['street'] = building_in_street.group(1).strip()
                result['building'] = building_in_street.group(2)
        
        return result
    
    def parse_improved(self, address, use_bert=None):
        """
        Parse an address with improved street handling
        
        Args:
            address: Raw address string
            use_bert: Whether to use BERT model
            
        Returns:
            Dictionary with parsed components and confidence
        """
        # Get initial parsing
        result = self.parse(address, use_bert=use_bert)
        
        # Apply post-processing rules
        result = self.apply_post_processing(address, result)
        
        # Estimate confidence
        confidence = self.estimate_confidence(address, result)
        
        return {
            'parsed': result,
            'confidence': confidence
        }


# Create Flask app with explicit template folder path
app = Flask(__name__, 
          template_folder=os.path.abspath('templates'),
          static_folder=os.path.abspath('static'))

# Debug output for template location
print(f"Current working directory: {os.getcwd()}")
print(f"Template folder: {app.template_folder}")
print(f"Template folder exists: {os.path.exists(app.template_folder)}")
if os.path.exists(app.template_folder):
    print(f"Files in template folder: {os.listdir(app.template_folder)}")

# Initialize address parser
address_parser = None

@app.route('/')
def home():
    """Render the home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return f"Error rendering template: {e}"

@app.route('/parse', methods=['POST'])
def parse_address():
    """Parse an address and return JSON result"""
    data = request.get_json()
    address = data.get('address', '')
    use_bert = data.get('use_bert', None)
    
    if not address:
        return jsonify({'error': 'No address provided'}), 400
    
    try:
        # Parse the address
        result = address_parser.parse(address, use_bert=use_bert)
        return jsonify(result)
    except Exception as e:
        print(f"Parse error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_parse():
    """Parse multiple addresses and return JSON results"""
    data = request.get_json()
    addresses = data.get('addresses', [])
    use_bert = data.get('use_bert', None)
    
    if not addresses:
        return jsonify({'error': 'No addresses provided'}), 400
    
    try:
        # Parse each address
        results = []
        for address in addresses:
            result = address_parser.parse(address, use_bert=use_bert)
            results.append({
                'original': address,
                'parsed': result
            })
        return jsonify(results)
    except Exception as e:
        print(f"Batch parse error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/parse', methods=['POST'])
def api_parse():
    """API endpoint for parsing addresses"""
    data = request.get_json()
    addresses = data.get('addresses', [])
    use_bert = data.get('use_bert', None)
    
    if not addresses:
        return jsonify({'error': 'No addresses provided'}), 400
    
    try:
        # Process addresses (single or batch)
        if isinstance(addresses, str):
            # Single address
            result = address_parser.parse(addresses, use_bert=use_bert)
            return jsonify({
                'status': 'success',
                'data': result
            })
        else:
            # Batch of addresses
            results = []
            for address in addresses:
                result = address_parser.parse(address, use_bert=use_bert)
                results.append({
                    'original': address,
                    'parsed': result
                })
            return jsonify({
                'status': 'success',
                'data': results
            })
    except Exception as e:
        print(f"API parse error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/parse_with_confidence', methods=['POST'])
def parse_with_confidence():
    """Parse an address and return result with confidence scores"""
    data = request.get_json()
    address = data.get('address', '')
    ground_truth = data.get('ground_truth', {})
    use_bert = data.get('use_bert', None)
    
    if not address:
        return jsonify({'error': 'No address provided'}), 400
    
    try:
        # Parse the address
        result = address_parser.parse(address, use_bert=use_bert)
        
        # Calculate confidence scores if ground truth is provided
        confidence = {}
        
        if ground_truth:
            # Calculate exact match confidence
            component_confidence = {}
            correct_components = 0
            total_components = 0
            
            for component in result.keys():
                if component in ground_truth:
                    total_components += 1
                    
                    # Handle None values
                    pred_val = result[component]
                    true_val = ground_truth[component]
                    
                    # Normalize None and "NONE" to be equivalent
                    if isinstance(pred_val, str) and pred_val.upper() == "NONE":
                        pred_val = None
                    if isinstance(true_val, str) and true_val.upper() == "NONE":
                        true_val = None
                    
                    # Check match
                    is_correct = pred_val == true_val
                    component_confidence[component] = 1.0 if is_correct else 0.0
                    
                    if is_correct:
                        correct_components += 1
            
            # Overall confidence
            overall_confidence = correct_components / total_components if total_components > 0 else 0.0
            
            confidence = {
                'components': component_confidence,
                'overall': overall_confidence
            }
        else:
            # Estimate confidence without ground truth
            confidence = address_parser.estimate_confidence(address, result)
        
        return jsonify({
            'parsed': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/parse_with_estimated_confidence', methods=['POST'])
def parse_with_estimated_confidence():
    """Parse an address and return result with estimated confidence scores"""
    data = request.get_json()
    address = data.get('address', '')
    use_bert = data.get('use_bert', None)
    
    if not address:
        return jsonify({'error': 'No address provided'}), 400
    
    try:
        # Parse the address
        result = address_parser.parse(address, use_bert=use_bert)
        
        # Estimate confidence
        confidence = address_parser.estimate_confidence(address, result)
        
        return jsonify({
            'parsed': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/parse_improved', methods=['POST'])
def parse_improved_endpoint():
    """Parse an address with improved street handling and confidence metrics"""
    data = request.get_json()
    address = data.get('address', '')
    use_bert = data.get('use_bert', None)
    
    if not address:
        return jsonify({'error': 'No address provided'}), 400
    
    try:
        # Parse the address with improvements
        result = address_parser.parse_improved(address, use_bert=use_bert)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add test route for troubleshooting
@app.route('/test')
def test():
    """Test route to verify template rendering"""
    return '''
    <html>
        <head><title>Address Parser Test</title></head>
        <body>
            <h1>Address Parser Test Page</h1>
            <p>If you see this, the Flask server is working correctly.</p>
            <p>Try to access the <a href="/">main page</a>.</p>
        </body>
    </html>
    '''

def create_app(model_dir='address_models', use_bert=False):
    """Create and configure the Flask application"""
    global address_parser
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Warning: Model directory {model_dir} did not exist and was created")
    
    address_parser = AddressParser(model_dir, use_bert)
    return app

if __name__ == '__main__':
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser(description='Run the address parser web application')
    arg_parser.add_argument('--model_dir', type=str, default='address_models',
                       help='Directory with trained models')
    arg_parser.add_argument('--use_bert', action='store_true',
                       help='Use BERT model if available')
    arg_parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web application on')
    arg_parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the web application on')
    arg_parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = arg_parser.parse_args()
    
    if args.use_bert and not BERT_AVAILABLE:
        print("Warning: BERT dependencies not available. Falling back to traditional models.")
        args.use_bert = False
    
    # Create and run app
    app = create_app(args.model_dir, args.use_bert)
    app.run(debug=args.debug, port=args.port, host=args.host)