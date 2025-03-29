# Azerbaijani Address Parser

This project provides a machine learning-based system for parsing Azerbaijani addresses into structured components (city, region, settlement, street, building, and apartment).

## Overview

The system includes:

1. **Data Preparation**: Tools to create and maintain a labeled dataset of addresses
2. **Model Training**: Scripts to train traditional machine learning models and advanced BERT models
3. **Web Application**: A Flask-based web interface for address parsing
4. **API Access**: RESTful API endpoints for integration with other systems

## Project Structure

```
azerbaijani-address-parser/
│
├── data/
│   ├── raw_addresses.txt         # Raw address data
│   ├── address_initial.json      # Initial parsed data
│   ├── address_verified.json     # Manually verified data
│   └── training_data.csv         # Processed training data
│
├── models/                       # Trained model files
│   ├── city_model.pkl
│   ├── region_model.pkl
│   ├── settlement_model.pkl
│   ├── street_model.pkl
│   ├── building_model.pkl
│   ├── apartment_model.pkl
│   └── bert_model/               # BERT model files (optional)
│
├── templates/
│   └── index.html                # Web interface template
│
├── address_data_preparation.py   # Data preparation utilities
├── model_training.py             # Model training script
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/azerbaijani-address-parser.git
cd azerbaijani-address-parser
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Creation and Preparation

### Step 1: Prepare the initial dataset

Create a text file with raw addresses, one per line:

```bash
# Example raw_addresses.txt
9MKRŞ MİR CƏLAL KÜÇ, 9, M 12
ZIĞ QƏS. NİZAMİ KÜŞ EV 60
V.ƏLİYEV K241
...
```

### Step 2: Generate initial parsing

```bash
python address_data_preparation.py --input data/raw_addresses.txt --output data/address_initial.json
```

### Step 3: Manually verify the parsed components

```bash
python address_data_preparation.py --verify data/address_initial.json --output data/address_verified.json
```

This will start an interactive verification process where you can correct any parsing errors.

### Step 4: Convert to training format

```bash
python address_data_preparation.py --convert data/address_verified.json --output data/training_data.csv
```

## Model Training

### Train traditional ML models

```bash
python model_training.py --data data/training_data.csv --output models/ --no-bert
```

### Train BERT model (optional, requires GPU)

```bash
python model_training.py --data data/training_data.csv --output models/ --use-bert
```

## Running the Web Application

Start the Flask application:

```bash
python app.py --model_dir models/ --port 5000
```

With BERT model (if available):

```bash
python app.py --model_dir models/ --use_bert --port 5000
```

Then open your browser to http://127.0.0.1:5000 to access the web interface.

## API Usage

The application provides RESTful API endpoints for programmatic access:

### Parse a single address

```bash
curl -X POST http://127.0.0.1:5000/parse \
  -H "Content-Type: application/json" \
  -d '{"address":"R.BAĞIROV KÜÇ EV 28/1 XƏTAİ R"}'
```

Response:
```json
{
  "city": null,
  "region": "XƏTAİ",
  "settlement": null,
  "street": "R.BAĞIROV KÜÇ",
  "building": "28/1",
  "apartment": null
}
```

### Batch processing

```bash
curl -X POST http://127.0.0.1:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"addresses":["R.BAĞIROV KÜÇ EV 28/1 XƏTAİ R", "9MKRŞ MİR CƏLAL KÜÇ, 9, M 12"]}'
```

Response:
```json
[
  {
    "original": "R.BAĞIROV KÜÇ EV 28/1 XƏTAİ R",
    "parsed": {
      "city": null,
      "region": "XƏTAİ",
      "settlement": null,
      "street": "R.BAĞIROV KÜÇ",
      "building": "28/1",
      "apartment": null
    }
  },
  {
    "original": "9MKRŞ MİR CƏLAL KÜÇ, 9, M 12",
    "parsed": {
      "city": null,
      "region": "9MKR",
      "settlement": null,
      "street": "MİR CƏLAL KÜÇ",
      "building": "9",
      "apartment": "12"
    }
  }
]
```

## Performance Considerations

### Model Selection

- **Traditional ML Models**: Faster inference, suitable for most applications
- **BERT Model**: Higher accuracy but slower inference, recommended for batch processing or when accuracy is critical

### Improving Accuracy

1. **More Training Data**: Add more labeled examples, especially for uncommon address formats
2. **Feature Engineering**: Add domain-specific features for traditional models
3. **Model Tuning**: Experiment with different hyperparameters
4. **Post-processing Rules**: Add specific rules for known edge cases

## Evaluation

The model evaluation produces reports for each address component, showing:

- Precision, recall, and F1-score per component
- Overall accuracy
- Confusion matrices for detailed error analysis

## Troubleshooting

### Common Issues

1. **Incorrect Parsing**: If certain addresses are consistently parsed incorrectly, add them to the training data with correct labels
2. **Model Loading Errors**: Ensure all model files are present in the specified directory
3. **BERT Model Issues**: Check GPU availability and CUDA configuration if using BERT

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed by [Your Name]
- Uses [Transformers](https://github.com/huggingface/transformers) for BERT implementation
- Uses [scikit-learn](https://scikit-learn.org/) for traditional ML models