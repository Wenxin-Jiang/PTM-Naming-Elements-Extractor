# PTM-Naming-Elements-Extractor
This is a tool to extract naming elements from pre-trained model names using OpenAI API with structured output support.

## Features
- Extracts architecture, model size, dataset, and other components from model names
- Uses OpenAI's structured output API with Pydantic models
- Supports both newer models (with parse method) and older models (with JSON schema)
- Processes model names in batches for efficiency
- Includes retry logic with exponential backoff

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python extractor.py --csv_file data/HF_pkgs.csv --batch_size 50
```
