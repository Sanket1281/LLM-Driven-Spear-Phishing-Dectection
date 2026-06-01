# LLM-Driven Spear Phishing Detection
Custom DeBERTa-inspired Transformer trained from scratch for 
3-class phishing email detection.

## Results
- Test Accuracy: 98.54%
- Spear-Phishing F1: 100.00%
- Parameters: 13.66M (trained from scratch)

## Setup
pip install torch transformers scikit-learn 
           seaborn matplotlib tqdm

## Run
python train.py              # full training
python evaluate.py           # evaluation + plots
python baselines.py          # classical ML comparison

## Dataset
- Benign: Enron Email Dataset
- Phishing: Kaggle Phishing Emails  
- Spear-Phishing: Synthetic (generated via Ollama)
