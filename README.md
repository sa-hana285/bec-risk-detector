#Business Email Compromise (BEC) Risk Detection System

An end-to-end machine learning system to detect suspicious corporate emails using NLP-based feature engineering, Isolation Forest anomaly detection, and rule-based risk analysis.

#Features
- NLP-driven urgency and financial intent detection
- Role-based behavior mismatch analysis
- Isolation Forest for anomaly detection
- Explainable risk levels (LOW / MEDIUM / HIGH)
- FastAPI backend for real-time inference
- Web UI for interactive email risk analysis
#System Architecture
UI (HTML + JS)
↓
FastAPI Backend
↓
NLP Feature Extraction
↓
Isolation Forest Model
↓
Risk Level + Explanation

#Tech Stack
- Python
- FastAPI
- Scikit-learn
- NLTK
- Isolation Forest
- HTML / CSS / JavaScript
- Render (Deployment)

##Project Structure
bec_project/
│
├── app.py # FastAPI backend
├── preprocess.py # NLP & feature extraction
├── bec_model.pkl # Trained model
├── index.html # Web UI
├── requirements.txt
└── README.md
#How to Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
