# TeachSmart Grammar Checker

A cloud-based grammar checking service built with Python, Flask, Hugging Face ML models, and Firebase. The service analyzes text for grammar issues, provides corrections, and calculates a grammar score.

## Features

- Grammar error detection and correction
- Detailed breakdown of grammar, spelling, and punctuation issues
- Grammar scoring based on error frequency and severity
- Cloud-ready with Firebase integration for result storage
- CORS support for cross-origin requests
- Fallback strategies for reliable operation

## Technology Stack

- **Backend**: Python 3.9+, Flask
- **ML Models**: Hugging Face Inference API
- **Database**: Firebase Firestore
- **Deployment**: Google Cloud Run
- **Container**: Docker

## Local Development Setup

### Prerequisites

- Python 3.9 or higher
- Pip package manager
- A Hugging Face API key
- Firebase project (optional for local testing)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/himasha89/grammar-check-service.git
   cd grammar-check-service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Windows
   set HUGGINGFACE_API_KEY=your_huggingface_api_key
   set GOOGLE_APPLICATION_CREDENTIALS=path/to/firebase-credentials.json
   
   # macOS/Linux
   export HUGGINGFACE_API_KEY=your_huggingface_api_key
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/firebase-credentials.json
   ```

5. Run the application:
   ```bash
   python main.py
   ```

The server will be available at `http://localhost:8080`.

## API Usage

### Check Grammar Endpoint

**Endpoint**: `POST /check_grammar`

**Request Format**:
```json
{
  "data": {
    "text": "Your text to check goes here."
  }
}
```

**Response Format**:
```json
{
  "id": "unique-document-id",
  "result": {
    "correctedText": "Your corrected text goes here.",
    "issues": [
      {
        "original": "original word",
        "suggestion": "suggested correction",
        "type": "grammar/spelling/punctuation",
        "explanation": "Explanation of the issue",
        "startIndex": 0,
        "endIndex": 5
      }
    ],
    "score": 85
  }
}
```

## Google Cloud Run Deployment

### Prerequisites

- Google Cloud account
- Google Cloud SDK installed
- Docker installed (for local builds)
- Firebase project with Firestore enabled

### Deployment Steps

1. Create a Dockerfile:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY main.py .
   COPY firebase-credentials.json .
   
   ENV GOOGLE_APPLICATION_CREDENTIALS=firebase-credentials.json
   
   CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
   ```

2. Enable required Google Cloud APIs:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API or Artifact Registry API
   - Firestore API

3. Build and deploy:
   ```bash
   # Set your project ID
   PROJECT_ID=your-project-id
   
   # Build the container image
   gcloud builds submit --tag gcr.io/$PROJECT_ID/grammar-check-service
   
   # Deploy to Cloud Run
   gcloud run deploy grammar-check-service \
     --image gcr.io/$PROJECT_ID/grammar-check-service \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars HUGGINGFACE_API_KEY=your-huggingface-api-key
   ```

## Firestore Integration

The service stores grammar check results in Firestore for future reference. Each check creates a document with:

- Original text
- Corrected text
- List of identified issues
- Grammar score
- Timestamp

## Machine Learning Models

The service uses two Hugging Face models:
1. **textattack/roberta-base-CoLA**: A lightweight model for initial grammatical assessment
2. **Unbabel/gec-t5_small**: A more comprehensive model for grammar correction

The system tries the lighter model first and falls back to the more powerful model when needed.

---

Made with ❤️ for TeachSmart
