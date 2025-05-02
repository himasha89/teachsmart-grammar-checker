import os
import json
import logging
import requests
import uuid
from typing import Iterator, Any, Dict, Optional, Union
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Firebase only if credentials are available
firebase_initialized = False
try:
    import firebase_admin
    from firebase_admin import firestore
    firebase_admin.initialize_app()
    firebase_initialized = True
    logging.info("Firebase initialized successfully")
except Exception as e:
    logging.warning(f"Firebase initialization skipped: {str(e)}")

# Initialize Hugging Face client
from huggingface_hub import InferenceClient
client = InferenceClient(token=os.environ.get('HUGGINGFACE_API_KEY'))

# Initialize Flask app
app = Flask(__name__)

# Define constants
ALLOWED_ORIGINS = ["http://localhost:3000", "https://teachsmart.vercel.app"]
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600'
}

def extract_grammar_issues(text: str, corrected_text: str) -> list:
    """Extract grammar issues from original and corrected text."""
    issues = []
    
    # Simple difference detection (this is a basic approach, can be improved)
    if text == corrected_text:
        return issues
    
    # Find words that differ between original and corrected
    original_words = text.split()
    corrected_words = corrected_text.split()
    
    # Use difflib or a similar algorithm for better diff detection
    # This is a simplified approach for demonstration
    i = 0
    j = 0
    while i < len(original_words) and j < len(corrected_words):
        if original_words[i] != corrected_words[j]:
            # Found a difference
            
            # Determine the type of issue (simplified version)
            issue_type = "grammar"
            if original_words[i].lower() == corrected_words[j].lower():
                issue_type = "punctuation"
            elif any(c.isalpha() for c in original_words[i]) and any(c.isalpha() for c in corrected_words[j]):
                if len(original_words[i]) == len(corrected_words[j]) + 1 or len(original_words[i]) == len(corrected_words[j]) - 1:
                    issue_type = "spelling"
                    
            # Calculate approximate positions in the original text
            start_pos = text.find(original_words[i])
            end_pos = start_pos + len(original_words[i])
            
            issue = {
                "original": original_words[i],
                "suggestion": corrected_words[j],
                "type": issue_type,
                "explanation": f"'{original_words[i]}' should be '{corrected_words[j]}'",
                "startIndex": start_pos,
                "endIndex": end_pos
            }
            issues.append(issue)
            
            # Skip to next word in both texts
            i += 1
            j += 1
        else:
            # Words match, move to next pair
            i += 1
            j += 1
    
    return issues


def calculate_grammar_score(text: str, issues: list) -> int:
    """Calculate a grammar score based on the number and severity of issues."""
    if not text:
        return 0
    
    # Base score
    score = 100
    
    # Word count
    word_count = len(text.split())
    
    # Deduct points for each issue
    for issue in issues:
        if issue["type"] == "grammar":
            score -= 5
        elif issue["type"] == "spelling":
            score -= 3
        elif issue["type"] == "punctuation":
            score -= 2
        else:
            score -= 1
    
    # Adjust score based on text length
    if word_count > 0:
        issues_per_100_words = len(issues) * 100 / word_count
        if issues_per_100_words > 10:
            score -= 10
        elif issues_per_100_words > 5:
            score -= 5
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))


@app.route('/check_grammar', methods=['POST', 'OPTIONS'])
def check_grammar():
    """Cloud Run Function to check grammar using Hugging Face Inference API."""
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = app.response_class(
            response="",
            status=204
        )
        for key, value in CORS_HEADERS.items():
            response.headers[key] = value
        return response

    # Set headers for CORS
    headers = CORS_HEADERS.copy()
    headers['Content-Type'] = 'application/json'
    
    try:
        # Parse request data
        request_data = request.get_json().get("data", {})
        
        if not request_data or 'text' not in request_data:
            return jsonify({
                'error': 'No text provided'
            }), 400, headers

        text = request_data['text']
        logging.info(f"Processing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Use a smaller model for faster responses
        MODEL_NAME = "textattack/roberta-base-CoLA"  # Smaller model for grammar checking
        
        try:
            # First try with the lighter classification model
            api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
            headers_req = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}
            payload = {"inputs": text}
            
            # Add a timeout to prevent hanging
            logging.info(f"Trying classification model: {MODEL_NAME}")
            response = requests.post(api_url, headers=headers_req, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Classification result: {result}")
                
                # This model only checks if the text is grammatical, it doesn't correct it
                # If result shows text is grammatical, return it as is
                if isinstance(result, list) and result[0][0]["label"] == "LABEL_1" and result[0][0]["score"] > 0.9:
                    corrected_text = text
                    logging.info("Text is grammatically correct according to classification model")
                else:
                    # Text has grammar issues, try the correction model
                    raise Exception("Grammar issues detected, switching to correction model")
            else:
                # If classification fails, fall back to correction model
                raise Exception(f"Classification API error: {response.status_code}")
                
        except Exception as e:
            logging.info(f"Using full correction model: {str(e)}")
            # Fall back to the correction model with multiple retry attempts
            try:
                # Try with the T5 correction model
                correction_model = "Unbabel/gec-t5_small"
                api_url = f"https://api-inference.huggingface.co/models/{correction_model}"
                headers_req = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}
                payload = {"inputs": text, "parameters": {"max_new_tokens": 250}}
                
                # Add a timeout to prevent hanging
                logging.info(f"Trying correction model: {correction_model}")
                response = requests.post(api_url, headers=headers_req, json=payload, timeout=30)
                
                # Handle potential errors
                if response.status_code != 200:
                    logging.error(f"Hugging Face API error: {response.status_code}, {response.text}")
                    corrected_text = text  # Fall back to original text
                else:
                    result = response.json()
                    logging.info(f"Correction result: {result}")
                    # Extract the corrected text from the result
                    corrected_text = result[0]["generated_text"] if isinstance(result, list) else result
                    logging.info(f"Corrected text: {corrected_text[:50]}{'...' if len(corrected_text) > 50 else ''}")
            except Exception as e:
                logging.error(f"All correction attempts failed: {str(e)}")
                corrected_text = text  # Fall back to original text
        
        # Extract grammar issues
        issues = extract_grammar_issues(text, corrected_text)
        logging.info(f"Found {len(issues)} grammar issues")
        
        # Calculate grammar score
        score = calculate_grammar_score(text, issues)
        logging.info(f"Grammar score: {score}")
        
        # Prepare response
        response_data = {
            "correctedText": corrected_text,
            "issues": issues,
            "score": score
        }
        
        # Store result in Firestore if initialized
        doc_id = str(uuid.uuid4())
        if firebase_initialized:
            try:
                db = firestore.client()
                doc_ref = db.collection('grammar_check_results').add({
                    'original_text': text,
                    'corrected_text': corrected_text,
                    'issues': issues,
                    'score': score,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                doc_id = doc_ref[1].id
                logging.info(f"Stored result in Firestore with ID: {doc_id}")
            except Exception as e:
                logging.warning(f"Failed to store in Firestore: {str(e)}")

        response = jsonify({
            'id': doc_id,
            'result': response_data
        })
        
        # Add CORS headers to response
        for key, value in headers.items():
            response.headers[key] = value
            
        return response

    except Exception as e:
        logging.error(f"An error occurred in check_grammar: {str(e)}")
        error_response = jsonify({
            'error': f'Grammar checking failed: {str(e)}'
        })
        
        # Add CORS headers to error response
        for key, value in headers.items():
            error_response.headers[key] = value
            
        return error_response, 500


# Main entry point for Cloud Run
if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    # Run the Flask application
    app.run(host="0.0.0.0", port=port, debug=True)