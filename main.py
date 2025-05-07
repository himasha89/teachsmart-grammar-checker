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
    # Try to initialize Firebase with credentials from environment or default location
    if os.path.exists('firebase-credentials.json'):
        cred = firebase_admin.credentials.Certificate('firebase-credentials.json')
        firebase_admin.initialize_app(cred)
    else:
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

# Updated models that are currently available (as of May 2025)
GRAMMAR_DETECTION_MODEL = "vennify/t5-base-grammar-correction"
GRAMMAR_CORRECTION_MODEL = "grammarly/coedit-large"

def extract_grammar_issues(text: str, corrected_text: str) -> list:
    """Extract grammar issues from original and corrected text."""
    issues = []
    
    # Simple difference detection (this is a basic approach, can be improved)
    if text == corrected_text:
        return issues
    
    # Find words that differ between original and corrected
    original_words = text.split()
    corrected_words = corrected_text.split()
    
    # Use difflib for better diff detection
    import difflib
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete', 'insert'):
            # Found a difference
            original_phrase = ' '.join(original_words[i1:i2]) if i1 < i2 else ''
            corrected_phrase = ' '.join(corrected_words[j1:j2]) if j1 < j2 else ''
            
            # Determine the type of issue (improved version)
            issue_type = "grammar"
            if original_phrase.lower() == corrected_phrase.lower():
                issue_type = "punctuation"
            elif (original_phrase and corrected_phrase and 
                  abs(len(original_phrase) - len(corrected_phrase)) <= 2):
                issue_type = "spelling"
            elif not original_phrase or not corrected_phrase:
                issue_type = "word_choice"
                
            # Calculate approximate positions in the original text
            start_pos = text.find(original_phrase) if original_phrase else 0
            end_pos = start_pos + len(original_phrase) if original_phrase else 0
            
            if start_pos < 0:  # If exact phrase not found, make an estimate
                # Find approximate position based on word index
                words_before = ' '.join(original_words[:i1])
                start_pos = len(words_before) + 1 if words_before else 0
                end_pos = start_pos + len(original_phrase) if original_phrase else start_pos
            
            # Create issue object
            issue = {
                "original": original_phrase,
                "suggestion": corrected_phrase,
                "type": issue_type,
                "explanation": f"'{original_phrase}' should be '{corrected_phrase}'" if original_phrase and corrected_phrase else 
                              (f"Remove '{original_phrase}'" if original_phrase else f"Add '{corrected_phrase}'"),
                "startIndex": start_pos,
                "endIndex": end_pos
            }
            issues.append(issue)
    
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
        elif issue["type"] == "word_choice":
            score -= 4
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
        
        corrected_text = text  # Default to original if all attempts fail
        
        # Try multiple models with fallbacks
        try:
            # First try with the correction model
            api_url = f"https://api-inference.huggingface.co/models/{GRAMMAR_CORRECTION_MODEL}"
            headers_req = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}
            payload = {"inputs": text}
            
            # Add a timeout to prevent hanging
            logging.info(f"Trying primary correction model: {GRAMMAR_CORRECTION_MODEL}")
            response = requests.post(api_url, headers=headers_req, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Correction result: {result}")
                
                # Extract corrected text based on model response format
                if isinstance(result, list) and result and "generated_text" in result[0]:
                    corrected_text = result[0]["generated_text"]
                elif isinstance(result, dict) and "generated_text" in result:
                    corrected_text = result["generated_text"]
                else:
                    logging.warning(f"Unexpected response format: {result}")
                
                logging.info(f"Corrected text: {corrected_text[:50]}{'...' if len(corrected_text) > 50 else ''}")
            else:
                # If the first model fails, try the fallback model
                raise Exception(f"Primary model API error: {response.status_code}")
                
        except Exception as e:
            logging.info(f"Using fallback correction model: {str(e)}")
            try:
                # Try with the fallback T5 model
                api_url = f"https://api-inference.huggingface.co/models/{GRAMMAR_DETECTION_MODEL}"
                headers_req = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}
                payload = {"inputs": text, "parameters": {"max_new_tokens": 250}}
                
                logging.info(f"Trying fallback model: {GRAMMAR_DETECTION_MODEL}")
                response = requests.post(api_url, headers=headers_req, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    logging.info(f"Fallback result: {result}")
                    
                    # Extract corrected text based on expected T5 model response format
                    if isinstance(result, list) and result:
                        corrected_text = result[0]["generated_text"]
                    elif isinstance(result, dict) and "generated_text" in result:
                        corrected_text = result["generated_text"]
                    else:
                        logging.warning(f"Unexpected fallback response format: {result}")
                else:
                    logging.error(f"Fallback model API error: {response.status_code}, {response.text}")
                    # If all models fail, implement a basic rule-based correction as last resort
                    corrected_text = basic_grammar_correction(text)
            except Exception as e2:
                logging.error(f"All correction attempts failed: {str(e2)}")
                # If all API attempts fail, try a basic rule-based approach
                corrected_text = basic_grammar_correction(text)
        
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
                doc_ref = db.collection('grammar_check_results').document()
                doc_ref.set({
                    'original_text': text,
                    'corrected_text': corrected_text,
                    'issues': issues,
                    'score': score,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                doc_id = doc_ref.id
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


def basic_grammar_correction(text):
    """
    Basic rule-based grammar correction as a fallback when API fails.
    This is very limited but better than nothing.
    """
    corrections = [
        ("have grammar mistake", "has grammar mistake"),
        ("have a grammar mistake", "has a grammar mistake"),
        ("have many mistake", "have many mistakes"),
        ("have many mistakes in it", "has many mistakes in it"),
        ("i think", "I think"),
        ("i am", "I am"),
        ("i will", "I will"),
        ("writed", "wrote"),
        ("grammer", "grammar"),
        ("sentance", "sentence"),
        ("there mistake", "their mistake"),
        ("alot", "a lot"),
    ]
    
    corrected = text
    for original, fixed in corrections:
        corrected = corrected.replace(original, fixed)
    
    # Fix missing period at end of sentence
    if corrected and not corrected.endswith(('.', '!', '?')):
        corrected += '.'
        
    return corrected


# Main entry point for Cloud Run
if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    # Run the Flask application
    app.run(host="0.0.0.0", port=port, debug=True)