FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and credentials
COPY main.py .
COPY firebase-credentials.json .

# Set environment variable for Firebase credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=firebase-credentials.json

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app