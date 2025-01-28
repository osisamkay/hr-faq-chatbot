# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt . 

# Install dependencies and spaCy model
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md

# Set environment variables for caching and avoiding permission issues
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache

# Preload the SentenceTransformer model to reduce runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application code
COPY . .

# Expose the port the application will run on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
