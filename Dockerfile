# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_md

# Copy the application code
COPY . .

# Expose the port Flask runs on
EXPOSE 10000

# Command to run the application
CMD ["python", "app.py"]
