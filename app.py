# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import openai
import os
import redis
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app, origins="*")

# Get the Redis URL from the environment or default to a local Redis instance
# redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create a Redis client
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)


# Test the connection (optional)
try:
    redis_client.ping()
    print("Connected to Redis!")
except redis.ConnectionError as e:
    print(f"Failed to connect to Redis: {e}")

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use GPU for spaCy if available
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

# Load the transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths to HR policy PDFs
pdf_folder = "./pdf_files"
pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# # Example FAQ database
# faqs = {
#     "What is the vacation policy?": "Our vacation policy allows 20 days of paid leave annually.",
#     "How can I access my pension?": "You can access your pension details via the HR portal.",
#     "What are the working hours?": "Working hours are from 9 AM to 5 PM, Monday to Friday.",
# }

# Advanced PDF search using BERT embeddings
def search_pdfs_advanced(query, pdf_paths):
    results = []
    query_embedding = model.encode(query, convert_to_tensor=True)

    for pdf_path in pdf_paths:
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            continue

        # Split content into chunks for better matching
        chunks = list(chunk_text(pdf_text, chunk_size=300))
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

        # Calculate similarity scores
        scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)

        # Extract top matches
        top_k = 5  # Number of matches to return
        top_results = sorted(zip(scores[0], chunks), key=lambda x: x[0], reverse=True)[:top_k]

        results.append({
            "pdf": pdf_path,
            "matches": [{"score": float(score), "text": chunk} for score, chunk in top_results]
        })

    return results

# Chunk text dynamically
def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Fuzzy matching for FAQs
# def match_faq(query):
#     question, score = process.extractOne(query, faqs.keys())
#     if score > 80:  # Match threshold
#         return faqs[question]
#     return None

# Query OpenAI GPT-3 with context
def gpt_response_with_context(prompt, user_input):
    """
    Generates a GPT response based on the provided prompt and caches the result.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        gpt_response = response['choices'][0]['message']['content'].strip()
        redis_client.set(user_input, gpt_response)  # Cache the response
        return jsonify({"response": gpt_response})
    except openai.error.OpenAIError as e:
        # Handle OpenAI API errors gracefully
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message")
    username = request.json.get("username", "guest")
    
    # Default user profile for personalization
    user_profile = {"employment_status": "guest"}

    try:
        # Check Redis cache for a pre-existing response
        cached_response = redis_client.get(user_input)
        if cached_response:
            return jsonify({"response": cached_response.decode("utf-8")})
        
        # Attempt to match the query with FAQs
        # faq_response = match_faq(user_input)
        # if faq_response:
        #     redis_client.set(user_input, faq_response)  # Cache the response
        #     return jsonify({"response": faq_response})
        
        # Perform PDF-based search
        pdf_response = search_pdfs_advanced(user_input, pdf_paths)
        if pdf_response:
            # Extract context from PDF matches
            context = "\n".join([match["text"] for match in pdf_response[0]["matches"]])
            prompt = f"Based on the following context, answer the question: {user_input}\n\n{context}"
            
            # Query OpenAI GPT for a response
            return gpt_response_with_context(prompt, user_input)
        
        # Fall back to a general GPT response if no PDF match is found
        fallback_prompt = f"""
        You are an HR assistant. A user with the profile {user_profile} asked: {user_input}.
        Answer based on HR policies and the user's profile.
        """
        return gpt_response_with_context(fallback_prompt, user_input)
    
    except Exception as e:
        # Catch unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files["file"]
    if file.filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file.filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully!", "path": file_path})
    return jsonify({"error": "Only PDF files are allowed"}), 400

@app.route("/user_profile/<username>", methods=["GET"])
def get_user_profile(username):
    user_profile = {"employment_status": "guest"}  # Example profile
    return jsonify(user_profile)

if __name__ == "__main__":
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
    app.run(host="0.0.0.0", port=10000)
