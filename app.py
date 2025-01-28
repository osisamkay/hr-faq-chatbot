from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import spacy
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import openai
import os
import redis
from typing import Optional

# FastAPI app setup
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the Redis URL from the environment or default to a local Redis instance
redis_url = os.getenv("REDIS_URL")

# Create a Redis client
redis_client = redis.StrictRedis.from_url(redis_url)

# Test the connection to Redis (optional)
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
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Utility Functions
def search_pdfs_advanced(query, pdf_paths):
    results = []
    query_embedding = model.encode(query, convert_to_tensor=True)

    for pdf_path in pdf_paths:
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            continue

        chunks = list(chunk_text(pdf_text, chunk_size=300))
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)

        top_k = 5  # Number of matches to return
        top_results = sorted(zip(scores[0], chunks), key=lambda x: x[0], reverse=True)[:top_k]

        results.append({
            "pdf": pdf_path,
            "matches": [{"score": float(score), "text": chunk} for score, chunk in top_results]
        })

    return results

def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

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

def gpt_response_with_context(prompt, user_input):
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
        return {"response": gpt_response}
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# Request Models
class ChatRequest(BaseModel):
    message: str
    username: Optional[str] = "guest"

# Routes
@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    user_input = request.message
    username = request.username

    # Default user profile for personalization
    user_profile = {"employment_status": "guest"}

    try:
        # Check Redis cache
        cached_response = redis_client.get(user_input)
        if cached_response:
            return {"response": cached_response.decode("utf-8")}

        # Perform PDF-based search
        pdf_response = search_pdfs_advanced(user_input, pdf_paths)
        if pdf_response:
            context = "\n".join([match["text"] for match in pdf_response[0]["matches"]])
            prompt = f"Based on the following context, answer the question: {user_input}\n\n{context}"
            return gpt_response_with_context(prompt, user_input)

        # Fall back to general GPT response
        fallback_prompt = f"""
        You are an HR assistant. A user with the profile {user_profile} asked: {user_input}.
        Answer based on HR policies and the user's profile.
        """
        return gpt_response_with_context(fallback_prompt, user_input)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"message": "File uploaded successfully!", "path": file_path}
    raise HTTPException(status_code=400, detail="Only PDF files are allowed")

@app.get("/user_profile/{username}")
async def get_user_profile(username: str):
    user_profile = {"employment_status": "guest"}  # Example profile
    return user_profile

# Serve static files (e.g., PDF files)
app.mount("/static", StaticFiles(directory=pdf_folder), name="static")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
