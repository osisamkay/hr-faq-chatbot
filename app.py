from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utilities import search_pdf_advanced, pdf_folder, pdf_paths, redis_client, gpt_response_with_context
import spacy
import openai
import os
from typing import Optional

# FastAPI app setup
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
# This is useful for front-end applications hosted on different domains.
# In production, restrict `allow_origins` to specific domains for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API Key
# This key is required to access OpenAI's GPT models for generating responses.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use GPU for spaCy if available, to speed up NLP tasks
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")  # Load the medium-sized English spaCy model

# Request Models
class ChatRequest(BaseModel):
    """
    Represents the request model for chatbot interactions.
    - `message` (str): The user's input message.
    - `username` (Optional[str]): The username of the user (default: "guest").
    """
    message: str
    username: Optional[str] = "guest"

# Routes

@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    """
    Handles chatbot interactions.

    Args:
        request (ChatRequest): The input from the user, including their message and optional username.

    Returns:
        dict: The response from the chatbot, either retrieved from Redis cache, a PDF document, or a GPT-generated fallback.
    """
    user_input = request.message
    username = request.username

    # Default user profile for personalization
    user_profile = {"employment_status": "guest"}

    try:
        # Check Redis cache for a previously saved response
        cached_response = redis_client.get(user_input)
        if cached_response:
            return {"response": cached_response.decode("utf-8")}

        # Perform a PDF-based search for relevant content
        pdf_response = search_pdf_advanced(user_input, pdf_paths)
        if pdf_response:
            # Extract relevant context from the matched PDF content
            context = "\n".join([match["text"] for match in pdf_response[0]["matches"]])
            prompt = f"Based on the following context, answer the question: {user_input}\n\n{context}"
            return gpt_response_with_context(prompt, user_input)

        # Fall back to a general GPT response when no PDF match is found
        fallback_prompt = f"""
        You are an HR assistant. A user with the profile {user_profile} asked: {user_input}.
        Answer based on HR policies and the user's profile.
        """
        return gpt_response_with_context(fallback_prompt, user_input)

    except Exception as e:
        # Handle unexpected errors gracefully
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload PDF files for the chatbot's document-based search feature.

    Args:
        file (UploadFile): The uploaded file.

    Returns:
        dict: A message indicating the success or failure of the upload.

    Raises:
        HTTPException: If the uploaded file is not a PDF.
    """
    if file.filename.endswith(".pdf"):
        # Save the uploaded PDF to the designated folder
        file_path = os.path.join(pdf_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"message": "File uploaded successfully!", "path": file_path}
    raise HTTPException(status_code=400, detail="Only PDF files are allowed")


@app.get("/user_profile/{username}")
async def get_user_profile(username: str):
    """
    Endpoint to retrieve a user profile.

    Args:
        username (str): The username of the user.

    Returns:
        dict: A user profile with basic attributes.
    """
    user_profile = {"employment_status": "guest"}  # Example profile for all users
    return user_profile


# Serve static files (e.g., PDF files) for access via the `/static` route
app.mount("/static", StaticFiles(directory=pdf_folder), name="static")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
