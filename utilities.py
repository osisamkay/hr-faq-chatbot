from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import openai
import os
import redis
from fastapi import HTTPException

# Load the sentence transformer model for semantic search
# 'all-MiniLM-L6-v2' is lightweight and effective for most use cases.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the folder where HR policy PDFs are stored
pdf_folder = "./pdf_files"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)  # Create the folder if it doesn't exist

# List all PDF file paths in the folder
pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Get the Redis URL from environment variables, defaulting to a local Redis instance
redis_url = os.getenv("REDIS_URL")

# Initialize the Redis client
redis_client = redis.StrictRedis.from_url(redis_url)

# Test the Redis connection (optional, for debugging purposes)
try:
    redis_client.ping()
    print("Connected to Redis!")
except redis.ConnectionError as e:
    print(f"Failed to connect to Redis: {e}")

# Utility Functions

def search_pdf_advanced(query, pdf_paths):
    """
    Search for a query across multiple PDF documents using semantic search.

    Args:
        query (str): The search query entered by the user.
        pdf_paths (list): List of paths to the PDF files.

    Returns:
        list: A list of dictionaries containing the PDF file path and top matching chunks.
    """
    results = []
    query_embedding = model.encode(query, convert_to_tensor=True)  # Encode the query for semantic search

    for pdf_path in pdf_paths:
        # Extract text from the PDF file
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            continue  # Skip empty PDFs

        # Chunk the extracted text for comparison
        chunks = list(chunk_text(pdf_text, chunk_size=300))
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

        # Compute semantic similarity scores between the query and text chunks
        scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)

        # Select the top K matching chunks
        top_k = 5  # Number of matches to return
        top_results = sorted(zip(scores[0], chunks), key=lambda x: x[0], reverse=True)[:top_k]

        # Append results with scores and text chunks
        results.append({
            "pdf": pdf_path,
            "matches": [{"score": float(score), "text": chunk} for score, chunk in top_results]
        })

    return results

def chunk_text(text, chunk_size=300):
    """
    Divide a large text into smaller chunks of a specified size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum number of words in each chunk.

    Yields:
        str: A chunk of the original text.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF file, or an empty string if extraction fails.
    """
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
    """
    Generate a response using the OpenAI GPT API with the given context.

    Args:
        prompt (str): The prompt or context to provide to the GPT model.
        user_input (str): The user's original query.

    Returns:
        dict: A dictionary containing the generated response.

    Raises:
        HTTPException: If the OpenAI API returns an error.
    """
    try:
        # Call the OpenAI API to get a GPT-based response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # GPT model used for completion
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,  # Limit the token count for concise responses
        )
        gpt_response = response['choices'][0]['message']['content'].strip()

        # Cache the response in Redis for faster future retrieval
        redis_client.set(user_input, gpt_response)
        return {"response": gpt_response}

    except openai.error.OpenAIError as e:
        # Raise an HTTP exception with a detailed error message if OpenAI API fails
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
