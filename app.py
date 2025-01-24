# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import pdfplumber
from PyPDF2 import PdfReader
import openai
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for the frontend

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

# Set OpenAI API Key

openai.api_key = os.getenv("OPENAI_API_KEY")

# Example in-memory database for demo purposes
users = {
    "john_doe": {"employment_status": "full-time", "vacation_days": 20},
    "jane_smith": {"employment_status": "part-time", "vacation_days": 10},
}
# Define paths to your policy PDFs
pdf_paths = [
    "./07_pay_benefits_and_leave_policy.pdf",
]

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


# Function to search for an answer in the PDF text# Function to search PDFs for answers
def search_pdfs_for_question(pdf_paths, question):
    """Search multiple PDFs for the given question."""
    question_doc = nlp(question)
    best_match = None
    best_score = 0
    best_file = None

    for pdf_path in pdf_paths:
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            continue

        # Process PDF text with spaCy
        pdf_doc = nlp(pdf_text)
        sentences = [sent.text for sent in pdf_doc.sents]
        similarity_scores = [question_doc.similarity(nlp(sent)) for sent in sentences]

        if similarity_scores:
            file_best_score = max(similarity_scores)
            if file_best_score > best_score:
                best_score = file_best_score
                best_match = sentences[similarity_scores.index(file_best_score)]
                best_file = os.path.basename(pdf_path)

    if best_match and best_score > 0.7:  # Adjust threshold as needed
        return f"Found in '{best_file}': {best_match}"
    return "I couldn't find a relevant match in the policy documents."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Endpoint for Interactive FAQ chatbot."""
    user_input = request.json.get("message")
    username = request.json.get("username", "guest")

    # Retrieve user profile for personalization
    user_profile = {"employment_status": "guest"}  # Example profile for demo

    # Search PDFs for relevant answers
    pdf_response = search_pdfs_for_question(pdf_paths, user_input)

    if "I couldn't find" not in pdf_response:
        # If the PDF search found a match, return it
        return jsonify({"response": pdf_response})
    else:
        # If no match was found, query OpenAI GPT for a response
        prompt = f"""
        You are an HR assistant. A user with the profile {user_profile} asked: {user_input}.
        Answer based on HR policies and the user's profile.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use the newer model
                messages=[
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )
            return jsonify({"response": response['choices'][0]['message']['content'].strip()})
        except openai.error.OpenAIError as e:
            return jsonify({"error": str(e)}), 500




@app.route("/document_search", methods=["POST"])
def document_search():
    """Endpoint for searching HR documents."""
    query = request.json.get("query")

    # Search for relevant documents using a simple match
    results = []
    for doc in hr_documents:
        if query.lower() in doc["content"].lower():
            results.append({"title": doc["title"], "content": doc["content"]})

    if results:
        summaries = []
        for result in results:
            summary_prompt = f"Summarize the following HR policy: {result['content']}"
            messages = [
                {"role": "system", "content": "You are a helpful assistant for summarizing HR policies."},
                {"role": "user", "content": summary_prompt}
            ]
            summary_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100,
                temperature=0.5
            )
            summaries.append(summary_response['choices'][0]['message']['content'].strip())

        return jsonify({"results": summaries})
    else:
        return jsonify({"results": "No relevant documents found."})



@app.route("/user_profile/<username>", methods=["GET"])
def get_user_profile(username):
    """Endpoint for personalized dashboard."""
    user_profile = users.get(username)
    if user_profile:
        return jsonify(user_profile)
    else:
        return jsonify({"error": "User not found"}), 404


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files["file"]
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    return jsonify({"message": "File uploaded successfully!", "path": file_path})


if __name__ == "__main__":
    app.run(debug=True)
