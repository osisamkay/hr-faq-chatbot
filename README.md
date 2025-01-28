# **Interactive HR FAQ Chatbot with PDF Search**

An AI-powered HR FAQ chatbot designed to assist employees with quick answers to policy-related questions. The application combines a hybrid search approach using PDF documents and OpenAI's language models, making it both efficient and user-friendly.

---

## **Features**

- **FAQ Query Handling**:
   - Quickly answers frequently asked HR questions from predefined FAQs.

- **PDF Search**:
   - Searches and extracts relevant information from uploaded HR policy documents and handbooks.

- **Redis Caching**:
   - Stores frequently requested responses to reduce latency and improve performance.

- **OpenAI Integration**
   - Leverages GPT-3.5-turbo for handling fallback responses and conversational queries.

- **Scalable and Efficient**:
   - Built using FastAPI (or Flask for older versions), designed to run in a containerized environment for scalability.

---

## **Technologies Used**

### **Backend**
- FastAPI for the backend API.
- Redis for caching frequently requested responses.
- Hugging Face SentenceTransformer for semantic search.
- spaCy for natural language processing.
- OpenAI GPT-3.5-turbo for generating contextual responses.
- PyPDF2 for extracting text from uploaded PDF documents.
- Docker for containerized deployment.

### **Frontend**
- React
- Material UI
- Axios (for API requests)

---

## **Setup and Installation**

### **Prerequisites**
1. **Python**: Version 3.8 or later.
2. **Node.js**: Version 16 or later.
3. OpenAI API Key.
4. Required Python libraries (listed in `requirements.txt`).

---

### **Backend Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/osisamkay/hr-faq-chatbot.git
   cd hr-faq-chatbot
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy language model:
   ```bash
   python -m spacy download en_core_web_md
   ```
5. Set Up Environment Variables
   ```bash
   OPENAI_API_KEY=<your_openai_api_key>
   REDIS_URL=redis://localhost:6379/0
   ```

6. Update the `pdf_paths` in `app.py` with the paths to your HR policy documents.

7. Start the Flask server:
   ```bash
   python app.py
   ```

---

### **Frontend Setup**

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

---

## **API Endpoints**

### **Chatbot Endpoint**
**POST** `/chatbot`

**Request Body**:
```json
{
  "message": "What is the maternity leave policy?",
  "username": "john_doe"
}
```

**Response**:
- If the answer is found in PDFs:
  ```json
  {
    "response": "Found in 'policy1.pdf': Employees are entitled to 20 days of maternity leave."
  }
  ```
- If fallback to OpenAI GPT is used:
  ```json
  {
    "response": "Employees are entitled to 20 days of maternity leave, as per company policy."
  }
  ```

---

## **Project Structure**

```
hr-faq-chatbot/
├── app.py                     # Flask backend
├── requirements.txt           # Python dependencies
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.js             # Main React component
│   │   ├── index.js           # Entry point for React
│   ├── public/                # Static files
├── README.md                  # Project documentation
```

---

## **Demo and Deployment**

### **Run Locally**
1. Start the Flask backend:
   ```bash
   python app.py
   ```
2. Start the React frontend:
   ```bash
   npm start
   ```
3. Access the application at `http://localhost:3000`.

### **Deploying**
- **Backend**: Use platforms like Heroku, AWS, or Google Cloud.
- **Frontend**: Deploy using Vercel, Netlify, or Firebase Hosting.

---

## **Future Improvements**
1. **Authentication**: Add user authentication for secure and personalized access.
3. **Admin Dashboard**: Allow HR teams to upload and manage policies dynamically.
4. **Multi-language Support**: Extend functionality for employees in different regions.

---

## **Contributing**
Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- OpenAI for providing powerful language models.
- spaCy for robust natural language processing.
- Material UI for a sleek and modern UI design.

---
